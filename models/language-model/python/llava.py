#!/usr/bin/env python3

import torch
import onnx, onnxruntime
from transformers import AutoProcessor, AutoTokenizer
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
import time

# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import utils

def get_provider_options(args):
    api_params = get_api_params()
    
    onnx_shape_params = ';'.join([f"{k}={v}" for k, v in onnx_symbols.items()])

    provider_options = {
            "etglow_greedy": "true", # Forces all ONNX nodes through ETGLOW EP regardless of detected Capabilities
            "etglow_onnx_shape_params": onnx_shape_params,
            "etglow_api_params": api_params
    }
    return provider_options

def extra_arguments(parser):
    parser.add_argument("-i", '--image', type = str, default = './doge.jpg',
                        help = 'Image to analyze')
    parser.add_argument("-p", '--prompt', type = str, default = '',
                        help = 'Query on the given image')
    parser.add_argument("-n", '--new-tokens', type = int, default = 10,
                        help = 'Number of new tokes to generate')

def _merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, labels):
    pad_token_id = -1
    image_token_index = 32000
    ignore_index = -100
    
    num_images, num_image_patches, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape
    left_padding = not np.sum(input_ids[:, -1] == pad_token_id)

    # 1. Create a mask to know where special image tokens are
    special_image_token_mask = input_ids == image_token_index
    num_special_image_tokens = np.sum(special_image_token_mask, axis = -1)
    # Compute the maximum embed dimension
    max_embed_dim = (num_special_image_tokens.max()*(num_image_patches - 1)) + sequence_length
    batch_indices, non_image_indices = np.where(input_ids != image_token_index)

    # 2. Compute the positions where text should be written
    # Calculate new positions for text tokens in merged image-text sequence.
    # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
    # `torch.cumsum` computes how each image token shifts subsequent text token positions.
    # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
    new_token_positions = np.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    if left_padding:
        new_token_positions += nb_image_pad[:, None]  # offset for left padding
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    # 3. Create the full embedding, already padded to the maximum position
    final_embedding = np.zeros(
        (batch_size, max_embed_dim, embed_dim), dtype=inputs_embeds.dtype
    )
    final_attention_mask = np.zeros(
        (batch_size, max_embed_dim), dtype = attention_mask.dtype
    )
    if labels is not None:
        final_labels = np.full(
            (batch_size, max_embed_dim), ignore_index, dtype = input_ids.dtype
        )
    # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
    # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
    if labels is not None:
        final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

    # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
    image_to_overwrite = np.all(final_embedding == 0, axis = -1)
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None]

    # if image_to_overwrite.sum() != image_features.shape[:-1].numel():
    #     print(image_to_overwrite.sum(), '\\n',image_features.shape[:-1].numel())
    #     raise ValueError(
    #         f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
    #         f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
    #     )

    final_embedding[image_to_overwrite] = np.ascontiguousarray(image_features).reshape(-1, embed_dim)
    final_attention_mask |= image_to_overwrite
    position_ids = np.ma.array((final_attention_mask.cumsum(-1) - 1), mask = (final_attention_mask == 0)).filled(fill_value=1)

    if labels is None:
        final_labels = None

    return final_embedding, final_attention_mask, final_labels, position_ids

def run_llama(llama_session, inputs_embeds, attention_mask, embeddings_session, onnx_symbols, max_new_tokens=10):
    res = []
    generated_ids = []
    print("Running LLaMA + Embeddings inference loop", end = " ")
    for _ in range(max_new_tokens):
        print(".", end = " ")
        logits = llama_session.run(None, {'attention_mask': attention_mask, 'inputs_embeds': inputs_embeds})[0]

        next_token_logits = logits[:, -1]
        index_pred = np.argmax(next_token_logits, -1)[0]
        generated_ids.append(index_pred)

        input_ids = np.zeros((1, onnx_symbols["sequence"]), dtype=np.int64)
        input_ids[0,-1] = index_pred
        new_embeds = embeddings_session.run(None, {"input_ids": input_ids})[0]
        inputs_embeds = np.concatenate([inputs_embeds[:,1:,:], new_embeds[:,-1:].astype(np.float16)], 1)
        attention_mask = np.concatenate([attention_mask[:,1:], np.expand_dims(np.ones(1, dtype = np.int64), 0)], 1)
        # position_ids = np.concat([np.expand_dims(position_ids[:,1:], (position_ids[:, -1] + 1), 0)], 1)
    print("")

    return generated_ids

def get_api_params():
    neura_params = " ".join(['--gccCompileThreads=16','--logDisableCodeGenBits=-1'])
    extra_params = '|'.join(['debug-glow=0', f'dev={neura_params}'])
    api_params = ';'.join(["glow-threads=2", "runDir=myrundir", f"extra-etsoc-params='{extra_params}'"])
    return api_params

def run_embeds(embeds_model : Path, tokenized_prompt, execution_provider, times, session_options, poptions):
    session_options.profile_file_prefix = f'embeds_{execution_provider}'

    print(f"Running Embeddings in {execution_provider}")
    input_ids = tokenized_prompt["input_ids"]
    start_time = time.time()
    session = onnxruntime.InferenceSession(embeds_model, sess_option=session_options, providers=[execution_provider], provider_options=[poptions])
    exe_time = time.time()
    outputs = session.run(None, {"input_ids": input_ids})
    end_time = time.time()
    times[0] += (exe_time - start_time)
    times[1] += (end_time - exe_time)

    return outputs, session

def run_clip(clip_model : Path, image : Image,  execution_provider, times, session_options, poptions):
    session_options.profile_file_prefix = f'clip_{execution_provider}'
    print(f"Running CLIP in {execution_provider}")
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf', torch_dtype = torch.float16)
    pixel_values = processor.image_processor(image, return_tensors = 'np')["pixel_values"].astype(np.float16)
    start_time = time.time()
    session_clip = onnxruntime.InferenceSession(clip_model, sess_option=session_options, providers = [execution_provider], provider_options=[poptions])
    exe_time = time.time()
    outputs_clip = session_clip.run(None, {"pixel_values": pixel_values})
    end_time = time.time()
    times[0] += (exe_time - start_time)
    times[1] += (end_time - exe_time)

    return outputs_clip

def run_mmp(proj_model : Path, outputs_clip, execution_provider, times, session_options, poptions):
    session_options.profile_file_prefix = f'mmp_{execution_provider}'
    print(f"Running MultiModal Projection in {execution_provider}")
    start_time = time.time()
    session_proj = onnxruntime.InferenceSession(proj_model,  sess_option=session_options, providers = [execution_provider], provider_options=[poptions])
    exe_time = time.time()
    outputs_proj = session_proj.run(None, {'image_features': outputs_clip[0][:, 1:]})
    end_time = time.time()
    times[0] += (exe_time - start_time)
    times[1] += (end_time - exe_time)

    return outputs_proj

def run_llava(args, execution_provider):
    artifacts = Path(args.artifacts)
    onnx_model_path = artifacts / "models/llava-1.5-7b-fp16/"
    times = [0.0,0.0]
    prompt = args.prompt
    image_path = Path(args.image)
    image = Image.open(image_path).convert('RGB')

    global onnx_symbols
    onnx_symbols = {
        "batch": 1,
        "sequence": 449, # 577 - 1 + 449 - 1 = 1024
        "width": 336,
        "height": 336,
        "channels": 3,
        "image_sequence": 577,
    }

    if execution_provider == 'EtGlowExecutionProvider':
        provider_options = get_provider_options(args)
    else:
        provider_options = {}
    
    print(f'Provider options: {provider_options}')

    sess_options = onnxruntime.SessionOptions()
    utils.set_verbose_output(sess_options, args.verbose)
    sess_options.enable_profiling = args.enable_tracing

    # Tokenize input prompt
    tokenizer = AutoTokenizer.from_pretrained('llava-hf/llava-1.5-7b-hf')
    prompt = 'USER: <image>\\n Here is a collaged image. Give a description for every scene.\\nASSISTANT:'
    tokenized_prompt = tokenizer(prompt, return_tensors = 'np', max_length = onnx_symbols["sequence"], truncation=True, padding="max_length")
    outputs_embeddings, session_embeddings = run_embeds(onnx_model_path / "embeds/model.onnx", tokenized_prompt, execution_provider, times, sess_options, provider_options)

    # Clip image
    outputs_clip = run_clip(onnx_model_path / "clip/model.onnx", image, execution_provider, times, sess_options, provider_options)

    # Update options: remove a token from image-sequence
    if execution_provider == 'EtGlowExecutionProvider':
        onnx_symbols["image_sequence"] = 576
        onnx_shape_params = ';'.join([f"{k}={v}" for k, v in onnx_symbols.items()])
        provider_options["etglow_onnx_shape_params"] = onnx_shape_params
        # print(provider_options)

    # Multimodal projection
    outputs_proj = run_mmp(onnx_model_path / "mmp/model.onnx", outputs_clip, execution_provider, times, sess_options, provider_options)

    # Combine prompt with image features
    inputs_embeds, attention_mask, labels, position_ids = _merge_input_ids_with_image_features(outputs_proj[0], outputs_embeddings[0], tokenized_prompt["input_ids"], tokenized_prompt["attention_mask"], labels = tokenized_prompt["input_ids"])

    # Update options
    if execution_provider == 'EtGlowExecutionProvider':
        onnx_symbols["sequence_with_image"] = attention_mask.shape[-1]
        onnx_shape_params = ';'.join([f"{k}={v}" for k, v in onnx_symbols.items()])
        provider_options["etglow_onnx_shape_params"] = onnx_shape_params
        # print(provider_options)

    start_time = time.time()
    session_llama = onnxruntime.InferenceSession(onnx_model_path / "llama/model.onnx", sess_options, providers=[execution_provider], provider_options=[provider_options])
    exe_time = time.time()
    generated_ids = run_llama(session_llama, inputs_embeds, attention_mask, session_embeddings, onnx_symbols, args.new_tokens)
    end_time = time.time()
    times[0] += (exe_time - start_time)
    times[1] += (end_time - exe_time)

    # print(generated_ids)
    return tokenizer.decode(generated_ids), times

def print_llava_results(execution_provider, answer, times):
    print(f"\n{execution_provider} stats")
    print(f"    Compilation took {times[0]:.2f}s.")
    print(f"    Execution took {times[1]:.2f}s.")
    print(f"    Answer: {answer}")

def main():
    parser = utils.get_common_arg_parser()
    extra_arguments(parser)
    args = parser.parse_args(sys.argv[1:])
    answer_cpu, times_cpu = run_llava(args, 'CPUExecutionProvider')
    answer_etsoc, times_etsoc = run_llava(args, 'EtGlowExecutionProvider')

    os.environ["GLOG_minloglevel"] = "2"
    os.environ["GLOG_logtostderr"] = "2"

    print_llava_results('CPUExecutionProvider', answer_cpu, times_cpu)
    print_llava_results('EtGlowExecutionProvider', answer_etsoc, times_etsoc)

    return 0

if __name__ == "__main__":
    sys.exit(main())
