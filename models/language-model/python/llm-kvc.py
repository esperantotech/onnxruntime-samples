#!/usr/bin/env python3
from scipy.special import softmax
import onnxruntime
import onnx
from onnx import numpy_helper
import time
import numpy as np
import argparse
import logging
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed

# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import utils

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Get the perplexity for a prompt and a model")
    parser.add_argument("-m", '--model-path', metavar = "DIR", type = str, default = './model.onnx', 
                        help = 'Path to the model.onnx')
    parser.add_argument("-t", '--tokenizer', metavar = "DIR", type = str, default = '.',
                         help = 'Path to the tokenizer folder')
    parser.add_argument("-p", '--prompt', metavar = "TEXT", type = str, default = 'Who is Messi?',
                        help = 'Given Prompt')
    parser.add_argument("-s", '--sequence-length', type = int, default = 256,
                        help = 'Total sequence length for the run')
    parser.add_argument("-c", '--context', type = int, default = 2048,
                        help = 'Total context length for the run')
    parser.add_argument("-n", '--no-context', action="store_true", default=False, 
                        help = 'If used, the model does not have the context extension in its ONNX')
    parser.add_argument("-g", '--generate-tokens', type = int, default = 0,
                        help = 'Total sequence length for the run')
    parser.add_argument('--enable_tracing', action = 'store_true', default = False,
                        help = 'Enable onnxruntime profiling and neuralizer traces')
    args = parser.parse_args()
    return args

# Define etglow api parameters
def get_etglow_api_params(enable_tracing):
    dev_params = " ".join(['--gccCompileThreads=32','--logDisableCodeGenBits=-1', ])
    extra_params = '|'.join(['debug-glow=0', f'dev={dev_params}'])
    api_params = [
        "device-type=silicon",
        "glow-threads=2",
        "runDir=myrundir",
        f"extra-etsoc-params='{extra_params}'"
    ]
    api_params = ';'.join(api_params)

    return api_params

# Defines dictionaries of onnx symbols
def get_onnx_symbols(ct, sl, kvc) -> dict[str, any]:
    if not kvc:
        symbols = {
            "batch": 1,
            "sequence": sl
        }
    else:
        symbols = {
        "batch": 1,
        "sequence": 1,
        "past_sequence": sl-1,
        "past_sequence+sequence": sl,
        "context": ct,
        "context-sequence": ct-1
    }
    return symbols

# Define onnx shape parameters
def get_onnx_shape_params(onnx_symbols):
    onnx_shape_params = ''
    for key, value in onnx_symbols.items():
        onnx_shape_params += f"{key}={value},"
    # Return generated string without the last comma character
    return onnx_shape_params[:-1]

def get_provider_options(onnx_symbols, enable_tracing) -> dict:
    onnx_shape_params = get_onnx_shape_params(onnx_symbols)
    api_params = get_etglow_api_params(enable_tracing)
    print(api_params)    

    poptions = {"etglow_greedy": "true",
                "etglow_onnx_shape_params": onnx_shape_params,
                "etglow_api_params": api_params}
    
    return poptions

def llm_kvc_inference(session : onnxruntime.InferenceSession, tokenizer : AutoTokenizer, input_tensors : dict,
                      prompt_tensor, num_tokens, context, sequence_len, window : int) -> tuple[str, float]:
    sum_perplexity = 0
    new_token = np.array([10])
    prompt_size = len(prompt_tensor[0])
    total_input = prompt_tensor.numpy()
    current_index = 0
    next_index = window

    output_names = [input.name for input in session.get_outputs()]
    inputs_names = [input.name for input in session.get_inputs()]

    # Run the inferences
    while next_index < num_tokens:
        if (new_token == tokenizer.eos_token_id).any():
            break

        output = session.run(output_names, input_tensors)

        outs_dictionary = {name: content for (name, content) in zip (output_names, output)}

        for name in inputs_names:
            if name == 'input_ids':
                current_index = next_index
                if next_index < prompt_size:
                    # Next token is part of the prompt
                    if prompt_size - next_index >= window :
                        next_index += window
                    else: 
                        next_index = prompt_size 

                    j = next_index - window
                else:
                    # Next token is part of the answer
                    next_index += 1
                    
                    j = next_index - window
                    new_token = outs_dictionary['logits'].argmax(-1).reshape(1, window) # Inf server
                    total_input = np.concatenate((total_input, new_token[: , -1:]), axis = 1)

                input_tensors['input_ids'] = total_input[:, j:next_index].reshape(1, window) # inf server
                sum_perplexity+= -np.log(softmax(outs_dictionary['logits'][0, 0])[input_tensors['input_ids'][0]])
            elif name == 'attention_mask':
                input_tensors['attention_mask'] = np.concatenate((np.zeros((1, sequence_len - next_index), dtype = 'int64'), np.ones((1, next_index), dtype = 'int64')), axis=1) # inf server
            else:
                old_name = name.replace("past_key_values", "present")
                input_tensors[name] = outs_dictionary[old_name][:, :, next_index - current_index:context - window + (next_index - current_index), :]
        

    sum_perplexity /= (num_tokens - 1)
    answer = tokenizer.decode(total_input[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return answer, sum_perplexity

def get_prompt_tensor(prompt, tokenizer):
    input_tensor = tokenizer(prompt, return_tensors="pt")
    return input_tensor['input_ids']


def fix_model_dimensions(model_path : Path, onnx_symbols) -> Path:
    # Fix dimensions in the model
    model_noexternaldata = onnx.load(model_path, load_external_data=False)

    n_heads = int(model_noexternaldata.graph.input[2].type.tensor_type.shape.dim[-3].dim_value)
    hidden = int(model_noexternaldata.graph.input[2].type.tensor_type.shape.dim[-1].dim_value)

    for key, value in onnx_symbols.items():
        make_dim_param_fixed(model_noexternaldata.graph, key, value)

    model_noexternaldata = model_noexternaldata.SerializeToString()
    fixeddim_model_path = model_path.parent / "model-fixed-dims.onnx"
    with open(fixeddim_model_path, "wb") as f:
        f.write(model_noexternaldata)
    
    return (fixeddim_model_path, n_heads, hidden)


def preprocess_llm_input_tensors(input_ids_tensor : np.ndarray, inputs_names, window, pads, n_heads, hidden, context) -> dict:
    inputs_dict = {}
    inputs_dict['input_ids'] = input_ids_tensor[:, :window].reshape(1, window).numpy()
    inputs_dict['attention_mask'] = np.concatenate((np.zeros([1, pads - window], dtype = 'int64'), np.ones((1, window), dtype = 'int64')), axis=1)
    for name in inputs_names:
        if name not in ['input_ids','attention_mask']:
            inputs_dict[name] = np.zeros([1, n_heads, context - window, hidden], dtype="float16")
    
    return inputs_dict


def print_llm_inference_results(label : str, comp_time : float, inf_time : float,  perplexity, answer : str, num_tokens):
    print(f'{label}:')
    print(f'    Compilation took {comp_time:.2f}s')
    print(f'    Inference took {inf_time:.2f}s ({(num_tokens / inf_time):.2f} tokens/s)')
    print(f'    Perplexity value: {float(perplexity):.2f}')
    print(f'    Question and answer: {answer}')


def main():
    args = parse_arguments()
    logging.basicConfig(level = logging.INFO)
  
    # Paths
    model_path = Path(args.model_path)
    tokenizer_path = Path(args.tokenizer)
    sequence_len = args.sequence_length
    context_len = args.context
    prompt = args.prompt
    num_tokens = args.generate_tokens
    if args.no_context:
        context_len = sequence_len
    model_name = str(model_path.parents[0]).replace(str(model_path.parents[1]), '').replace(r'/', '')

    # Check inputs
    if not model_path.exists():
        raise FileNotFoundError(f"Error: model file {model_path} does not exist.")

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Error: tokenizer file {tokenizer_path} does not exist.")
    
    # Check sequence length is long enough
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompt_tensor = get_prompt_tensor(prompt, tokenizer)
    prompt_size = len(prompt_tensor[0])
    assert(prompt_size + num_tokens <= sequence_len), f"Sequence length is not long enough (should be at least: {prompt_size + num_tokens}"

    # Exit if the model does not use KVC
    model = onnx.load(model_path)
    assert(len(model.graph.input) > 2), "Non Key-Value Cache models are not yet supported"
    kvc = True

    # Session options
    session_options = onnxruntime.SessionOptions()
    utils.set_verbose_output(session_options, False)
    session_options.enable_profiling = args.enable_tracing
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    # Provider options
    onnx_symbols = get_onnx_symbols(context_len, sequence_len, kvc)
    provider_options = get_provider_options(onnx_symbols, args.enable_tracing)

    # Fix model dimensions
    fixed_model_path, n_heads, hidden = fix_model_dimensions(model_path, onnx_symbols)

    window = 1
    # Create cpu ORT session
    start = time.time()
    session_options.profile_file_prefix = f'{model_name}_cpu_window_{window}'
    session_cpu = onnxruntime.InferenceSession(fixed_model_path,  sess_options=session_options, providers=['CPUExecutionProvider'])
    comp_time_cpu = time.time() - start

    # Process inputs
    inputs_names = [input.name for input in session_cpu.get_inputs()]
    input_tensors = preprocess_llm_input_tensors(prompt_tensor, inputs_names, window, sequence_len, n_heads, hidden, context_len)

    # Execute inference on CPU
    start = time.time()
    answer_cpu, perplexity_cpu = llm_kvc_inference(session_cpu, tokenizer, input_tensors, prompt_tensor, num_tokens, context_len, sequence_len, window)
    session_cpu.end_profiling()
    inf_time_cpu = time.time() - start

    print_llm_inference_results('CPU EP results', comp_time_cpu, inf_time_cpu, perplexity_cpu, answer_cpu, args.generate_tokens)


    # Create etglow ORT session
    start = time.time()
    session_options.profile_file_prefix = f'{model_name}_etglow_window_{window}'
    session_etglow = onnxruntime.InferenceSession(fixed_model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'], provider_options=[provider_options])
    etsoc_comp_time = time.time() - start 

    # Process inputs
    inputs_names = [input.name for input in session_etglow.get_inputs()]
    input_tensors = preprocess_llm_input_tensors(prompt_tensor, inputs_names, window, sequence_len, n_heads, hidden, context_len)

    # Launch ETSoC inference
    start = time.time()
    answer_etglow, perplexity_etglow = llm_kvc_inference(session_etglow, tokenizer, input_tensors, prompt_tensor, num_tokens, context_len, sequence_len, window)
    session_etglow.end_profiling()
    inf_time_etsoc = time.time() - start

    print_llm_inference_results('ETGlow EP results', etsoc_comp_time, inf_time_etsoc, perplexity_etglow, answer_etglow, args.generate_tokens)

if __name__ == "__main__":
    main()
