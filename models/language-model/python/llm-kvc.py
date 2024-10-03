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
import copy

# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import utils

from concurrent.futures import ThreadPoolExecutor, as_completed

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
    parser.add_argument('--verbose', action = 'store_true', default = False,
                        help = 'Enable onnxruntime logs')

    # options to be validated
    parser.add_argument('--use-kvc', action="store_true", default=True,
                        help = 'Enables KV-cache')
    parser.add_argument("-l", '--num-layers', type = int, default = 32,
                        help = 'The number of layers of the model')
    args = parser.parse_args()
    return args

def is_odd(number):
    return number % 2 != 0

def is_even(number):
    return not is_odd(number)

def get_device_placeholders(args, sep=';'):
    dph=""
    for id in range(args.num_layers):
        dph+=f"{sep}implicit-placeholder=past_key_values.{id}.key"
        dph+=f"{sep}implicit-placeholder=past_key_values.{id}.value"
        dph+=f"{sep}placeholder=present.{id}.key"
        dph+=f"{sep}placeholder=present.{id}.value"
    return dph

# Define etglow api parameters
def get_etglow_api_params(args, sep=';'):
    dev_params = " ".join(['--gccCompileThreads=32','--logDisableCodeGenBits=-1', ])
    extra_params = '|'.join(['debug-glow=0', f'dev={dev_params}'])
    api_params = [
        "device-type=silicon",
        "glow-threads=2",
        "runDir=myrundir",
        f"extra-etsoc-params='{extra_params}'"
    ]
    api_params = sep.join(api_params)
    if args.use_kvc:
        api_params += get_device_placeholders(args, sep)
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
        "sequence": 1, # this is the window
        "past_sequence": sl-1,
        "past_sequence+sequence": sl, # sequence_len (past_sequence + window)
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

def get_provider_options(args, onnx_symbols) -> dict:
    onnx_shape_params = get_onnx_shape_params(onnx_symbols)
    api_params = get_etglow_api_params(args)

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

        logits = outs_dictionary['logits']

        # Prepare next inference inputs
        for name in inputs_names:
            if name == 'input_ids':
                current_index = next_index
                next_index = update_next_index(next_index, prompt_size, window)
                j = next_index - window

                if current_index >= prompt_size:
                    top1 = logits.argmax(-1)
                    new_token = top1.reshape(1, window) # Inf server
                    total_input = np.concatenate((total_input, new_token[: , -1:]), axis = 1)

                input_tensors['input_ids'] = total_input[:, j:next_index].reshape(1, window) # inf server
            elif name == 'attention_mask':
                attention_mask = np.zeros((1, sequence_len), dtype='int64')
                attention_mask[0, -next_index:] = 1
                input_tensors['attention_mask'] = attention_mask  # inf server
            else:
                old_name = name.replace("past_key_values", "present")
                start_idx = next_index - current_index
                end_idx = start_idx + context - window
                input_tensors[name] = outs_dictionary[old_name][:, :, start_idx:end_idx, :]

        sum_perplexity += -np.log(softmax(logits[0, 0])[input_tensors['input_ids'][0]])

    sum_perplexity /= (num_tokens - 1)
    answer = tokenizer.decode(total_input[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return answer, sum_perplexity


def llm_kvc_inference_iobindings(session : onnxruntime.InferenceSession, tokenizer : AutoTokenizer, input_tensors : dict,
                                 prompt_tensor, num_tokens, context, sequence_len, window : int, batch=1, nheads=8,
                                 hidden=128) -> tuple[str, float]:
    sum_perplexity = 0
    new_token = np.array([10])
    prompt_size = len(prompt_tensor[0])
    total_input = prompt_tensor.numpy()
    current_index = 0
    next_index = window

    inputs_names = [input.name for input in session.get_inputs()]
    outputs_names = [input.name for input in session.get_outputs()]

    # Pre-allocate inputs
    ortvalues = preallocate_input_outputs(outputs_names, input_tensors, window, batch, nheads, context, hidden)

    # Create IOBindings
    io_binding = session.io_binding()
    bind_input_outputs(io_binding, inputs_names, outputs_names, ortvalues, batch, nheads, context, hidden)

    with ThreadPoolExecutor() as executor:
        futures = []

        # Run the inferences
        while next_index < num_tokens:
            if (new_token == tokenizer.eos_token_id).any():
                break

            session.run_with_iobinding(io_binding)

            logits = ortvalues['logits'].numpy()

            # Prepare next inference inputs
            for name in inputs_names:
                if name == 'input_ids':
                    current_index = next_index
                    next_index = update_next_index(next_index, prompt_size, window)
                    j = next_index - window

                    if current_index >= prompt_size:
                        top1 = logits.argmax(-1)
                        new_token = top1.reshape(1, window) # Inf server
                        total_input = np.concatenate((total_input, new_token[: , -1:]), axis = 1)

                    next_input_ids = total_input[:, j:next_index]
                    ortvalues['input_ids'].update_inplace(np.ascontiguousarray(next_input_ids))
                elif name == 'attention_mask':
                    attention_mask = np.zeros((1, sequence_len), dtype='int64')
                    attention_mask[0, -next_index:] = 1
                    ortvalues['attention_mask'].update_inplace(attention_mask)
                else:
                    update_kvc_view(io_binding, name, ortvalues, current_index, batch, nheads, context, hidden)

            # Offload perplexity calculation to a separate thread
            futures.append(executor.submit(compute_perplexity, logits[0, 0], next_input_ids[0]))

        # Accumulate perplexity results from helper thread
        for future in as_completed(futures):
            sum_perplexity += future.result()

    sum_perplexity /= (num_tokens - 1)
    answer = tokenizer.decode(total_input[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return answer, sum_perplexity


# Function to calculate perplexity
def compute_perplexity(logits, next_input_ids):
    # Perform the softmax and perplexity calculation
    return -np.log(softmax(logits)[next_input_ids])

def preallocate_input_outputs(output_names, input_tensors: dict, window=1, batch=1, nheads=8, context=2048, hidden=128,
                              logits_last_dim=128256, device_type="et", device_id=0) -> dict:
    ortvalues = {
        'input_ids':      onnxruntime.OrtValue.ortvalue_from_numpy(input_tensors['input_ids'], device_type, device_id),
        'attention_mask': onnxruntime.OrtValue.ortvalue_from_numpy(input_tensors['attention_mask'], device_type, device_id),
        'logits':      onnxruntime.OrtValue.ortvalue_from_shape_and_type((batch, window, logits_last_dim), np.float16, device_type, device_id),
    }
    # All KVC input (past_key_value) & output (present) tensors will share same underlying allocation.
    zeros = np.zeros((batch, nheads, context, hidden), dtype=np.float16)
    zeros_padded = np.pad(zeros, ((0,0), (0,1), (0, 0), (0,0)), mode='constant')
    for name in output_names:
        if 'present' in name:
            ortvalues[name] = onnxruntime.OrtValue.ortvalue_from_numpy(zeros_padded, device_type, device_id)
    return ortvalues


def bind_input_outputs(io_binding, inputs_names, outputs_names, ortvalues, batch=1, nheads=8, context=2048, hidden=128):
    # Bind inputs (will bind them to the allocated memory)
    for name in inputs_names:
        if name in ['input_ids', 'attention_mask']:
            # For input_ids or attention_mask lets bind the ortvalue directly
            io_binding.bind_ortvalue_input(name, ortvalues[name])
        else:
            # For 'past_key_value' we need to bind the buffer_ptr to the underlying allocation
            out_name = name.replace("past_key_values", "present")
            io_binding.bind_input(name,
                                  device_type=ortvalues[out_name].device_name(),
                                  device_id=0,
                                  element_type=np.float16,
                                  shape=(batch, nheads, context - 1, hidden),
                                  buffer_ptr=ortvalues[out_name].data_ptr())
    # Bind outputs (to non pre-allocated output)
    for name in outputs_names:
        if 'logits' in name:
            io_binding.bind_ortvalue_output(name, ortvalues[name])
        else:
            io_binding.bind_output(name,
                                   device_type=ortvalues[name].device_name(),
                                   device_id=0,
                                   element_type=np.float16,
                                   shape=(batch, nheads, context, hidden),
                                   buffer_ptr=ortvalues[name].data_ptr())

def update_kvc_view(io_binding, name, ortvalues, current_index, batch, nheads, context, hidden):
    out_name = name.replace("past_key_values", "present")
    last_dim_value = hidden
    element_size_bytes = 2
    num_bytes_per_cache_line = last_dim_value * element_size_bytes
    offset_bytes = current_index * num_bytes_per_cache_line
    buffer_ptr = ortvalues[out_name].data_ptr() + offset_bytes
    device_type = ortvalues[out_name].device_name()
    io_binding.bind_input(name,
                          device_type=device_type,
                          device_id=0,
                          element_type=np.float16,
                          shape=(batch, nheads, context - 1, hidden),
                          buffer_ptr=buffer_ptr)
    io_binding.bind_output(out_name,
                           device_type=device_type,
                           device_id=0,
                           element_type=np.float16,
                           shape=(batch, nheads, context, hidden),
                           buffer_ptr=buffer_ptr)

def update_next_index(next_index, prompt_size, window):
    if next_index < prompt_size:
        # Next token is part of the prompt
        if prompt_size - next_index >= window:
            next_index += window
        else:
            next_index = prompt_size
    else:
        # Next token is part of the answer
        next_index += 1
    return next_index


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
    
    return fixeddim_model_path, n_heads, hidden


def preprocess_llm_input_tensors(input_ids_tensor : np.ndarray, inputs_names, window, pads, n_heads, hidden, context) -> dict:
    inputs_dict = {
        'input_ids': copy.deepcopy(input_ids_tensor[:, :window].reshape(1, window).numpy()),
        'attention_mask': np.concatenate((np.zeros([1, pads - window], dtype = 'int64'), np.ones((1, window), dtype = 'int64')), axis=1)
    }
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
    utils.set_verbose_output(session_options, args.verbose)
    session_options.enable_profiling = args.enable_tracing
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    # Provider options
    onnx_symbols = get_onnx_symbols(context_len, sequence_len, kvc)
    provider_options = get_provider_options(args, onnx_symbols)

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
    input_tensors_cpu = preprocess_llm_input_tensors(prompt_tensor, inputs_names, window, sequence_len, n_heads, hidden, context_len)

    # Execute inference on CPU
    start = time.time()
    answer_cpu, perplexity_cpu = llm_kvc_inference(session_cpu, tokenizer, input_tensors_cpu, prompt_tensor, num_tokens, context_len, sequence_len, window)
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
    input_tensors_etglow = preprocess_llm_input_tensors(prompt_tensor, inputs_names, window, sequence_len, n_heads, hidden, context_len)

    # Launch ETSoC inference
    start = time.time()
    answer_etglow, perplexity_etglow = llm_kvc_inference_iobindings(session_etglow, tokenizer, input_tensors_etglow, prompt_tensor, num_tokens, context_len, sequence_len, window)
    session_etglow.end_profiling()
    inf_time_etsoc = time.time() - start

    print_llm_inference_results('ETGlow EP results', etsoc_comp_time, inf_time_etsoc, perplexity_etglow, answer_etglow, args.generate_tokens)

if __name__ == "__main__":
    main()
