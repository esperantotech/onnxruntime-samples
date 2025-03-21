#!/usr/bin/env python3
import onnxruntime
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
from onnxruntime.transformers.io_binding_helper import IOBindingHelper, TypeHelper

from transformers import AutoTokenizer, AutoProcessor
from scipy.special import softmax
import onnx
import numpy as np
import math
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from pathlib import Path
from difflib import SequenceMatcher
import sys
import os

# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import utils

class PathValidationAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists(values):
            raise argparse.ArgumentError(self, f"The file '{values}' does not exist.")
        setattr(namespace, self.dest, values)

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Get the perplexity for a prompt and a model")
    parser.add_argument("-m", '--model-path', metavar = "DIR", type = Path, default = './model.onnx',  action=PathValidationAction,
                        help = 'Path to the model.onnx')
    parser.add_argument("-t", '--tokenizer', metavar = "DIR", type = Path, default = '.', action=PathValidationAction,
                         help = 'Path to the tokenizer folder')
    parser.add_argument("-p", '--prompt', metavar = "TEXT", type = str, default = 'Who is Messi?',
                        help = 'Given Prompt')
    parser.add_argument("-s", '--sequence-length', type = int, default = 64,
                        help = 'Total sequence length for the run')
    parser.add_argument("-c", '--context', type = int, default = 2048,
                        help = 'Total context length for the run')
    parser.add_argument("-n", '--no-context', action="store_true", default=False, 
                        help = 'If used, the model does not have the context extension in its ONNX')
    parser.add_argument("-g", '--generate-tokens', type = int, default = 0,
                        help = 'Total sequence length for the run')
    parser.add_argument('--enable-tracing', action = 'store_true', default = False,
                        help = 'Enable onnxruntime profiling and neuralizer traces')
    parser.add_argument('-v', '--verbose', action = 'store_true', default = False,
                        help = 'Show more output. Including ONNXRuntime logs')

    # options to be validated
    parser.add_argument("-r", '--run-dir', metavar = "DIR", type = str, default = 'myrundir',
                        help = 'Path to the run directory created by neuralizer')
    parser.add_argument('--use-kvc', action="store_true", default=True,
                        help = 'Enables KV-cache')
    parser.add_argument("-w", '--window-size', type = int, default = 1,
                        help = 'Maximum number of tokens that a KV-cache model can process per inference (must be smaller than the total sequence length)')
    parser.add_argument("-b", '--batch', type = int, default = 1,
                        help = 'Number of batched inferences')
    parser.add_argument("-l", '--num-layers', type = int, default = 32,
                        help = 'The number of layers of the model')
    parser.add_argument("--optimization-level", type = str, default = 'ORT_ENABLE_ALL', choices = ['ORT_DISABLE_ALL', 'ORT_ENABLE_BASIC', 'ORT_ENABLE_EXTENDED', 'ORT_ENABLE_ALL'],
                        help = 'Graph Optimization level in ONNX Runtime')
    parser.add_argument("--etglow-implementation", type = str, default = 'llm_kvc_inference_iobindings', choices = ['llm_kvc_inference', 'llm_kvc_inference_iobindings'],
                        help = 'Choose implementation that will be used to run with ETGLOW EP')
    args = parser.parse_args()

    if args.window_size > args.sequence_length:
        parser.error(f"Argument --window-size ({args.window_size}) should be less or equal to --sequence-legth ({args.sequence_length})")

    if args.no_context:
        args.context = None
    return args


# Define ONNX Runtime Graph Optimization level
def get_graph_optimization_level(level):
    match level:
        case 'ORT_DISABLE_ALL':
            return onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        case 'ORT_ENABLE_BASIC':
            return onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        case 'ORT_ENABLE_EXTENDED':
            return onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        case 'ORT_ENABLE_ALL':
            return onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        case _:
            sys.exit(f"Error in setting ORT Optimization Level. The optimization level {level} is not recognized")

def get_etglow_api_params(args, sep=';'):
    """Define etglow api parameters"""
    def get_device_placeholders(args, sep=';'):
        """Define implicit placheolders."""
        dph=""
        for id in range(args.num_layers):
            dph+=f"{sep}implicit-placeholder=past_key_values.{id}.key"
            dph+=f"{sep}implicit-placeholder=past_key_values.{id}.value"
            dph+=f"{sep}placeholder=present.{id}.key"
            dph+=f"{sep}placeholder=present.{id}.value"
        return dph

    dev_params = " ".join(['--gccCompileThreads=32','--logDisableCodeGenBits=-1', ])
    extra_params = '|'.join(['debug-glow=0', f'dev={dev_params}'])
    api_params = [
        "device-type=silicon",
        "glow-threads=2",
        f"runDir={args.run_dir}",
        f"extra-etsoc-params='{extra_params}'"
    ]
    api_params = sep.join(api_params)
    if args.use_kvc and args.etglow_implementation == "llm_kvc_inference_iobindings":
        api_params += get_device_placeholders(args, sep)
    return api_params

def get_sequence_size(args):
    """Get sequence size for a model"""
    return args.window_size if args.use_kvc else args.sequence_length

def get_past_sequence_size(args):
    """Get past sequence size for a model"""
    return args.sequence_length - args.window_size if args.use_kvc else 0

def get_context_size(args):
    """Get context size for a model"""
    if args.context is None:
        return get_sequence_size(args) + get_past_sequence_size(args)
    else:
        return args.context

# Defines dictionaries of onnx symbols
def get_onnx_symbols(args) -> dict[str, any]:
    if not args.use_kvc:
        symbols = {
            "batch": args.batch,
            "sequence": get_sequence_size(args)
        }
    else:
        symbols = {
        "batch": args.batch,
        "sequence": get_sequence_size(args),
        "past_sequence": get_past_sequence_size(args),
        "past_sequence+sequence": get_sequence_size(args) + get_past_sequence_size(args),
        "context": get_context_size(args),
        "context-sequence": get_context_size(args) - get_sequence_size(args)
    }
    return symbols


def get_onnx_shape_params(onnx_symbols):
    """Define onnx shape parameters"""
    onnx_shape_params = ''
    for key, value in onnx_symbols.items():
        onnx_shape_params += f"{key}={value},"
    # Return generated string without the last comma character
    return onnx_shape_params[:-1]


def get_etglow_provider_options(args, onnx_symbols) -> dict:
    """Constructs ETGLOW EP provider options to run LLMs"""
    poptions = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": get_onnx_shape_params(onnx_symbols),
        "etglow_api_params": get_etglow_api_params(args)
    }
    return poptions

def llm_kvc_inference(session : onnxruntime.InferenceSession, run_options : onnxruntime.RunOptions,
                      tokenizer : AutoTokenizer, input_tensors : dict,
                      prompt_tensor, num_tokens, context, sequence_len, window : int, batch : int) -> tuple[str, float]:
    sum_perplexity = 0
    new_token = np.array([10])
    prompt_size = len(prompt_tensor[0])
    total_input = prompt_tensor
    current_index = 0
    next_index = window

    inputs_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]

    # Run the inferences
    while next_index < num_tokens:
        if (new_token == tokenizer.eos_token_id).any():
            break

        output = session.run(output_names, input_tensors, run_options)

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
                    new_token = top1.reshape(batch, window) # Inf server
                    total_input = np.concatenate((total_input, new_token[: , -1:]), axis = 1)

                input_tensors['input_ids'] = total_input[:, j:next_index].reshape(batch, window) # inf server
            elif name == 'attention_mask':
                attention_mask = np.zeros((batch, sequence_len), dtype='int64')
                attention_mask[:, -next_index:] = 1
                input_tensors['attention_mask'] = attention_mask  # inf server
            elif name == 'position_ids':
                pos = np.concatenate((np.zeros([sequence_len - next_index], dtype = 'int64'), np.arange(next_index, dtype = 'int64').reshape(next_index)), axis=0)
                input_tensors['position_ids'] = np.tile(pos, (batch, 1))
            elif name == 'tree_attention':
                continue
            else:
                old_name = name.replace("past_key_values", "present")
                start_idx = next_index - current_index
                end_idx = start_idx + context - window
                input_tensors[name] = outs_dictionary[old_name][:, :, start_idx:end_idx, :]

        sum_perplexity += -np.log(softmax(logits[0, 0])[input_tensors['input_ids'][0]])

    sum_perplexity /= (num_tokens - 1)
    answers = tokenizer.batch_decode(total_input, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return answers, sum_perplexity


def llm_kvc_inference_iobindings(session : onnxruntime.InferenceSession, run_options : onnxruntime.RunOptions,
                                 tokenizer : AutoTokenizer, input_tensors : dict,
                                 prompt_tensor, num_tokens, context, sequence_len, window : int, batch : int) -> tuple[str, float]:

    sum_perplexity = 0
    new_token = np.array([10])
    prompt_size = len(prompt_tensor[0])
    total_input = prompt_tensor
    current_index = 0
    next_index = window
    # TODO: make 'llm_kvc_inference_iobindings' non-kvc friendly
    logits_last_dim = int(session.get_outputs()[0].shape[-1])
    hidden = int(session.get_outputs()[1].shape[-1])
    nheads = int(session.get_outputs()[1].shape[-3])

    inputs_names = [input.name for input in session.get_inputs()]
    outputs_names = [output.name for output in session.get_outputs()]

    # Pre-allocate inputs
    ortvalues = preallocate_input_outputs(session, outputs_names, input_tensors, window, batch, nheads, context, hidden, logits_last_dim)

    # Create IOBindings
    io_binding = session.io_binding()
    bind_input_outputs(io_binding, inputs_names, outputs_names, ortvalues, batch, nheads, context, hidden)

    with ThreadPoolExecutor() as executor:
        futures = []

        # Run the inferences
        while next_index < num_tokens:
            if (new_token == tokenizer.eos_token_id).any():
                break

            session.run_with_iobinding(io_binding, run_options)

            logits = ortvalues['logits'].numpy()

            # Prepare next inference inputs
            for name in inputs_names:
                if name == 'input_ids':
                    current_index = next_index
                    next_index = update_next_index(next_index, prompt_size, window)
                    j = next_index - window

                    if current_index >= prompt_size:
                        top1 = logits.argmax(-1)
                        new_token = top1.reshape(batch, window) # Inf server
                        total_input = np.concatenate((total_input, new_token[: , -1:]), axis = 1)

                    next_input_ids = total_input[:, j:next_index]
                    ortvalues['input_ids'].update_inplace(np.ascontiguousarray(next_input_ids))
                elif name == 'attention_mask':
                    attention_mask = np.zeros((batch, sequence_len), dtype='int64')
                    attention_mask[:, -next_index:] = 1
                    ortvalues['attention_mask'].update_inplace(attention_mask)
                elif name == 'position_ids':
                    pos = np.concatenate((np.zeros([sequence_len - next_index], dtype = 'int64'), np.arange(next_index, dtype = 'int64').reshape(next_index)), axis=0)
                    position_ids = np.tile(pos, (batch, 1))
                    ortvalues['position_ids'].update_inplace(position_ids)
                elif name == 'tree_attention':
                    continue
                else:
                    update_kvc_view(io_binding, name, ortvalues, current_index, batch, nheads, context, hidden)

            # Offload perplexity calculation to a separate thread
            futures.append(executor.submit(compute_perplexity, logits[0, 0], next_input_ids[0]))

        # Accumulate perplexity results from helper thread
        for future in as_completed(futures):
            sum_perplexity += future.result()

    sum_perplexity /= (num_tokens - 1)
    answers = tokenizer.batch_decode(total_input, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return answers, sum_perplexity


# Function to calculate perplexity
def compute_perplexity(logits, next_input_ids):
    # Perform the softmax and perplexity calculation
    return -np.log(softmax(logits)[next_input_ids])

def preallocate_input_outputs(session : onnxruntime.InferenceSession, output_names: list[str], input_tensors: dict,
                              window: int, batch: int, nheads: int, context: int, hidden: int, logits_last_dim: int,
                              device_type="et", device_id=0) -> dict:
    ortvalues = {
        'input_ids':      onnxruntime.OrtValue.ortvalue_from_numpy(input_tensors['input_ids'], device_type, device_id),
        'attention_mask': onnxruntime.OrtValue.ortvalue_from_numpy(input_tensors['attention_mask'], device_type, device_id),
        'position_ids': onnxruntime.OrtValue.ortvalue_from_numpy(input_tensors['position_ids'], device_type, device_id),
        'tree_attention': onnxruntime.OrtValue.ortvalue_from_numpy(input_tensors['tree_attention'], device_type, device_id),
        'logits':      onnxruntime.OrtValue.ortvalue_from_shape_and_type((batch, window, logits_last_dim), TypeHelper.ort_type_to_numpy_type(TypeHelper.get_output_type(session, 'logits')), device_type, device_id),
    }
    # All KVC input (past_key_value) & output (present) tensors will share same underlying allocation.
    zeros = np.zeros((batch, nheads, context, hidden), dtype=TypeHelper.ort_type_to_numpy_type(TypeHelper.get_output_type(session, 'present.0.key')))
    zeros_padded = np.pad(zeros, ((0,0), (0,1), (0, 0), (0,0)), mode='constant')
    for name in output_names:
        if 'present' in name:
            ortvalues[name] = onnxruntime.OrtValue.ortvalue_from_numpy(zeros_padded, device_type, device_id)
    return ortvalues


def bind_input_outputs(io_binding, inputs_names, outputs_names, ortvalues, batch : int, nheads : int, context : int, hidden : int):
    # Bind inputs (will bind them to the allocated memory)
    for name in inputs_names:
        if name in ['input_ids', 'attention_mask', 'position_ids', 'tree_attention']:
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

def update_kvc_view(io_binding, name, ortvalues, current_index, batch : int, nheads, context, hidden):
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


def get_prompt_tensor(prompt, tokenizer, batch):
    input_tensor = tokenizer(prompt, return_tensors="pt")
    input_ids = input_tensor['input_ids']
    batched_input_ids = np.tile(input_ids,(batch, 1))
    return batched_input_ids


def fix_model_dimensions(model_path : Path, onnx_symbols: dict) -> Path:
    # Fix dimensions in the model
    model_noexternaldata = onnx.load(model_path, load_external_data=False)
    for key, value in onnx_symbols.items():
        make_dim_param_fixed(model_noexternaldata.graph, key, value)
    model_noexternaldata = model_noexternaldata.SerializeToString()
    fixeddim_model_path = model_path.parent / "model-fixed-dims.onnx"
    with open(fixeddim_model_path, "wb") as f:
        f.write(model_noexternaldata)
    return fixeddim_model_path


def preprocess_llm_input_tensors(session : onnxruntime.InferenceSession, input_ids_tensor: np.ndarray, args) -> dict:
    window = args.window_size
    batch = args.batch
    pads = args.sequence_length
    context = get_context_size(args)

    pos = np.concatenate((np.zeros([pads - window], dtype = 'int64'), np.arange(window, dtype = 'int64').reshape(window)), axis=0)
    trian = np.triu(-65504*np.ones(pads), k= 1).astype('float16').reshape(1, 1, pads, pads)
    inputs_names = [input.name for input in session.get_inputs()]
    inputs_dict = {
        'input_ids': copy.deepcopy(input_ids_tensor[:, :window].reshape(batch, window)),
        'attention_mask': np.concatenate((np.zeros([batch, pads - window], dtype = 'int64'), np.ones((batch, window), dtype = 'int64')), axis=1),
        'position_ids': np.tile(pos, (batch, 1)),
        'tree_attention': np.tile(trian, (batch, 1, 1, 1))
    }
    if args.use_kvc:
        # deduce n_heads and hidden from model using ORT
        n_heads = int(session.get_outputs()[1].shape[-3])
        hidden = int(session.get_outputs()[1].shape[-1])
        for name in inputs_names:
            if name not in ['input_ids','attention_mask', 'position_ids', 'tree_attention']:
                inputs_dict[name] = np.zeros([batch, n_heads, context - window, hidden], dtype="float16")
    return inputs_dict


def print_llm_inference_setup(model_name, prompt_size, generate_tokens, onnx_symbols):
    print(f"Running model: {model_name}")
    print(f"    Prompt size: : {prompt_size}")
    print(f"    Generating #tokens: {generate_tokens}")
    print(f"    ONNX symbols: {onnx_symbols}")


def print_llm_inference_results(label : str, comp_time : float, inf_time : float,  perplexity, answers : [str], num_tokens):
    print(f'{label}:')
    print(f'    Compilation took {comp_time:.2f}s')
    print(f'    Inference took {inf_time:.2f}s ({(num_tokens / inf_time):.2f} tokens/s)')
    print(f'    Perplexity value: {float(perplexity.item()):.2f}')
    for i in range(len(answers)):
        print(f'    Question and answer[{i}]: {answers[i]}')

def compare_results(perplexity_cpu, perplexity_etglow, answers_cpu : [str], answers_etglow : [str]):
    perplexity_tolerance = 0.05  # Define a relative tolerance for perplexity comparison
    similarity_threshold = 0.9  # Set a threshold for answer similarity

    if not math.isclose(perplexity_cpu, perplexity_etglow, rel_tol=perplexity_tolerance):
        # preplexities should be close
        raise ValueError(f"Perplexity scores differ significantly: CPU = {perplexity_cpu}, ETGlow = {perplexity_etglow}. "
                         f"Difference exceeds tolerance of {perplexity_tolerance*100}%.")

    for i, (answer_cpu, answer_etglow) in enumerate(zip(answers_cpu, answers_etglow)):
        # Calculate similarity ratio using SequenceMatcher
        similarity_ratio = SequenceMatcher(None, answer_cpu, answer_etglow).ratio()
        if similarity_ratio < similarity_threshold:
            raise ValueError(f"Answer {i+1} differs significantly between CPU and ETGlow. "
                             f"Similarity ratio: {similarity_ratio:.4f} (Threshold: {similarity_threshold}).\n"
                             f"CPU Answer: {answer_cpu}\nETGlow Answer: {answer_etglow}")


def main():
    args = parse_arguments()
    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"args {args}")
    # Paths
    model_path = args.model_path
    context_len = get_context_size(args)
    model_name = str(model_path.parents[0]).replace(str(model_path.parents[1]), '').replace(r'/', '')

    # Check sequence length is long enough
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt_tensor = get_prompt_tensor(args.prompt, tokenizer, args.batch)
    prompt_size = len(prompt_tensor[0])
    assert(prompt_size + args.generate_tokens <= args.sequence_length), f"Sequence length is not long enough (should be at least: {prompt_size + args.generate_tokens}"

    # Exit if the model does not use KVC
    model = onnx.load(model_path)
    assert(len(model.graph.input) > 2), "Non Key-Value Cache models are not yet supported"

    onnx_symbols = get_onnx_symbols(args)
    print_llm_inference_setup(model_name, prompt_size, args.generate_tokens, onnx_symbols)

    # Common Session options
    session_options = onnxruntime.SessionOptions()
    utils.set_verbose_output(session_options, args.verbose)
    session_options.enable_profiling = args.enable_tracing
    session_options.graph_optimization_level = get_graph_optimization_level(args.optimization_level)

    # Fix model dimensions (this would not be necessary if we provided all ONNX symbols to Glow)
    fixed_model_path = fix_model_dimensions(model_path, onnx_symbols)

    # Create cpu ORT session
    start = time.time()
    session_options.profile_file_prefix = f'{model_name}_cpu_window_{args.window_size}'
    session_cpu = onnxruntime.InferenceSession(fixed_model_path,  sess_options=session_options, providers=['CPUExecutionProvider'])
    comp_time_cpu = time.time() - start
    
    run_options_cpu = onnxruntime.RunOptions()
    run_options_cpu.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")

    # Process inputs
    input_tensors_cpu = preprocess_llm_input_tensors(session_cpu, prompt_tensor, args)

    # Execute inference on CPU
    start = time.time()
    answers_cpu, perplexity_cpu = llm_kvc_inference(session_cpu, run_options_cpu, tokenizer, input_tensors_cpu, prompt_tensor, args.generate_tokens, context_len, args.sequence_length, args.window_size, args.batch)
    session_cpu.end_profiling()
    inf_time_cpu = time.time() - start

    print_llm_inference_results('CPU EP results', comp_time_cpu, inf_time_cpu, perplexity_cpu, answers_cpu, args.batch * args.generate_tokens)

    # Create etglow Provider options
    provider_options = get_etglow_provider_options(args, onnx_symbols)
    # Create etglow ORT session
    start = time.time()
    session_options.profile_file_prefix = f'{model_name}_etglow_window_{args.window_size}'
    session_etglow = onnxruntime.InferenceSession(fixed_model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'], provider_options=[provider_options])
    etsoc_comp_time = time.time() - start 

    run_options_etglow = onnxruntime.RunOptions()
    run_options_etglow.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")  # force memory shrinkage

    # Process inputs
    input_tensors_etglow = preprocess_llm_input_tensors(session_etglow, prompt_tensor, args)

    # Launch ETSoC inference
    start = time.time()
    match args.etglow_implementation:
        case 'llm_kvc_inference_iobindings':
            answers_etglow, perplexity_etglow = llm_kvc_inference_iobindings(session_etglow, run_options_etglow, tokenizer, input_tensors_etglow, prompt_tensor, args.generate_tokens, context_len, args.sequence_length, args.window_size, args.batch)
        case "llm_kvc_inference":
            answers_etglow, perplexity_etglow = llm_kvc_inference(session_etglow, run_options_etglow, tokenizer, input_tensors_etglow, prompt_tensor, args.generate_tokens, context_len, args.sequence_length, args.window_size, args.batch)
        case _:
            logger.error(f'Unknown etglow_implementation: {args.etglow_implementation}')
    session_etglow.end_profiling()
    inf_time_etsoc = time.time() - start

    print_llm_inference_results('ETGlow EP results', etsoc_comp_time, inf_time_etsoc, perplexity_etglow, answers_etglow,  args.batch * args.generate_tokens)

    compare_results(perplexity_cpu, perplexity_etglow, answers_cpu, answers_etglow)

if __name__ == "__main__":
    main()
