#!/usr/bin/env python3
import whisper
import onnx
import sys
import time
import onnxruntime
from typing import Sequence, Optional
import numpy as np
from pathlib import Path
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed

# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import utils

def extra_arguments(parser):
    parser.add_argument("-i", '--input-audio', type = str, default = './1984.m4a',
                        help = 'Audio to transcribe')
    parser.add_argument("-n", '--new-tokens', type = int, default = 20,
                        help = 'Number of new tokens to generate')

# Define etglow api parameters
def get_etglow_api_params(args):
    api_params = [
        "device-type=silicon",
        "glow-threads=2",
        "runDir=myrundir",
        "kernel-launch-timeout=20",
        "extra-etsoc-params='" + '|'.join([
            'debug-glow=0',
            'dev=' + " ".join([
                '--gccCompileThreads=32',
                '--logDisableCodeGenBits=-1',
                '--enableGraphLogs=none',
            ]),
        ]) + "'"
    ]        
    return ';'.join(api_params)


# Defines dictionaries of onnx symbols
def get_onnx_symbols_encoder(timesteps : int):
    symbols = {
        "batch": 1,
        "timesteps": timesteps,
        "timesteps_div_2": int(timesteps / 2)
    }
    return symbols


def get_onnx_symbols_decoder(context : int, timesteps_div_2 : int):
    sequence = 1

    symbols = {
        "batch": 1,
        "context": context,
        "sequence": sequence,
        "context-sequence": context - sequence,
        "past_sequence+sequence": context,
        "timesteps_div_2": timesteps_div_2
    }
    return symbols


# Define onnx shape parameters
def get_onnx_shape_params(onnx_symbols):
    onnx_shape_params = ''
    for key, value in onnx_symbols.items():
        onnx_shape_params += f"{key}={value},"
    # Return generated string without the last comma character
    return onnx_shape_params[:-1]


def run_whisper_decoder(decoder_model_path, execution_provider, session_options, decoder_output_names, cross_attn_tensors, num_new_tokens, provider_options = {}):
    session_options.profile_file_prefix = f'whisper_decode_{execution_provider}'

    start = time.time()
    decoder_session = onnxruntime.InferenceSession(decoder_model_path, sess_options=session_options, providers=[execution_provider], provider_options=[provider_options])
    compile_time = time.time()
    transcription = decoder_loop(decoder_session, decoder_output_names, cross_attn_tensors, num_new_tokens)
    inference_time = time.time()
    decoder_session.end_profiling()
    print(f'Results for {execution_provider}:')
    print(f'    Compilation took {compile_time - start:.2f}s')
    print(f'    Inference took {inference_time - compile_time:.2f}s')
    print(f'    Transcription: {transcription}')

    return transcription


def decoder_loop(decoder_session, decoder_output_names, cross_attn_tensors, num_new_tokens):
    # Generate start of transcription tokens
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
    first_tokens = np.array([tokenizer.sot, 0, tokenizer.transcribe, tokenizer.no_timestamps], dtype=np.int64)

    # Self attention mask key, value vectors
    self_attn_past_k = []
    self_attn_past_v = []
    for i in range(32):
        self_attn_past_k.append(np.zeros((1, 20, 447, 64), dtype=np.float32))
        self_attn_past_v.append(np.zeros((1, 20, 447, 64), dtype=np.float32))

    # Cross attention
    cross_attn_k = cross_attn_tensors[0::2]
    cross_attn_v = cross_attn_tensors[1::2]

    # Attention mask
    attn_mask_size = 448
    attn_mask = np.zeros((1,attn_mask_size), dtype=np.int64)

    # Process first tokens
    for j in range(len(first_tokens)):
        tokens = np.array([first_tokens[j]], dtype=np.int64).reshape(1, 1)
        attn_mask[0,-1 - j] = 1

        decoder_input = {"input_ids": tokens, "attention_mask": attn_mask}
        for i in range(32):
            decoder_input[f"past_key_values.{str(i)}.key"] = self_attn_past_k[i]
            decoder_input[f"past_key_values.{str(i)}.value"] = self_attn_past_v[i]
            decoder_input[f"cross_attn.{str(i)}.key"] = cross_attn_k[i]
            decoder_input[f"cross_attn.{str(i)}.value"] = cross_attn_v[i]

        logits, *cache_tensors = decoder_session.run(decoder_output_names, decoder_input)
        next_token = np.argmax(logits[0,0])

        self_attn_k = cache_tensors[0::2]
        self_attn_v = cache_tensors[1::2]
        for i in range(32):
            self_attn_past_k[i] = self_attn_k[i][:,:,1:,:]
            self_attn_past_v[i] = self_attn_v[i][:,:,1:,:]

        if (j == 0):
            # set language token
            first_tokens[1] = next_token

    transcribed_tokens = [next_token]
    for j in range(4, 4 + num_new_tokens):
        tokens = np.array([transcribed_tokens[-1]], dtype=np.int64).reshape(1, 1)
        attn_mask[0,-1 - j] = 1

        decoder_input = {"input_ids": tokens, "attention_mask": attn_mask}
        for i in range(32):
            decoder_input[f"past_key_values.{str(i)}.key"] = self_attn_past_k[i]
            decoder_input[f"past_key_values.{str(i)}.value"] = self_attn_past_v[i]
            decoder_input[f"cross_attn.{str(i)}.key"] = cross_attn_k[i]
            decoder_input[f"cross_attn.{str(i)}.value"] = cross_attn_v[i]

        logits, *cache_tensors = decoder_session.run(decoder_output_names, decoder_input)
        next_token = np.argmax(logits[0,0])
        # print(j, next_token)
        if next_token == tokenizer.eot: # end_of_transcription
            break
        transcribed_tokens.append(next_token)
        self_attn_k = cache_tensors[0::2]
        self_attn_v = cache_tensors[1::2]
        for i in range(32):
            self_attn_past_k[i] = self_attn_k[i][:,:,1:,:]
            self_attn_past_v[i] = self_attn_v[i][:,:,1:,:]

    return tokenizer.decode(transcribed_tokens)


def main(argv: Optional[Sequence[str]] = None):
    num_seconds = 28.8

    parser = utils.get_common_arg_parser()
    extra_arguments(parser)
    args = parser.parse_args(argv)

    artifacts_path = Path(args.artifacts)
    speech_path = Path(args.input_audio)
    encoder_model_path = artifacts_path / 'models/whisper-large-v2/encoder/model.onnx'
    decoder_model_path = artifacts_path / 'models/whisper-large-v2/decoder/model.onnx'

    # Encoder api params
    api_params = get_etglow_api_params(args)
    onnx_symbols = get_onnx_symbols_encoder(int(100*num_seconds)) # timestamps is 100*<length of the audio clip>
    onnx_shape_params = get_onnx_shape_params(onnx_symbols)
    provider_options = {"etglow_greedy": "true",
                        "etglow_onnx_shape_params": onnx_shape_params,
                        "etglow_api_params": api_params}
    print(provider_options)

    # Fix encoder model dimensions
    encoder = onnx.load(encoder_model_path, load_external_data=False)
    for key, value in onnx_symbols.items():
        make_dim_param_fixed(encoder.graph, key, value)
    # Write new model to disk
    encoder_fixed_dims = artifacts_path / 'models/whisper-large-v2/encoder/encoder_fixed_dims.onnx'
    with open(encoder_fixed_dims, "wb") as f:
        f.write(encoder.SerializeToString())

    # Load audio
    print(f"Spectrogram speech audio file {speech_path}... ", end="")
    audio = whisper.load_audio(speech_path)
    audio = whisper.pad_or_trim(audio, length=int(num_seconds*16000))
    mel = whisper.log_mel_spectrogram(audio).unsqueeze(0) # Unsqueeze to set batch=1
    print("OK")

    print("Running encoder... ", end="")

    # Session options
    session_options = onnxruntime.SessionOptions()
    utils.set_verbose_output(session_options, args.verbose)
     # Disable all the graph optimizations
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.enable_profiling = args.enable_tracing

    # Encode
    encoder_input = {"mel": mel.numpy()}
    encoder_output_names = [tensor.name for tensor in encoder.graph.output]
    # CPU encoding
    cpu_provider = 'CPUExecutionProvider'
    session_options.profile_file_prefix = f'whisper_encode_{cpu_provider}'
    enc_session_cpu = onnxruntime.InferenceSession(encoder_fixed_dims, sess_options=session_options, providers=[cpu_provider])
    cross_attn_tensors_cpu = enc_session_cpu.run(encoder_output_names, encoder_input)
    enc_session_cpu.end_profiling()
    # Etglow encoding
    glow_provider = 'EtGlowExecutionProvider'
    session_options.profile_file_prefix = f'whisper_encode_{glow_provider}'
    enc_session_etglow = onnxruntime.InferenceSession(encoder_fixed_dims, sess_options=session_options, providers=[glow_provider], provider_options=[provider_options])
    cross_attn_tensors_etglow = enc_session_etglow.run(encoder_output_names, encoder_input)
    enc_session_etglow.end_profiling()

    print("OK")

    # DECODE API PARAMS
    max_context = 448
    api_params = get_etglow_api_params(args)
    print("api_params: " + api_params)

    onnx_symbols = get_onnx_symbols_decoder(max_context, int(100*num_seconds/2))
    onnx_shape_params = get_onnx_shape_params(onnx_symbols)
    print(onnx_symbols)
    provider_options = {"etglow_compile_only": "false",
                        "etglow_dump_subgraphs": "false",
                        "etglow_greedy": "true",
                        "etglow_onnx_shape_params": onnx_shape_params,
                        "etglow_api_params": api_params}

    # Fix decoder model dimensions
    decoder = onnx.load(decoder_model_path, load_external_data=False)
    for key, value in onnx_symbols.items():
        make_dim_param_fixed(decoder.graph, key, value) 
        
    # Write new model to disk
    decoder_fixed_model = artifacts_path / 'models/whisper-large-v2/decoder/decoder_fixed_dims.onnx'
    with open(decoder_fixed_model, "wb") as f:
        f.write(decoder.SerializeToString())



    # Run decoder model CPU
    decoder_output_names = [tensor.name for tensor in decoder.graph.output]
    
    run_whisper_decoder(decoder_fixed_model, cpu_provider, session_options, decoder_output_names, cross_attn_tensors_cpu, args.new_tokens)
    run_whisper_decoder(decoder_fixed_model, glow_provider, session_options, decoder_output_names, cross_attn_tensors_etglow, args.new_tokens, provider_options)

 
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
