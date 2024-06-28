#!/usr/bin/env python3

import onnxruntime as ort
import sys
import numpy
import json
from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional
from pathlib import Path

# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import utils

def get_answer_bert(output_tensors, input_tensors, tokenizer):
    start = numpy.argmax(output_tensors[0])
    end = numpy.argmax(output_tensors[1])
    answer = tokenizer.decode(input_tensors['input_ids'][0][start:end + 1])
    print(tokenizer.decode(input_tensors['input_ids'][0]))
    return answer

def get_tokenizer(model : str):
    if model in ['bert', 'bert-large']:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")          
    elif model == 'albert':
        from transformers import AlbertTokenizer
        tokenizer = AlbertTokenizer.from_pretrained("twmkn9/albert-base-v2-squad2")
    elif model == 'distilbert':
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    else:
        print("Model not recognized")
        return 1
    
    return tokenizer

def get_provider_options(args) -> dict:
    poptions = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": "batch_size=1;sequence_length=128;Squeezeoutput_start_logits_dim_1=128"}
    
    if args.enable_tracing:
        poptions['etglow_api_params'] = utils.get_tracing_params()

    return poptions

def main(argv: Optional[Sequence[str]] = None):
    """Launch BERT onnx model on cpu and etglow and compare results."""
    parser = utils.get_arg_parser()
    parser.add_argument("-b", "--bert-variant", default="bert", choices=['bert', 'bert-large', 'albert', 'distilbert'], help="Selects which type of bert model variant to run. Options available: [ bert | bert-large | albert | distilbert ]")
    args = parser.parse_args(argv)

    artifacts_path = Path(args.artifacts)
    if args.bert_variant == 'bert':
        modelname = 'bert_base_onnx'
        tensorspath = artifacts_path / f'input_tensors/bert_squad_128/data/inference-98'
    elif args.bert_variant == 'bert-large':
        modelname = 'bert_large_onnx'
        tensorspath = artifacts_path / f'input_tensors/bert_squad_128/data/inference-98'
    elif args.bert_variant == 'albert':
        modelname = 'albert_128_pth_to_onnx'
        tensorspath = artifacts_path / f'input_tensors/albert_squad_128/data/inference-0'
    elif args.bert_variant == 'distilbert':
        modelname = 'distilbert_128_pth_to_onnx'
        tensorspath = artifacts_path / f'input_tensors/distilbert_squad_128/data/inference-98'
    else:
        print(f'Model {args.bert_variant} is invalid.')

    tokenizer = get_tokenizer(args.bert_variant)
    modelpath = artifacts_path / f'models/{modelname}/model.onnx'
    predictionpath = tensorspath / 'prediction.json'
    if not predictionpath.exists:
       raise FileNotFoundError(f"Prediction file: {predictionpath} does not exist.")

    # session and provider options
    sess_options = ort.SessionOptions()
    utils.set_verbose_output(sess_options, args.verbose)
    sess_options.enable_profiling = args.enable_tracing
    poptions = get_provider_options(args) 

    print('Executing inferences...\n')

    # Run inferences on cpu
    sess_options.profile_file_prefix = f'{modelname}_cpu'
    session_cpu    = ort.InferenceSession(modelpath, sess_options, providers=['CPUExecutionProvider'])
    input_tensors_cpu, output_tensor_cpu = utils.test_with_tensor(tensorspath, session_cpu)
    answer_cpu = get_answer_bert(output_tensor_cpu, input_tensors_cpu, tokenizer)
    session_cpu.end_profiling()

    # Run inferences on etglow
    sess_options.profile_file_prefix = f'{modelname}_etglow'
    session_etglow = ort.InferenceSession(modelpath, sess_options, providers=['EtGlowExecutionProvider'], provider_options=[poptions])
    input_tensors_etglow, output_tensor_etglow = utils.test_with_tensor(tensorspath, session_etglow)
    answer_etglow = get_answer_bert(output_tensor_etglow, input_tensors_etglow, tokenizer)
    session_etglow.end_profiling()

    with predictionpath.open('r') as file:
        prediction = json.load(file)
        
    print(f"Context:\n-----------------\n{prediction[0]['context']}")
    print(f"Dataset answer is: {prediction[0]['answer']}")
    print(f"CPU EP answer is: {answer_cpu}")
    print(f"ETGLOW EP answer is: {answer_etglow}")

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
