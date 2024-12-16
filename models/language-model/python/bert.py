#!/usr/bin/env python3

import onnxruntime as ort
import sys
import numpy
import json
from typing import Sequence, Optional
from pathlib import Path
import os

# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import utils

def get_answer_bert(output_tensors, input_tensors, tokenizer, batch, b):

    start = end = 0
    if (batch == 1):        
        start = numpy.argmax(output_tensors[0])
        end = numpy.argmax(output_tensors[1])
    else:
        start = numpy.argmax(output_tensors[0][b])
        end = numpy.argmax(output_tensors[1][b])
    answer = tokenizer.decode(input_tensors['input_ids'][b][start:end + 1])
    print(tokenizer.decode(input_tensors['input_ids'][b]))
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

def get_api_params():
    neura_params = " ".join(['--gccCompileThreads=16','--logDisableCodeGenBits=-1'])
    extra_params = '|'.join(['debug-glow=0', f'dev={neura_params}'])
    api_params = ';'.join(["glow-threads=2", "runDir=myrundir", f"extra-etsoc-params='{extra_params}'"])
    return api_params

def get_provider_options(args) -> dict:
    api_params = get_api_params()
    if args.fp16:
        api_params += ";"+';'.join([f"useFP16='{args.fp16}'"])   
    
    provider_options = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": f'batch_size={args.batch};sequence_length=128;Squeezeoutput_start_logits_dim_1=128',
        "etglow_api_params": api_params
    }
    
    return provider_options
    
def main(argv: Optional[Sequence[str]] = None):
    """Launch BERT onnx model on cpu and etglow and compare results."""
    parser = utils.get_common_arg_parser()
    parser.add_argument("--bert-variant", default="bert", choices=['bert', 'bert-large', 'albert', 'distilbert'], help="Selects which type of bert model variant to run. Options available: [ bert | bert-large | albert | distilbert ]")
    parser = utils.extra_arguments(parser)
    args = parser.parse_args(argv)

    artifacts_path = Path(args.artifacts)
    if args.bert_variant == 'bert':
        modelname = 'bert_base_onnx'
        tensorspath = artifacts_path / f'input_tensors/bert_squad_128/data'
    elif args.bert_variant == 'bert-large':
        modelname = 'bert_large_onnx'
        tensorspath = artifacts_path / f'input_tensors/bert_squad_128/data'
    elif args.bert_variant == 'albert':
        modelname = 'albert-s128-fp32-onnx'
        tensorspath = artifacts_path / f'input_tensors/albert_squad_128/data'
    elif args.bert_variant == 'distilbert':
        modelname = 'distilbert-s128-fp32-onnx'
        tensorspath = artifacts_path / f'input_tensors/distilbert_squad_128/data'
    else:
        print(f'Model {args.bert_variant} is invalid.')

    tokenizer = get_tokenizer(args.bert_variant)
    modelpath = artifacts_path / f'models/{modelname}/model.onnx'
    predictionpath = tensorspath / 'prediction.json'
    if not predictionpath.exists:
       raise FileNotFoundError(f"Prediction file: {predictionpath} does not exist.")

    # session and provider options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    utils.set_verbose_output(sess_options, args.verbose)
    sess_options.enable_profiling = args.enable_tracing
    poptions = get_provider_options(args) 

    print('Executing inferences...\n')

    # Run inferences on cpu
    sess_options.profile_file_prefix = f'{modelname}_cpu'
    session_cpu    = ort.InferenceSession(modelpath, sess_options, providers=['CPUExecutionProvider'])
    input_tensors_cpu, output_tensor_cpu, time_cpu = utils.test_with_tensor(tensorspath, session_cpu, args)
    session_cpu.end_profiling()

    # Run inferences on etglow
    sess_options.profile_file_prefix = f'{modelname}_etglow'
    session_etglow = ort.InferenceSession(modelpath, sess_options, providers=['EtGlowExecutionProvider'], provider_options=[poptions])
    input_tensors_etglow, output_tensor_etglow, time_et = utils.test_with_tensor(tensorspath, session_etglow, args)   
    session_etglow.end_profiling()

    for i in range(args.launches):
        for b in range(args.batch):
            answer_cpu = get_answer_bert(output_tensor_cpu[i][0], input_tensors_cpu[i], tokenizer, args.batch, b)
            answer_etglow = get_answer_bert(output_tensor_etglow[i][0], input_tensors_etglow[i], tokenizer, args.batch, b)
            predictionpath = Path(os.path.join(tensorspath, f'inference-{str((i+b)%128)}/prediction.json'))
            if not predictionpath.exists:
                raise FileNotFoundError(f"Prediction file: {predictionpath} does not exist.")

            with predictionpath.open('r') as file:
                prediction = json.load(file)
   
            print(f"Context:\n-----------------\n{prediction[0]['context']}")
            print(f"Dataset answer is: {prediction[0]['answer']}")
            #Inference time is only show when warm_up is true. 
            message = f'CPU EP answer is: {answer_cpu}'
            print(f'{message} in {output_tensor_etglow[i][1]:.4f} s.' if args.warm_up else message)
            message = f'ETGLOW EP answer is: {answer_etglow}'
            print(f'{message} in {time_et:.4f} s.' if args.warm_up else message)

    
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
