#!/usr/bin/env python3

import onnxruntime as ort
import sys
import os
import numpy
import json
from typing import Sequence, Optional
from pathlib import Path
import time

# Import utils.py
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent.parent.parent.as_posix(), 'models'))
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

    if args.enable_tracing:
        api_params += ";"+utils.get_tracing_params()

    if args.fp16:
        api_params += ";"+';'.join([f"useFP16='{args.fp16}'"])
   
    provider_options = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": f'batch_size={args.batch};sequence_length=128;Squeezeoutput_start_logits_dim_1=128',
        "etglow_api_params": api_params
    }
    
    return provider_options

def extra_arguments(parser):    
    parser.add_argument("--fp16", action = 'store_true',
                        help = 'Force the use of 16-bit floating point values when true')
     
def main(argv: Optional[Sequence[str]] = None):
    """Launch BERT onnx model on etglow provider for getting performance data."""
    parser = utils.get_common_arg_parser()
    parser.add_argument("--bert-variant", default="bert", choices=['bert', 'bert-large', 'albert', 'distilbert'], help="Selects which type of bert model variant to run. Options available: [ bert | bert-large | albert | distilbert ]")
    extra_arguments(parser)
    args = parser.parse_args(argv)

    artifacts_path = Path(args.artifacts)
    if args.bert_variant == 'bert':
        modelname = 'bert_base_onnx'
        tensorspath = artifacts_path / f'input_tensors/bert_squad_128/data/'
    elif args.bert_variant == 'bert-large':
        modelname = 'bert_large_onnx'
        tensorspath = artifacts_path / f'input_tensors/bert_squad_128/data'
    elif args.bert_variant == 'albert':
        modelname = 'albert_128_pth_to_onnx'
        tensorspath = artifacts_path / f'input_tensors/albert_squad_128/data'
    elif args.bert_variant == 'distilbert':
        modelname = 'distilbert_128_pth_to_onnx'
        tensorspath = artifacts_path / f'input_tensors/distilbert_squad_128/data'
    else:
        print(f'Model {args.bert_variant} is invalid.')

    tokenizer = get_tokenizer(args.bert_variant)
    modelpath = artifacts_path / f'models/{modelname}/model.onnx'

    # session and provider options
    sess_options = ort.SessionOptions()
    utils.set_verbose_output(sess_options, args.verbose)
    sess_options.enable_profiling = args.enable_tracing
    poptions = get_provider_options(args) 

    print('Executing inferences...\n')

    # Run inferences on etglow
    sess_options.profile_file_prefix = f'{modelname}_etglow'
    session_etglow = ort.InferenceSession(modelpath, sess_options, providers=['EtGlowExecutionProvider'], provider_options=[poptions])
    input_tensors_etglow_list, output_tensor_etglow, et_total_time = utils.test_with_tensor(tensorspath, session_etglow, args)    
    session_etglow.end_profiling()      

    for i in range(args.launches):
        for b in range(args.batch):
            answer_etglow = get_answer_bert(output_tensor_etglow[i][0], input_tensors_etglow_list[i], tokenizer, args.batch, b)
            predictionpath = Path(os.path.join(tensorspath, f'inference-{str((i+b)%128)}/prediction.json'))
            if not predictionpath.exists:
                raise FileNotFoundError(f"Prediction file: {predictionpath} does not exist.")

            with predictionpath.open('r') as file:
                prediction = json.load(file)

            print(f"Context:\n-----------------\n{prediction[0]['context']}")
            print(f"Dataset answer is: {prediction[0]['answer']}")
            message = f'ETGLOW EP answer is: {answer_etglow}'
            print(f'{message} in {output_tensor_etglow[i][1]:.4f} s.' if args.warm_up else message)        

    print(f'Total launches {args.launches} batch used {args.batch} in {et_total_time} s.')
    print(f'ET_provider Performance: {(args.launches * args.batch) / et_total_time:.4f} inf/sec')
        
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
