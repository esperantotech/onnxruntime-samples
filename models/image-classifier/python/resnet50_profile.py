#!/usr/bin/env python3
import onnxruntime as ort
import sys
from typing import Sequence, Optional
from pathlib import Path
import os

# Import utils.py
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent.parent.parent.as_posix(), 'models'))
from common import utils

def get_provider_options(args) -> dict:
    poptions = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": f'N={args.batch}'
    }

    api_params = "glow-threads=4"

    if api_params:
        poptions["etglow_api_params"] =  api_params

    return poptions

def main(argv: Optional[Sequence[str]] = None):
    """Launch RESNET50 onnx model on cpu and etglow providers and compare results."""
    parser = utils.get_common_arg_parser()
    parser = utils.get_img_classifier_arg_parser(parser)   
    args = parser.parse_args(argv)

    # Paths
    artifacts_path = Path(args.artifacts)
    modelname = 'resnet50-v2-7'
    modelpath = artifacts_path / f'models/{modelname}/model.onnx'
    imagespath = artifacts_path /'input_data/images/imagenet/'      

    # Session and provider options
    sess_options = ort.SessionOptions()
    if (args.mode == "async"):        
        sess_options.intra_op_num_threads=10
    print (f'Session intra-op num. threads: {sess_options.intra_op_num_threads}')

    utils.set_verbose_output(sess_options, args.verbose)
    sess_options.enable_profiling = args.enable_tracing
    # Set graph optimization level
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    poptions = get_provider_options(args)
    
    print('Executing inferences...\n')
    
    sess_options.profile_file_prefix = f'{modelname}_etglow_inf_{args.launches}_batch_{args.batch}'
    session_etglow = ort.InferenceSession(modelpath, sess_options, providers=['EtGlowExecutionProvider'], provider_options=[poptions])
    results_imagenet_etglow, et_total_time = utils.test_with_images(imagespath, session_etglow, args)
    session_etglow.end_profiling()

    print(f'Total Inferences {args.batch * args.launches} batch used {args.batch} in {et_total_time} s.')
    print(f'ET_provider Performance: {(args.launches * args.batch)/et_total_time:.4f} inf/sec')

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
