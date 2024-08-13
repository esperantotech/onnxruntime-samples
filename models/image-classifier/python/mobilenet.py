#!/usr/bin/env python3

import onnxruntime as ort
import sys
from typing import Sequence, Optional
from pathlib import Path

# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import utils

def get_provider_options(enable_tracing) -> dict:
    poptions = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": "batch=1;height=224;width=224",
    }
    
    api_params = ""
    if api_params:
        poptions["etglow_api_params"] = api_params

    return poptions


def main(argv: Optional[Sequence[str]] = None):
    """Launch mobilenet onnx model on cpu and etglow and compare results."""
    parser = utils.get_common_arg_parser()
    parser = utils.get_img_classifier_arg_parser(parser)
    args = parser.parse_args(argv)
    batch = args.batch
    num_inferences = args.launches

    # Paths
    artifacts_path = Path(args.artifacts)
    modelname = 'mobilenet_v3_small_denso_onnx'
    modelpath = artifacts_path / f'models/{modelname}/model.onnx'
    imagespath = artifacts_path /'input_data/images/imagenet/'
    labelspath = imagespath / 'index_to_name.json'

    # Session and provider options
    sess_options = ort.SessionOptions()
    if (args.mode == "async"):        
        sess_options.intra_op_num_threads=2
    print (f'Session intra-op num. threads: {sess_options.intra_op_num_threads}')

    utils.set_verbose_output(sess_options, args.verbose)
    sess_options.enable_profiling = args.enable_tracing
    poptions = get_provider_options(args)
    print(poptions)

    print('Executing inferences...\n')

    # Run cpu provider inferences
    sess_options.profile_file_prefix = f'{modelname}_cpu_inf_{num_inferences}_batch_{batch}'
    session_cpu = ort.InferenceSession(modelpath, sess_options, providers=['CPUExecutionProvider'])
    results_imagenet_cpu = utils.test_with_images(imagespath, session_cpu, args)
    session_cpu.end_profiling()
    
    # ETSoC
    sess_options.profile_file_prefix = f'{modelname}_etglow_inf_{num_inferences}_batch_{batch}'
    session_etglow = ort.InferenceSession(modelpath, sess_options, providers=['EtGlowExecutionProvider'], provider_options=[poptions])
    results_imagenet_etglow = utils.test_with_images(imagespath, session_etglow, args)
    session_etglow.end_profiling()

    # Compare cpu and etglow results
    is_correct = utils.check_and_compare(results_imagenet_cpu[0], results_imagenet_etglow[0], batch, num_inferences)
    if not is_correct:
        raise RuntimeError('Error: cpu and etglow provider results are not equal!') 

    # Print cpu and etglow stats
    utils.print_img_classification_results('Reference CPU', labelspath, results_imagenet_cpu[0])
    utils.print_img_classification_results('ETSoC', labelspath, results_imagenet_etglow[0])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
