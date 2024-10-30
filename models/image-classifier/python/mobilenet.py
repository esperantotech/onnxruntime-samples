#!/usr/bin/env python3

import onnxruntime as ort
import sys
from typing import Sequence, Optional
from pathlib import Path

# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import utils

def get_provider_options(args) -> dict:
    poptions = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": f'batch={args.batch};height=224;width=224',
    }
    
    api_params = "glow-threads=4"
    if api_params:
        poptions["etglow_api_params"] = api_params

    return poptions


def main(argv: Optional[Sequence[str]] = None):
    """Launch mobilenet onnx model on cpu and etglow and compare results."""
    parser = utils.get_common_arg_parser()
    parser.add_argument("--performance", action = 'store_true')
    args = parser.parse_args(argv)
    batch = args.batch
    num_launches = args.launches

    # Paths
    artifacts_path = Path(args.artifacts)
    modelname = 'mobilenet_v3_small_denso_onnx'
    modelpath = artifacts_path / f'models/{modelname}/model.onnx'
    imagespath = artifacts_path /'input_data/images/imagenet/'
    labelspath = imagespath / 'index_to_name.json'

    # Session and provider options
    sess_options = ort.SessionOptions()
    if (args.mode == "async"):        
        sess_options.intra_op_num_threads=5
    print (f'Session intra-op num. threads: {sess_options.intra_op_num_threads}')

    utils.set_verbose_output(sess_options, args.verbose)
    sess_options.enable_profiling = args.enable_tracing
    poptions = get_provider_options(args)
    print(poptions)

    print('Executing inferences...\n')

    # ETSoC
    sess_options.profile_file_prefix = f'{modelname}_etglow_inf_{num_launches}_batch_{batch}'
    session_etglow = ort.InferenceSession(modelpath, sess_options, providers=['EtGlowExecutionProvider'], provider_options=[poptions])
    results_imagenet_etglow, et_total_time = utils.test_with_images(imagespath, session_etglow, args)
    session_etglow.end_profiling()

    print(f'Executed {num_launches} inferences with batch {batch} in {et_total_time:.4f}s.')
    print(f'ET_provider Performance: {(num_launches*batch)/et_total_time:.4f} inf/sec')

    if (not args.performance):
        # Run cpu provider inferences
        sess_options.profile_file_prefix = f'{modelname}_cpu_inf_{num_launches}_batch_{batch}'
        session_cpu = ort.InferenceSession(modelpath, sess_options, providers=['CPUExecutionProvider'])
        results_imagenet_cpu, time = utils.test_with_images(imagespath, session_cpu, args)
        session_cpu.end_profiling()
        
        # Compare cpu and etglow results
        is_correct = utils.check_equal_results(results_imagenet_cpu, results_imagenet_etglow, num_launches)
        if not is_correct:
            raise RuntimeError('Error: cpu and etglow provider results are not equal!') 

        # Print cpu and etglow stats
        utils.print_img_classification_results('Reference CPU', labelspath, results_imagenet_cpu)
        utils.print_img_classification_results('ETSoC', labelspath, results_imagenet_etglow)
    

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
