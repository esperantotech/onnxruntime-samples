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
        "etglow_onnx_shape_params": f'N={args.batch}'
    }

    api_params = ""
    if args.enable_tracing:
        api_params += utils.get_tracing_params()

    if api_params:
        poptions["etglow_api_params"] =  api_params

    return poptions

def extra_arguments(parser):
    parser.add_argument("-p", "--performance", action = 'store_true',
                        help='Performance are calculate over images on silicon skiping other executions.')    

def main(argv: Optional[Sequence[str]] = None):
    """Launch RESNET50 onnx model on cpu and etglow providers and compare results."""
    parser = utils.get_img_classifier_arg_parser()   
    extra_arguments(parser)
    args = parser.parse_args(argv)

    batch = args.batch
    num_launches = args.launches

    # Paths
    artifacts_path = Path(args.artifacts)
    modelname = 'resnet50-v2-7'
    modelpath = artifacts_path / f'models/{modelname}/model.onnx'
    imagespath = artifacts_path /'input_data/images/imagenet/'
    protobufpath =  artifacts_path / 'input_data/protobuf/resnet50-v2-7/'
    labelspath = imagespath / 'index_to_name.json'

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
    
    if (not args.performance):
        # Run cpu provider inferences
        sess_options.profile_file_prefix = f'{modelname}_cpu_inf_{num_launches}_batch_{batch}'
        session_cpu    = ort.InferenceSession(modelpath, sess_options, providers=['CPUExecutionProvider'])
        results_proto_cpu = utils.test_with_protobuf(protobufpath, session_cpu)
        results_imagenet_cpu, cpu_inf_sec = utils.test_with_images(imagespath, session_cpu, args)
        session_cpu.end_profiling()

        # Run etglow provider inferences
        sess_options.profile_file_prefix = f'{modelname}_etglow_inf_{num_launches}_batch_{batch}'
        session_etglow = ort.InferenceSession(modelpath, sess_options, providers=['EtGlowExecutionProvider'], provider_options=[poptions])
        results_proto_etglow = utils.test_with_protobuf(protobufpath, session_etglow)
        results_imagenet_etglow, et_total_time = utils.test_with_images(imagespath, session_etglow, args)
        session_etglow.end_profiling()

        # Compare cpu and etglow results
        is_correct = utils.check_and_compare(results_imagenet_cpu, results_imagenet_etglow, batch, num_launches)
        if not is_correct:
            raise RuntimeError('Error: cpu and etglow provider results are not equal!') 
        
        # Print cpu and etglow stats
        utils.print_img_classification_results('Reference CPU', labelspath, results_imagenet_cpu)
        print(f'Protobuf test took {results_proto_cpu[0][1]:.3f}s\n')
        utils.print_img_classification_results('ETSoC', labelspath, results_imagenet_etglow)
        print(f'Protobuf test took {results_proto_etglow[0][1]:.3f}s\n')
    else:
        sess_options.profile_file_prefix = f'{modelname}_etglow_inf_{num_launches}_batch_{batch}'
        session_etglow = ort.InferenceSession(modelpath, sess_options, providers=['EtGlowExecutionProvider'], provider_options=[poptions])
        results_imagenet_etglow, et_total_time = utils.test_with_images(imagespath, session_etglow, args)
        session_etglow.end_profiling()

        print(f'Total launches {num_launches} batch used {batch} in {et_total_time} s.')
        print(f'ET_provider Performance: {(num_launches*batch)/et_total_time:.4f} inf/sec')

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
