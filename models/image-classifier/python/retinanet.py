"""Inference retinanet with ONNX Runtime"""
# pylint: disable=import-error,import-outside-toplevel
import os
import sys
from pathlib import Path
import onnxruntime
import onnx
import numpy as np

# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import utils


def get_args():
    """
    Get arguments using ArgumentParser.
    Returns:
        parser.parse_args()
    """
    parser = utils.get_common_arg_parser()
    return parser.parse_args()


def get_etglow_api_params() -> str:
    """
    Get the etglow api params.
    Returns:
        str with the api params.
    """
    api_params = [
        "device-type=silicon",
        "glow-threads=2",
        #"trace-neuralizer-nodes=true",
        "runDir=myrundir2",
        "extra-etsoc-params='" + '|'.join([
            'debug-glow=0',
            'dev=' + " ".join([
                '--gccCompileThreads=32',
                '--logDisableCodeGenBits=-1',
                '--enableGraphLogs=None',
            ]),
        ]) + "'"
    ]
    return ';'.join(api_params)


def get_onnx_symbols() -> dict:
    """
    Defines dictionaries of onnx symbols.
    Returns:
        dict with the onnx symbols
    """
    symbols = {
        "batch": 1,
    }
    return symbols


def get_onnx_shape_params() -> str:
    """
    Define onnx shape parameters.
    Returns:
        str: Return generated string without the last comma character
    """
    onnx_shape_params = ''
    onnx_symbols = get_onnx_symbols()
    for key, value in onnx_symbols.items():
        onnx_shape_params += f"{key}={value},"
    # Return generated string without the last comma character
    return onnx_shape_params[:-1]


def main():
    args = get_args()
    # Paths
    artifacts_path = Path(args.artifacts)
    reponame = 'retinanet-resnet50-fp16-onnx'
    modelname = 'retinanet-resnet50-fp16-512.onnx'
    modelpath = artifacts_path / f'models/{reponame}/{modelname}'


    sess_options = onnxruntime.SessionOptions()

    # Set graph optimization level
    sess_options.enable_profiling = True
    sess_options.log_severity_level = 2
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Generates some glow specific parameters
    onnx_shape_params = get_onnx_shape_params()
    api_params = get_etglow_api_params()
    print("api_params: " + api_params)
    provider_options_dict = {"etglow_greedy": "true",
                            "etglow_compile_only": "false",
                            "etglow_dump_subgraphs": "false",
                            "etglow_onnx_shape_params": onnx_shape_params,
                            "etglow_api_params": api_params}

    from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
    onnx_symbols = get_onnx_symbols()
    print(onnx_symbols)
    model = onnx.load(modelpath, load_external_data=False)
    for key, value in onnx_symbols.items():
        make_dim_param_fixed(model.graph, key, value)
    model = model.SerializeToString()
    new_model = os.path.join(os.path.dirname(modelpath), "model-fixed-dims.onnx")
    with open(new_model, "wb") as f:
        f.write(model)

    # Creates the ONNX Runtime session
    #We input a zero tensor, can be relaced by any value
    inp = np.zeros([1, 3, 512, 512], dtype='float16')
    session_retinanet_et = onnxruntime.InferenceSession(
        new_model,
        providers=['EtGlowExecutionProvider'],
        provider_options=[provider_options_dict],
        sess_options=sess_options
    )

    session_retinanet_cpu = onnxruntime.InferenceSession(
        new_model,
        providers=['CPUExecutionProvider'],
    )

    results_et = session_retinanet_et.run(None, {"input": inp})[0]
    results_cpu = session_retinanet_cpu.run(None, {"input": inp})[0]

    assert results_et.shape == results_cpu.shape
    assert np.allclose(results_et, results_cpu, atol=0.000061) is True


if __name__ == "__main__":
    main()
