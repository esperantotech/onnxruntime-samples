#!/usr/bin/env python3

import onnxruntime as ort
import sys
from typing import Sequence, Optional
from pathlib import Path

# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import utils


def main(argv: Optional[Sequence[str]] = None):
    """Launch vgg16 onnx model on cpu and etglow and compare results."""
    parser = utils.get_arg_parser()
    args = parser.parse_args(argv)

    artifacts_path = Path(args.artifacts)
    modelname = 'vgg16_bn_denso_onnx'
    modelpath = artifacts_path / f'models/{modelname}/model.onnx'
    imagespath = artifacts_path /'input_data/images/imagenet/'

    sess_options = ort.SessionOptions()
    utils.set_verbose_output(sess_options, args.verbose)

    poptions = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": "batch=1;height=224;width=224"
    }

    session_cpu    = ort.InferenceSession(modelpath, sess_options, providers=['CPUExecutionProvider'])
    session_etglow = ort.InferenceSession(modelpath, sess_options, providers=['EtGlowExecutionProvider'], provider_options=[poptions])

    print('*** Reference CPU results ***')
    utils.test_with_images(imagespath, session_cpu)
    print('*** ETSoC results ***')
    utils.test_with_images(imagespath, session_etglow)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
