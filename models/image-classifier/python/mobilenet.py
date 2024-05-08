#!/usr/bin/env python3

import onnxruntime as ort
import sys
from typing import Sequence, Optional
from common import utils
from pathlib import Path

def main(argv: Optional[Sequence[str]] = None):
    """Launch mobilenet onnx model on cpu and etglow and compare results."""
    args = utils.parse_args(argv)
   
    artifacts_path = Path(args.artifacts)
    modelname = 'mobilenet_v3_small_denso_onnx'  # modelname = 'mobilenetv2-7' # modelname = 'mobilenet_v2_denso_onnx'
    modelpath = artifacts_path / f'models/{modelname}/model.onnx'
    imagespath = artifacts_path /'input_data/images/imagenet/'

    log_severity_verbose = 0
    log_severity_warning = 2
    log_severity_error = 3
    ort.set_default_logger_severity(log_severity_warning)
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = log_severity_warning

    poptions = {}
    poptions['etglow_onnx_shape_params'] = "batch=1;height=224;width=224"

    session = ort.InferenceSession(modelpath)   
    session_etsoc = ort.InferenceSession(modelpath, providers=['EtGlowExecutionProvider'], provider_options=[poptions])

    print('*** Reference CPU results ***')
    utils.test_with_images(imagespath, session)
    print('*** ETSoC results ***')
    utils.test_with_images(imagespath, session_etsoc)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
