#!/usr/bin/env python3

import numpy as np
import onnxruntime as ort
import os
import sys
import cv2
from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional
from pathlib import Path

# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import utils


def get_in_out_names(session):
    """Get input and output data name of model"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return (input_name, output_name)    


def load_mnist_test(test_path):
    """Load images to be tested"""

    extension = '.png'  

    im_files=[]
    im_files_path=[]

    for i in test_path:
        im_files = os.listdir(i)
        for j in im_files:
            im_files_path.append(os.path.join(i, j))

    testset = {}

    for f in im_files_path:
        if f.endswith(extension):
            testset[f] = {}
            img = cv2.imread(f)
            #When translating a color image to grayscale (mode “L”), the library uses the ITU-R 601-2 luma transform
            #L = R * 299/1000 + G * 587/1000 + B * 114/1000
            img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
            img = cv2.resize(img, dsize=(28, 28), interpolation = cv2.INTER_AREA).astype(np.float32)
            img.resize((1,1,28,28))    
            testset[f]['data'] = img
            testset[f]['result'] = None
            testset[f]['resultcpu'] = None

    return(testset)


def checkresults(testset):
    """Compare result between cpu and provider"""
    
    for k, v in testset.items():
        if testset[k]['result'] == testset[k]['resultcpu'] :
            print(f'Image {k} is OK! .. \tidentified as number {testset[k]["result"]}')
        else:
            print("****************")
            print(f'Image {k} FAILS:')
            print(f'\tExpected result is {testset[k]["resultcpu"]} \n\tobtained in provider {testset[k]["result"]}')
            print("****************")


def test_images(imagespath : Path, session, sessionEtsoc):
    """Test current session model against real inputs."""

    test_path = [ imagespath / 'images_1-1-28-28', imagespath / 'images_3-400-640']
    
    in_name, out_name = get_in_out_names(session)

    # Load real images
    testset = load_mnist_test(test_path)    
    for k,v in testset.items():
        result = session.run([out_name], {in_name: testset[k]['data']})
        testset[k]['resultcpu'] = int(np.argmax(np.array(result).squeeze(), axis = 0))

        result = sessionEtsoc.run([out_name], {in_name: testset[k]['data']})
        testset[k]['result'] = int(np.argmax(np.array(result).squeeze(), axis = 0))

    checkresults(testset)


def main(argv: Optional[Sequence[str]] = None):    
    """Launch MNIST onnx model over cpu and etglow and compare results."""
    parser = utils.get_arg_parser()
    args = parser.parse_args(argv)

    artifacts_path = Path(args.artifacts) 

    modelname = 'mnist' # resnet50_onnx # resnet50_denso_onnx
    modelpath = artifacts_path / f'models/{modelname}/model.onnx'
    imagespath = artifacts_path /'input_data/images/'
    protobufpath =  artifacts_path / 'input_data/protobuf/mnist/'

    sess_options = ort.SessionOptions()
    utils.set_verbose_output(sess_options, args.verbose)

    session_cpu    = ort.InferenceSession(modelpath, sess_options, providers=['CPUExecutionProvider'])
    session_etglow = ort.InferenceSession(modelpath, sess_options, providers=['EtGlowExecutionProvider'])

    test_images(imagespath, session_cpu, session_etglow)

    utils.test_with_protobuf(protobufpath, session_cpu)
    utils.test_with_protobuf(protobufpath, session_etglow)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
