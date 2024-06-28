#!/usr/bin/env python3

import os
import cv2
import json
import time
import numpy as np
import onnx
import sys
from onnx import numpy_helper
import onnxruntime as ort
from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional
from pathlib import Path

import threading

class AsyncRunHandle:
    def __init__(self, id):
        self.__event = threading.Event()
        self.__id = id
        self.__results = []
        self.__err = ''
        self.__event.clear()
        self.__start = time.time()
        
    def get_id(self):
        return self.__id

    def get_results(self):
        return self.__results
    
    def get_err(self):
        return self.__err

    def get_duration(self):
        return self.__duration

    def save_outputs(self, res, err) -> None:
        self.__results = res
        self.__err = err
        self.__event.set()
        self.__duration =  time.time() - self.__start

    def wait(self):
        self.__event.wait()
        
def callback(out: np.ndarray, user_data: AsyncRunHandle, err : str) -> None :
    if err:
        async_run_id = user_data.get_id()
        raise RuntimeError(f"async_run (id:{async_run_id}) failed with the following error: {err}")

    user_data.save_outputs(out, err)


def load_pb_data(protobufpath : Path) -> np.ndarray:
    """Load protobuf test dataset"""
    pb_data = []
    tensor = onnx.TensorProto()
    with open(protobufpath, 'rb') as f:
        tensor.ParseFromString(f.read())
        pb_data = numpy_helper.to_array(tensor)

    return pb_data


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def postprocess(result):
    return softmax(np.array(result)).tolist()


def get_in_out_names(session):
    """Get input and output data name of model"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return (input_name, output_name)


def check_and_compare(golden_outputs : np.ndarray, etsoc_outputs : np.ndarray, batch = 1, num_inferences = 1):
    """Compare the results with reference outputs up to 4 decimal places"""
    try:
        for inf in range(num_inferences):
            for i in range(batch):
                np.testing.assert_almost_equal(np.array(golden_outputs[inf][0][i]), np.array(etsoc_outputs[inf][0][i]), 4, err_msg='test', verbose=True)
    except AssertionError:
        return False
    
    return True
    

def test_with_protobuf(protobufpath : Path, session : ort.InferenceSession):
    """Test pb inputs given by the owner of model"""
    num_datasets = 1
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    results = []
    golden_results = []
    for i in range(num_datasets):
        input_tensor = load_pb_data(protobufpath / f'test_data_set_{i}/input_0.pb')
        start = time.time()
        batch_result = session.run([output_name], {input_name: input_tensor})
        end = time.time()
        results.append([batch_result, end - start])
        golden_results.append([[load_pb_data(protobufpath / f'test_data_set_{i}/output_0.pb')], 0.0])

    if not check_and_compare(golden_results, results):
        raise RuntimeError('Error: results do not match the protobuf golden!')
        
    return results


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def preprocess(input_data):
    # convert the input data to float32 datatype
    img_data = input_data.astype('float32')

    # normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

    # Add batch dimension
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
            
    return norm_img_data


def load_image_files(images_path : Path) -> dict:
    """Load .png images from the specified directory"""

    if not images_path.exists():
        raise FileNotFoundError(f"The path {images_path} does not exist.")

    images = {}
    for f in images_path.iterdir():
        if f.is_file() and f.suffix == '.png':
            images[f] = {}
            img_rgb = cv2.imread(str(f))
            img_brg = np.array(img_rgb).transpose(2, 0, 1)
            images[f]['data'] = preprocess(img_brg)
            images[f]['result'] = None
            images[f]['resultcpu'] = None

    return(images)


def print_img_classification_results(device_string, labels_path, results):
    """Print image classification results including inference execution time and confidence percentage"""
    labels = load_json(labels_path)

    print(f'--- {device_string} results ---')
    for inference_results in results:
        batch_results = inference_results[0]
        inference_time = inference_results[1]
        print('-----------------------------------------')
        for result in batch_results:
            top_idx = str(np.argmax(result))
            sorted_idx = np.flip(np.squeeze(np.argsort(result)))    

            print(f'Image classified as {labels[top_idx][1]}! (took {inference_time * 1000:.2f}ms)')
            print('Top 3 labels: ', end='')
            for i in range(3):
                confidence = f"{result[sorted_idx[i]] * 100:.2f}%"
                print(f'{labels[str(sorted_idx[i])][1]}({confidence}) ', end = '')
            print('\n')


def test_with_tensor(tensorspath : Path, session : ort.InferenceSession):
    output_tensors_names = [tensor.name for tensor in session.get_outputs()]

    input_tensors = {}
    for input in session.get_inputs():
        input_tensors[input.name] = np.fromfile(f'{tensorspath}/{input.name}.bin', dtype=np.int64)
        input_tensors[input.name] = input_tensors[input.name].reshape((1, 128))
        # print(f'Loaded input tensor {input.name} - shape: {input_tensors[input.name].shape}')
    
    output_tensors = session.run(output_tensors_names, input_tensors)
    # print(f'Model output tensor(s) names: {output_tensors_names}')

    return input_tensors, output_tensors


def test_with_images(imagespath : Path, session : ort.InferenceSession, batch = 1, num_inferences = 1, mode = "sync"):
    """Test current session model against real inputs."""
    in_name, out_name = get_in_out_names(session)

    # Load image files
    images = load_image_files(imagespath / 'images')
    image_batch = []
    images_list = list(images.keys())
    for id in range(batch):
        next_image = images[images_list[id%len(images_list)]]
        image_batch.append(next_image['data'][0,:,:,:])
    batch_data = np.array(image_batch)

    # Run inferences in sync or async mode
    results = []
    if (mode == "sync"):
        for id in range(num_inferences):
            start = time.time()
            batch_results = session.run([out_name], {in_name: batch_data})
            end = time.time()
            # Post-process
            results.append([[postprocess(result) for result in batch_results], end - start])

    elif (mode == "async"):
        async_handles = []
        for id in range(num_inferences):
            handle = AsyncRunHandle(id)
            async_handles.append(handle)
            session.run_async([out_name], {in_name: batch_data}, callback, handle)

        # Wait for all async inferences to complete
        for handle in async_handles:
            handle.wait()

        # Post-process (softmax)
        for handle in async_handles:
            batch_results = handle.get_results()
            results.append([[postprocess(result) for result in batch_results], handle.get_duration()])

    return results

def set_verbose_output(options, enabled):
    log_severity_verbose = 0
    log_severity_warning = 2
    log_severity_error = 3
    
    if not enabled:
        ort.set_default_logger_severity(log_severity_error)
        options.log_severity_level = log_severity_error
    else:
        ort.set_default_logger_severity(log_severity_verbose)
        options.log_severity_level = log_severity_warning

def check_positive(value):
    try:
        value = int(value)
        if value <= 0 or value > 10000:
            raise argparse.ArgumentTypeError(f'{value} is not a positive integer.')
    except ValueError:
        raise Exception(f'{value} is not an integer.')
    return value

def get_arg_parser(argv: Optional[Sequence[str]] = None) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-a", "--artifacts", default="../../../DownloadArtifactory", type=Path)
    parser.add_argument("-t", "--enable-tracing", action='store_true')
    parser.add_argument("-v", "--verbose", default=False)
    return parser

def get_img_classifier_arg_parser(argv: Optional[Sequence[str]] = None) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-a", "--artifacts", default="../../../DownloadArtifactory")
    parser.add_argument("-v", "--verbose",
                        help = "It shows help info messages",
                        type = bool, default=False)
    parser.add_argument("-b", "--batch", 
                        help = "specify the number of batches", 
                        type = check_positive, default = 1)
    parser.add_argument("-i", "--image", 
                        help = "specify the image to do batch or/and inference", 
                        type = str, default="")
    parser.add_argument("-m", "--mode", choices = ["sync","async"], 
                        help = "specify the runmode",
                        type = str, default="sync")
    parser.add_argument("-t", "--total-inferences", 
                        help = "Defines the total number of inferences to do",
                        type = check_positive, default=1)
    parser.add_argument("-d", "--delete-tensor",
                        help = "remove tensor once run have been done. Only in async mode",
                        type = bool, default=False)
    parser.add_argument("--enable-tracing", action='store_true')

    return parser


def get_tracing_params(): 
    return ';'.join(["trace-neuralizer-nodes=true", "trace-neuralizer-path=neuralizer-trace.json",
                     "trace-runtime-path=runtime-trace.json"])
