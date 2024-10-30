#!/usr/bin/env python3

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
from scipy.special import softmax

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


def get_in_out_names(session):
    """Get input and output data name of model"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return (input_name, output_name)


def check_equal_results(golden_results : np.ndarray, etsoc_results : np.ndarray, num_launches = 1):
    for i in range(num_launches):
        if (not check_equal_values(golden_results[i][0], etsoc_results[i][0])):
            return False
    return True


def check_equal_values(a_tensor : np.ndarray, b_tensor : np.ndarray):
    """Compare the results with reference outputs up to 4 decimal places"""
    try:
        np.testing.assert_almost_equal(np.array(a_tensor).flatten(), np.array(b_tensor).flatten(), 4, err_msg='test', verbose=True)
    except AssertionError as err:
        print(err)
        return False
    
    return True
    

def test_with_protobuf(protobufpath : Path, session : ort.InferenceSession):
    """Test pb inputs given by the owner of model"""
    num_datasets = 1
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    total_time = 0.0
    for i in range(num_datasets):
        input_tensor = load_pb_data(protobufpath / f'test_data_set_{i}/input_0.pb')
        start = time.time()
        output = session.run([output_name], {input_name: input_tensor})
        total_time += time.time() - start
        golden_output = load_pb_data(protobufpath / f'test_data_set_{i}/output_0.pb')

        if not check_equal_values(golden_output, output[0]):
            raise RuntimeError(f'Error: results do not match the protobuf golden ({input_tensor})!')
        
    return total_time


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

    print(f'----- {device_string} results -----')
    for launch_results in results:
        batch_results = launch_results[0]
        launch_time = launch_results[1]
        print('-----------------------------------------')
        for result in batch_results:
            sorted_idx = np.flip(np.squeeze(np.argsort(result)))

            # Details about top 1 label
            top_idx = str(np.argmax(result))
            top_label = labels[top_idx][1]
            print(f'Image classified as {top_label}! (took {launch_time * 1000:.2f}ms)')
            print('Top 3 labels: ', end='')
            # List the top 3
            for i in range(3):
                confidence = f"{result[sorted_idx[i]] * 100:.2f}%"
                lbl = labels[str(sorted_idx[i])][1]
                print(f'{lbl}({confidence}) ', end = '')
            print('\n')
        break # Print only first results

def test_with_tensor(tensorspath : Path, session : ort.InferenceSession, args : ArgumentParser):
    output_tensors_names = [tensor.name for tensor in session.get_outputs()]

    input_tensors_list = []
    
    for i in range(args.launches):
        input_tensors = {}
        for input in session.get_inputs():
            input_tensors[input.name] = np.array([], dtype = np.int64)
        for b in range(args.batch):
            current_tensorspath = f'{tensorspath}/inference-{str((i+b)%128)}'
            for input in session.get_inputs():
                values = np.fromfile(f'{current_tensorspath}/{input.name}.bin', dtype = np.int64)
                input_tensors[input.name] = np.append(input_tensors[input.name], values)
        for input in session.get_inputs():
            input_tensors[input.name] = input_tensors[input.name].reshape((args.batch, 128))
        
        input_tensors_list.append(input_tensors)

    results = []
    #Launching a warm-up run is recommended to avoid initialization overheads when measuring performance
    if (args.warm_up):
        if (args.mode == "sync"):
            output_tensors = session.run(output_tensors_names, input_tensors_list[0])
        else:
            handle = AsyncRunHandle(1)
            session.run_async(output_tensors_names, input_tensors_list[0], callback, handle)
            handle.wait()
        
    if (args.mode == "sync"):
        start_time = time.time()
        for i in range(args.launches):
            start_launch_time = time.time()
            output_tensors = session.run(output_tensors_names, input_tensors_list[i])
            end_launch_time = time.time()
            results.append([output_tensors, (end_launch_time - start_launch_time)])
        end_time = time.time()            
    else:
        async_handles = []
        start_time = time.time()
        for i in range(args.launches):
            handle = AsyncRunHandle(i)
            async_handles.append(handle)
            session.run_async(output_tensors_names, input_tensors_list[i], callback, handle)
        
        #Wait for al async inferences to complete
        for handle in async_handles:
            handle.wait()

        end_time = time.time()
        # print(f'Model output tensor(s) names: {output_tensors_names}')
        for handle in async_handles:
            results.append([handle.get_results(), handle.get_duration()])

    return input_tensors_list, results, (end_time - start_time)


def test_with_images(imagespath : Path, session : ort.InferenceSession, args):
    """Test current session model against real inputs."""
    in_name, out_name = get_in_out_names(session)

    # Load image files
    images = load_image_files(imagespath / 'images')
    image_batch = []
    images_list = list(images.keys())
    for id in range(args.batch):
        next_image = images[images_list[id%len(images_list)]]
        image_batch.append(next_image['data'][0,:,:,:])
    input_batch = np.array(image_batch)

    # Run inferences in sync or async mode
    results = []
    if (args.mode == "sync"):
        
        if (args.warm_up):
            output_batch = session.run([out_name], {in_name: input_batch})

        total_start = time.time()
        for id in range(args.launches):
            start = time.time()
            output_batch = session.run([out_name], {in_name: input_batch})
            end = time.time()
            # Post-process
            results.append([softmax(output_batch[0], axis=1), end - start])
        total_end = time.time()

    elif (args.mode == "async"):
        async_handles = []

        if (args.warm_up):
            handle = AsyncRunHandle(1)
            session.run_async([out_name], {in_name: input_batch}, callback, handle)
            handle.wait()

        total_start = time.time()
        for id in range(args.launches):
            handle = AsyncRunHandle(id)
            async_handles.append(handle)
            session.run_async([out_name], {in_name: input_batch}, callback, handle)
        # Wait for all async inferences to complete
        for handle in async_handles:
            handle.wait()
        total_end = time.time()

        # Post-process (softmax)
        for handle in async_handles:
            output_batch = handle.get_results()
            results.append([softmax(output_batch[0], axis=1), handle.get_duration()])

    return results, (total_end - total_start)

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

def get_common_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-a", "--artifacts", default="../../../DownloadArtifactory", 
                        type=Path)
    parser.add_argument("-b", "--batch", 
                        help = "specify the number of batches", 
                        type = check_positive, default = 1)
    parser.add_argument("-m", "--mode", choices = ["sync","async"], 
                        help = "specify the runmode",
                        type = str, default="sync")
    parser.add_argument("-l", "--launches", 
                        help = "Defines the total number of executions to do",
                        type = check_positive, default=1)
    parser.add_argument("-w", "--warm-up", action = 'store_true',
                        help='Skip first session run for getting accurate measures on run session')
    parser.add_argument("-t", "--enable-tracing", action='store_true',
                        help = 'Enable onnxruntime profiling and neuralizer traces')
    parser.add_argument("-v", "--verbose",
                        help = "It shows help info messages",
                        type = bool, default=False)
    return parser

def extra_arguments(parser : ArgumentParser) -> ArgumentParser:
    parser.add_argument("--fp16", action = 'store_true',
                        help = 'Force the use of 16-bit floating point values when true')
    return parser
