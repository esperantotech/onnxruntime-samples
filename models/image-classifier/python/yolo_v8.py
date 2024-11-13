"""
File that contains the script to test yolo_v8 vs coco dataset
"""
# pylint: disable=line-too-long,import-error,redefined-outer-name,import-outside-toplevel
import sys
from typing import Sequence, Optional
from pathlib import Path
import onnxruntime
import onnx
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    parser.add_argument('--expected-result', nargs='+', type=str)
    parser.add_argument('--image', '-i', type=str, default='', help="path to the image to test")
    parser.add_argument('--output_name', '-o', metavar='STR',
                        default='output.jpg', help='Filenames of output images')
    parser.add_argument('--save-picture', type=bool,
                        default=False, help='Enable/Disable saving the output picture')
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
        "runDir=myrundir",
        "extra-etsoc-params='" + '|'.join([
            'debug-glow=0',
            'dev=' + " ".join([
                '--gccCompileThreads=32',
                '--logDisableCodeGenBits=-1',
                '--enableGraphLogs=none',
            ]),
        ]) + "'"
    ]
    return ';'.join(api_params)


#
def get_onnx_symbols(batch) -> dict:
    """
    Defines dictionaries of onnx symbols.
    Args:
        batch: batch to use
    Returns:
        dict with the onnx symbols
    """
    symbols = {
        "batch": batch,
        "n_selected": 1
    }
    return symbols


#
def get_onnx_shape_params(batch) -> str:
    """
    Define onnx shape parameters.
    Args:
        batch: batch to use
    Returns:
        str: Return generated string without the last comma character
    """
    onnx_shape_params = ''
    onnx_symbols = get_onnx_symbols(batch)
    for key, value in onnx_symbols.items():
        onnx_shape_params += f"{key}={value},"
    return onnx_shape_params[:-1]


def compile_yolov8(onnx_model_path, batch):
    """
    Compile yolov8 model
    Args:
        onnx_model_path: onnx model path
        batch: batch to use
    Returns:
        session_yolov8
    """
    # Disable all the graph optimizations
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    # Generates some glow specific parameters
    onnx_shape_params = get_onnx_shape_params(batch)
    api_params = get_etglow_api_params()
    print("api_params: " + api_params)
    provider_options_dict = {"etglow_greedy": "true",
                             "etglow_compile_only": "false",
                             "etglow_dump_subgraphs": "false",
                             "etglow_onnx_shape_params": onnx_shape_params,
                             "etglow_api_params": api_params}

    from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
    onnx_symbols = get_onnx_symbols(batch)
    print(onnx_symbols)
    model = onnx.load(onnx_model_path, load_external_data=False)
    for key, value in onnx_symbols.items():
        make_dim_param_fixed(model.graph, key, value)
    model = model.SerializeToString()

    # Creates the ONNX Runtime session
    session_yolov8 = onnxruntime.InferenceSession(model, providers=['EtGlowExecutionProvider'],
                                                  provider_options=[provider_options_dict])
    return session_yolov8


def main():
    """ Execute yolo v8 to detect objects and compare results """
    # Default values
    batch = 1
    detected_objects = []
    args = get_args()
    save_picture = args.save_picture
    # Paths
    artifacts_path = Path(args.artifacts)
    reponame = 'yolov8-fp16-onnx'
    modelname = 'yolov8-fp16.onnx'
    postmodelname = 'post-process.onnx'
    modelpath = artifacts_path / f'models/{reponame}/{modelname}'
    postpath = artifacts_path / f'models/{reponame}/{postmodelname}'

    # Detected classes by Yolov8
    classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
               8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
               14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
               22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
               29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
               35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
               40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
               48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
               55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
               62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
               69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
               76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(640),
        transforms.CenterCrop(640),
        transforms.ToTensor(),
    ])

    # Create ORT Session
    session_yolov8 = compile_yolov8(modelpath, batch)

    # Get image path
    image_path = args.image

    # Start object detection
    # It is always converting to RGB to let black and white images
    # to be processed. Take this into account in case printing/saving
    # the output picture
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(dim=0)

    input_dict = {}
    input_dict["images"] = image.numpy().astype("float16")
    output0 = session_yolov8.run(None, input_dict)[0]

    session_post_process_cpu = onnxruntime.InferenceSession(postpath, providers=['CPUExecutionProvider'])
    input_dict = {}
    input_dict["output0"] = output0.astype("float32")
    results = session_post_process_cpu.run(None, input_dict)

    if save_picture is True:
        preprocess_final = transforms.Compose([
            transforms.Resize(640),
            transforms.CenterCrop(640),
        ])
        image = Image.open(image_path)
        im = preprocess_final(image)
        # Create figure and axes
        _, ax = plt.subplots()

        # Display the image
        ax.imshow(im)
        low = results[0]-results[1]/2

    for i in range(len(results[0])):
        object_detected = classes[results[2][i][1]]
        detected_objects.append(object_detected)
        print(f"Detected: {object_detected}")
        try:
            # When removing an element from list, action result is None
            assert (object_detected in args.expected_result) is True
        except ValueError:
            print(f"object_detected: {object_detected} not in expected list: picture_dict['categories_str']")

        if save_picture is True:
            #Create a Rectangle patch
            rect = patches.Rectangle(low[i], results[1][i, 0], results[1][i, 1], linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            plt.text(low[i][0]+5, low[i][1]+20, object_detected)

    if save_picture is True:
        plt.savefig(args.output_name)


if __name__ == '__main__':
    main()
