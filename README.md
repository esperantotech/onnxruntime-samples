# ONNXRuntime Samples

This repository contains examples that demonstrate how to use onnxruntime with **EtGlowExecutionProvider** provider with a set of popular models.
Executions with  CPU EP are also provided for comparison.

## Repository Organization
We have a common folder at root directory called model with differents subfolder as many as generic subtypes of models we have. 
For each subtype model we found a python and C++ folders which represents the API to be used for running on ONNX. 
Inside the API folder we find source code or a script following this naming structure, model name and suffix <model>.py or <model>.cpp It depends on which folder belongs.

```
.
├── README.md
├── DownloadArtifactory
│   ├── input_data
│   │   ├── images
│   │   │   ├── imagenet
│   │   │   │   ├── images
│   │   │   │   │   ├── 1_cat_285.png
│   │   │   │   │   ├── 2_dog_207.png
│   │   │   │   │   └── 3_zebra_340.png
│   │   │   │   ├── index_to_name.json
│   │   │   │   ├── labels.txt
│   │   │   │   ├── positions.txt
│   │   │   │   └── true_predictions.txt
│   │   │   └── ...
│   │   └── protobuf
│   │       ├── ...
│   │       └── ...
│   │           └── test_data_set_0
│   │               ├── input_0.pb
│   │               └── output_0.pb
│   └── models
│       ├── mnist
│       │   ├── metadata.json
│       │   └── model.onnx
│       ├── mobilenet
│       │   ├── metadata.json
│       │   └── model.onnx
│       ├── resnet50
│       │   ├── metadata.json
│       │   └── model.onnx
│       ├── ...
│       └── vgg16_..
│           ├── metadata.json
│           └── model.onnx
├── artifacts
│   ├── models.yaml
│   └── squad-dataset-to-input-tensor.py
├── models
│   ├── common
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── image-classifier
│   │   ├── c++
│   │   │   ├── CMakeLists.txt
│   │   │   ├── Config.h.in
│   │   │   ├── PngHelp
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   ├── ...
│   │   │   │   └── ...
│   │   │   ├── mnist.cpp
│   │   │   └── resnet.cpp
│   │   └── python
│   │       ├── mnist.py
│   │       ├── mobilenet.py
│   │       ├── resnet50.py
│   │       └── vgg16.py
│   └── language-model
│       └── python
│           └── bert.py
├── requirements.txt
└── sonar-project.properties
```

## Prepare development environment
The onnxruntime-samples repository requires a system that has the Esperanto SDK pre-installed. 
The SDK is a set of utilities and tools that allow the user to transparently use the Esperanto SW stack, including the onnxruntime esperanto fork.

### Getting sources
First step is to get the onnxruntime-samples sources.
```
git clone git@gitlab.com:esperantotech/software/onnxruntime-samples.git
cd onnxruntime-samples/
```

### Start dockerized environment
Assuming you have a `sw-platform` installation in your `$HOME`:
```
./sw-platform/dock.py --image=convoke/ubuntu-22.04-et-sw-develop-stack prompt
```

## Installing dependencies
Before installing datasets we must ensure `artifacts_mgr_client` (and other python packages) is installed.
We also need to install some system-level dependencies required for C++ samples.

```
export PATH=$PATH:$HOME/.local/bin
pip3 install -r requirements.txt --index-url https://sc-artifactory1.esperanto.ai/artifactory/api/pypi/pypi-virtual/simple
sudo apt update
sudo apt install -y --no-install-recommends ffmpeg libsm6 libxext6 
```

### Getting models and datasets
Then we can download all models and datasets required by the examples. 
The yaml file `artifacts/models.yaml` contains the bill of materials to be retrieved from artifactory.
Samples assume by default that artifacts will be placed in `DownloadArtifactory` directory but the user can customize the install location and later use script parameter to point to the download location.
```
artifacts_mgr_client --inputfile artifacts/models.yaml --artifactpath DownloadArtifactory
```

## Build instructions
### Python examples
Python examples do not require any kind of build step.

### C++ examples
To execute C++ examples we need to build them first.
1. Verify you have a ET_SDK_HOME envvar pointing to `/usr/local/esperanto`
```
$ env | grep ET_SDK_HOME
ET_SDK_HOME=/usr/local/esperanto
```
2. Now you can configure the project passing (1) the source directory location, (2) the build directory location, (3) the location of the toolchain file, (4) the build type and (5) the underlying build system (ninja):
```
cmake -S models/image-classifier/c++/ -B models/image-classifier/c++/build -DCMAKE_TOOLCHAIN_FILE=$ET_SDK_HOME/.builds/host/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -G Ninja
```
3. Compile the examples:
```
cmake --build models/image-classifier/c++/build/
```


## Execution instructions

**OBS: you will need to have access to Esperanto EtSoC-1 accelerators to be able to successfully execute**

Reduce Esperanto Compiler verbosity with
```
export GLOG_minloglevel=2
```

## Execute models with Python API.

To execute python script we need to be inside `models/<model-sub-family>/python`.
Models can be executed as a script
Into the desired  models/modelSubTypeFamily_X/python folder all the python scripts can be executed as 
\$ <model_name>.py
You can also specify where you have your download dataset otherwise It is assuming a relative path ../../../DownloadArtifactory.

## Execute model in C++ API.
Launch the execution, choose the model with the following paremeters:
```
./<modelname>  -a <path_to_yout_downloaded_artifactory> -mf <path model_json_description_filename.json>
```

The <model_json_description_filename.json> contains the features of how the model wants the image inference input, so it describes
the name, class of model, format, model_file_name, layout, number of channels, etc..... (more detailed has to be provided here.)

It might be interesting to firstly call `-h` to see what paramters each example accepts:
```
./models/image-classifier/c++/build/mnist -help
mnist: Usage: example-function [OPTION] ...

  Flags from /home/rafa/WorkSpace/onnxruntime-samples/models/image-classifier/c++/mnist.cpp:
    -artifact_folder (Absolute folder where the models and datasets are)
      type: string default: "../../../DownloadArtifactory"
    -h (help) type: bool default: false
    -verbose (Verbose output) type: bool default: false
```

Example execution:
```
$ ./models/image-classifier/c++/build/mnist -artifact_folder /home/rafa/WorkSpace/onnxruntime-samples/DownloadArtifactory -verbose
artifacts folder placed on ./DownloadArtifactory/
EtGlowExecutionProvider
CPUExecutionProvider
WARNING: Logging before InitGoogleLogging() is written to STDERR
E0527 08:46:38.464800  3909 ETSOCDeviceManager.cpp:237] User requested memory: 32000000000 (bytes) is larger than available device memory: 17085497344 (bytes). Lowering max memory requirements to fit device memory size.
2024-05-27 08:46:38.466365740 [W:onnxruntime:Default, etglow_execution_provider_bundle.h:92 operator()] Not incorporating initialized tensors (weights) as part of the bundle hash. Be aware! consider using ORT_ETGLOW_BUNDLE_CACHE_ENABLE=0 envvar
[08:46:41.251] CPU and EtGlowProvider matches the result: 0  for image: ./DownloadArtifactory/input_data/images/images_1-1-28-28/0_1009.png
[08:46:42.835] CPU and EtGlowProvider matches the result: 1  for image: ./DownloadArtifactory/input_data/images/images_1-1-28-28/1_1008.png
[08:46:44.427] CPU and EtGlowProvider matches the result: 2  for image: ./DownloadArtifactory/input_data/images/images_1-1-28-28/2_1065.png
```


### Known pitfalls and possible solutions

1. When downloading models from AWS S3 this message can occur if credentials are not properly configured:
```
ERROR: Cannot generate s3 presign url for s3://et-sw-sdk-release-artifacts/sw-ci-inputs/...
ERROR: Unable to locate credentials. You can configure credentials by running "aws configure"
```
Copy `.aws/credentials` file form user `et` home  to your home folder:  `mkdir -p $HOME/.aws && cp /home/et/.aws/credentials $HOME/.aws/credentials`
