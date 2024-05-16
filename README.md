# ONNXRuntime Samples


This repository contains a set of examples on how to run onnxruntime
stack using Esperanto **EtGlowExecutionProvider** provider with a set of popular models.
Exectuions with  CPU EP are also provided for comparision.

[Wip] The MNIST and RESNET50 model will be the first models to be checked it and the
models supported by our techdemo will be coming soon. 

## Repository Organization
We have a common folder at root directory called model with differents subfolder
as many as generic subtypes of models we have. For each subtype model we found a python 
and C++ folders which represents the API to be used for running on ONNX. 
Inside the API folder we find source code or a script following this naming structure,
model name and suffix <model>.py or <model>.cpp It depends on which folder belongs.

Models and datasets used by executables are downloading from artifactory server
so that DownloadArtifactory folder is created once we execute the following cli inside de docker.
<artifacts_mgr_client --inputfile artifacts/models.yaml --artifactpath DownloadArtifactory>

- **README.md**
- **<artifacts>**
- **<DownloadArtifactory>**
- **<models>** 
  - **<modelSubTypeFamily_X>**
    - **python**
    - **c++**  
  - **<modelSubTypeFamily_Y>**
    - **python**
    - **c++**  
  - **<modelSubTypeFamily_Z>**
    - **python**
    - **c++**  
  - ...

## Before start
Once you been inside the docker you should execute the cli bellows to set properly the environment:
<sudo pip3 install artifacts_mgr_client --index-url https://sc-artifactory1.esperanto.ai/artifactory/api/pypi/pypi-virtual/simple>
<sudo apt-get install -y ffmpeg libsm6 libxext6 --no-install-recommends>
<sudo pip install onnx>
<sudo pip install opencv-python>


## Execute models in Python API.
Into the desired  models/modelSubTypeFamily_X/python folder all the python scripts can be executed as 
\$ <model_name>.py
You can also specify where you have your download dataset otherwise It is assuming a relative path ../../../DownloadArtifactory.

## Execute model in C++ API.
Previously to execute you should to compile the code models, to do that:
1. Create a `build` folder in models/modelSubTypeFamily_X/c++ path
2. Execute the following cli inside, where `build_type` could be "`Release`" or "`Debug`".
```code
\$ cmake -S .. -B . -DCMAKE_TOOLCHAIN_FILE="/usr/local/esperanto/.builds/host/conan_toolchain.cmake" -DCMAKE_BUILD_TYPE=<build_type> -G Ninja
\$ cmake --build .
```code

3. Launch the execution, choose the model with the following paremeters:
```code
\$ ./<modelname>  -a <path_to_yout_downloaded_artifactory> -mf <path model_json_description_filename.json>
```code

The <model_json_description_filename.json> contains the features of how the model wants the image inference input, so it describes
the name, class of model, format, model_file_name, layout, number of channels, etc..... (more detailed has to be provided here.)
