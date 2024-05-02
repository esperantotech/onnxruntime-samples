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
<sudo apt-get install -y ffmpeg libsm6 libxext6 --no-install-recommends>
<sudo pip install onnx>
<sudo pip install opencv-python>
