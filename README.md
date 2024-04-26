# ONNXRuntime Samples


This repository contains a set of models and shows how to run them over onnxruntime
stack using the CPU as well as our **EtGlowExecutionProvider** onnx provider.
Each model uses the latest model populated at https://github.com/onnx/models 

[Wip] The MNIST and RESNET50 model will be the first models to be checked it and the
models supported by our techdemo will be coming soon. 

## Repository Organization
Each folder at root directory has the name model to be executed.
Inside we find basically one folder for python onnxruntime API example and other c++
folder for c++ onnxruntime API . Apart from that, there is a tarball file with
the latest model available and some pb files test and also sometimes another
folder with more files to be used as tests.

- **<modelName>** Set of utilities to launch execution and verification.
  - **python**
  - **c++**
  - **<testfiles>**  
  - **<modelName.tar.gz>**  Models to be use as an examples.
