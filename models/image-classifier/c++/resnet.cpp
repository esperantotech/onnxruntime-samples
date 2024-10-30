/*-------------------------------------------------------------------------
 * Copyright (C) 2024, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef> // byte, size_t
#include <filesystem>
#include <fmt/format.h>
#include <gflags/gflags.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <onnxruntime/core/providers/etglow/etglow_provider_options.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <optional>
#include <string_view>
#include <thread>
#include <unistd.h>

#include "HelperFunctions.hpp"

DEFINE_bool(h, false, "help"); // if you wanna to use '-h'
DECLARE_bool(help);            // make sure you cna access FALGS_help
DECLARE_bool(helpshort);       // make sure you can access FLAGS_hlepshort

DEFINE_string(artifact_folder, "../../../DownloadArtifactory", "Absolute folder where the models and datasets are");
DEFINE_bool(verbose, false, "Verbose output");
DEFINE_uint64(batchSize, 1, "Number of images per batch");
DEFINE_string(image, "", "Image to be inferenced.");
DEFINE_bool(asyncMode, false, "Parallel inferences. By default use synchronous launch mode.");
DEFINE_int64(totalInferences, 1000, "Number of inferences to do.");
DEFINE_bool(deleteAsyncTensors, false, "Remove output tensors at callback on asynchronous run.");
DEFINE_bool(
  iobinding, false,
  "Inputs to be copied to the device and for outputs to be pre-allocated on the device prior to calling Run().");

static bool ValidateBatchSize(const char* flagname, uint64_t value) {
  if (value >= 1 && value < 32768) {
    return true;
  }
  INFO("Invalid value for " << flagname << "with value " << value);
  return false;
}

DEFINE_validator(batchSize, &ValidateBatchSize);

constexpr int64_t TEN_MILI_SECONDS = 10000000;

// Flag that defines the output verbosity.
bool g_verbose = false;

enum class EP { CPU = 0, ETGLOW = 1 };

constexpr std::string_view modelOnnxName = "model.onnx";

static std::atomic_bool atomic_wait{false};
static std::atomic_int atomicTotalAsyncCalls(0);

static std::atomic_bool atomic_waitEt{false};
static std::atomic_int atomicTotalAsyncCallsEt(0);
static bool all_async_runs_successtul = true;

struct userDataHelper {
  int64_t position;
  std::vector<Ort::Value> outs;
  std::vector<Ort::Value>* cpuOuts;

  ~userDataHelper() {
    outs.clear();
    cpuOuts->clear();
  }
};

// This is the structure to interface with the RESNET model
// After instantiation, set the input_image data to be the 2246x224 pixel image of the number to recognize
// Then call Run() to fill in the results_ data with the probabilities of each
// result_ holds the index with highest probability (aka the number the model thinks is in the image)
class RESNET {
public:
  explicit RESNET(int64_t batchSize, EP deviceProvider, bool binding = false)
    : batch_(batchSize)
    , typeOfProvider_(deviceProvider)
    , binding_(binding) {
  }

  int createSession(std::filesystem::path modelPath) {
    try {
      if (typeOfProvider_ == EP::CPU) {
        // Model path is const wchar_t*
        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
      } else if (typeOfProvider_ == EP::ETGLOW) {
        // fill et_options if need
        std::string parameters =
          "N=" + std::to_string(batch_) + ";batch=" + std::to_string(batch_) + ";height=224;width=224";
        etOptions_.et_onnx_symbols = parameters.c_str();
        etOptions_.device_id = 0;
        etOptions_.et_greedy = 1;
        sessionOptions_.AppendExecutionProvider_EtGlow(etOptions_);
        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
      } else {
        return -1;
      }
    } catch (const Ort::Exception& oe) {
      INFO("ONNX exception caught: " << oe.what() << ". Code " << oe.GetOrtErrorCode());
      return -1;
    }

    // define names
    Ort::AllocatorWithDefaultOptions allocator;

    numInputNodes_ = session_.GetInputCount();
    numOutputNodes_ = session_.GetOutputCount();

    for (size_t i = 0; i < numInputNodes_; i++) {
      Ort::AllocatedStringPtr inputNodeName = session_.GetInputNameAllocated(i, allocator);
      inputNodeNameAllocatedStrings.push_back(std::move(inputNodeName));
      inputNodeNames_.push_back(inputNodeNameAllocatedStrings.back().get());
    }
    for (size_t i = 0; i < numOutputNodes_; i++) {
      Ort::AllocatedStringPtr outputNodeName = session_.GetOutputNameAllocated(i, allocator);
      outputNodeNameAllocatedStrings.push_back(std::move(outputNodeName));
      outputNodeNames_.push_back(outputNodeNameAllocatedStrings.back().get());
    }
    return 0;
  }

  auto run(std::vector<Ort::Value>& inputTensor) -> std::vector<Ort::Value> {
    // It calculates and allocates an outputTensor and left result on it
    auto outTensor = session_.Run(runOptions_, inputNodeNames_.data(), inputTensor.data(), numInputNodes_,
                                  outputNodeNames_.data(), numOutputNodes_);
    return outTensor;
  }

  int run(std::vector<Ort::Value>& inputTensor, std::vector<Ort::Value>& outputTensor) {
    // Results are setting at provided outputTensor given from user
    try {
      session_.Run(runOptions_, inputNodeNames_.data(), inputTensor.data(), numInputNodes_, outputNodeNames_.data(),
                   outputTensor.data(), numOutputNodes_);
    } catch (const Ort::Exception& oe) {
      INFO("ONNX runBindig exception caught: " << oe.what() << ". Code " << oe.GetOrtErrorCode());
      return -1;
    }
    return 0;
  }

  int runasync(const Ort::Value* inputTensorPoolValue, Ort::Value* outputTensorPoolValue, RunAsyncCallbackFn callback,
               void* userdata) {
    try {
      session_.RunAsync(runOptions_, inputNodeNames_.data(), inputTensorPoolValue, numInputNodes_,
                        outputNodeNames_.data(), outputTensorPoolValue, numOutputNodes_, callback, userdata);

    } catch (const Ort::Exception& oe) {
      INFO("ONNX runAsync exception caught: " << oe.what() << ". Code " << oe.GetOrtErrorCode());
      return -1;
    }
    return 0;
  }

  int runBinding(void) {
    try {
      session_.Run(runOptions_, iobinding_);
    } catch (const Ort::Exception& oe) {
      INFO("ONNX runBindig exception caught: " << oe.what() << ". Code " << oe.GetOrtErrorCode());
      return -1;
    }
    return 0;
  }

  void bindSession(void) {
    iobinding_ = Ort::IoBinding(session_);
  }
  void bindInput(const Ort::Value& inputTensor) {
    iobinding_.BindInput("input", inputTensor);
  }
  void bindOutput(Ort::MemoryInfo& outMemInfo) {
    iobinding_.BindOutput("output", outMemInfo);
  }
  void bindOutput(Ort::Value& outputTensor) {
    iobinding_.BindOutput("output", outputTensor);
  }
  auto bindGetOutputValues(void) -> std::vector<Ort::Value> {
    return iobinding_.GetOutputValues();
  }
  bool isBinding(void) {
    return binding_;
  }

  static constexpr const int64_t numChannels = 3;
  static constexpr const int64_t width = 224;
  static constexpr const int64_t height = 224;
  Ort::Session session_{nullptr};

private:
  int64_t batch_;
  EP typeOfProvider_;
  bool binding_;
  Ort::IoBinding iobinding_{nullptr};
  Ort::Env env_{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default"};
  Ort::SessionOptions sessionOptions_;
  Ort::RunOptions runOptions_;
  OrtEtGlowProviderOptions etOptions_;
  size_t numInputNodes_;
  size_t numOutputNodes_;
  std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
  std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
  std::vector<const char*> inputNodeNames_;
  std::vector<char*> outputNodeNames_;
};

void readRawImageFiles(std::vector<fs::path> imgFilesPath,
                       std::vector<std::pair<std::filesystem::path, std::vector<float>>>& fileRawPixels) {
  // Mean and stddev normalization values of input tensors.
  // These are standard normalization factors for imagenet, adjusted for
  // normalizing values in the 0to255 range instead of 0to1, as seen at:
  // https://github.com/pytorch/examples/blob/master/imagenet/main.py
  const std::vector<float> mean{0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
  const std::vector<float> stddev{0.229f, 0.224f, 0.225f};
  constexpr auto scale = float(1.0 / 255);
  std::vector<float> vec;

  for (const auto& fname : imgFilesPath) {
    auto imgRaw = loadImage(fname);
    const auto* pPixel = imgRaw.data();
    for (size_t i = 0; i < (imgRaw.height() * imgRaw.rowBytes()); i = i + imgRaw.bytesPerPixel()) {
      // BGR
      // 0..255 to 0 to 1
      // adjust to training set mean and stddev.
      vec.emplace_back(((float(pPixel[i]) - mean[0]) / stddev[0]) * scale);
      vec.emplace_back(((float(pPixel[i + 1]) - mean[1]) / stddev[1]) * scale);
      vec.emplace_back(((float(pPixel[i + 2]) - mean[2]) / stddev[2]) * scale);
    }

    // Transpose (Height, Width, Chanel)(224,224,3) to (Chanel, height, Width)(3,224,224)
    std::vector<float> outputT;
    for (size_t ch = 0; ch < 3; ++ch) {
      for (size_t i = ch; i < vec.size(); i += 3) {
        outputT.emplace_back(vec[i]);
      }
    }
    fileRawPixels.emplace_back(fname.filename(), outputT);

    vec.clear();
    outputT.clear();
  }
}

void fillInputTensorWithDataImages(const size_t batchSize, std::string imgName,
                                   std::vector<std::pair<std::filesystem::path, std::vector<float>>>& fileRawPixels,
                                   std::vector<float>& inputTensorValues) {

  auto rasterInputTensor = inputTensorValues.data();
  // size is the same for all images, getting the first one.
  auto sizeRaster = fileRawPixels[0].second.size();

  if (imgName != "") {
    int64_t imgFoundAt = -1;
    for (size_t i = 0; i < fileRawPixels.size(); i++) {
      if (fileRawPixels[i].first.generic_string().find(imgName) != std::string::npos) {
        imgFoundAt = i;
        break;
      }
    }
    if (imgFoundAt != -1) {
      auto raster = fileRawPixels[imgFoundAt].second.data();
      for (size_t i = 0; i < batchSize; i++) {
        memcpy(rasterInputTensor + (i * sizeRaster), raster, sizeRaster * sizeof(float));
      }
    }
  } else {
    for (size_t i = 0; i < batchSize; i++) {
      auto raster = fileRawPixels[(i % fileRawPixels.size())].second.data();
      memcpy(rasterInputTensor + (i * sizeRaster), raster, sizeRaster * sizeof(float));
    }
  }
}

void sortResults(size_t batchSize, const Ort::Value& outValue,
                 std::vector<std::vector<std::pair<size_t, float>>>& batchlabelVal) {

  Ort::TensorTypeAndShapeInfo ts = outValue.GetTensorTypeAndShapeInfo();
  const float* outval = outValue.GetTensorData<float>();
  for (size_t i = 0; i < batchSize; i++) {
    // outs elements are returned in flattered array.
    for (size_t j = 0; j < (ts.GetElementCount() / batchSize); j++) {
      batchlabelVal[i][j] = std::pair<size_t, float>(j, outval[j + (i * 1000)]);
    }
    std::sort(batchlabelVal[i].begin(), batchlabelVal[i].end(),
              [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
  }
}

void showResults(size_t batchSize, std::string imgName,
                 std::vector<std::pair<std::filesystem::path, std::vector<float>>>& fileRawPixels,
                 std::vector<std::string>& labels, std::vector<Ort::Value>& outputTensor) {

  std::vector<std::vector<std::pair<size_t, float>>> batchlabelVal(batchSize,
                                                                   std::vector<std::pair<size_t, float>>(1000));

  sortResults(batchSize, *outputTensor.data(), batchlabelVal);

  INFO("Current BatchSize " << batchSize);
  if (imgName == "") {
    for (size_t i = 0; i < batchSize; i++) {
      INFO("Batch element " << i + 1 << " --> with image: " << fileRawPixels[i % fileRawPixels.size()].first);
      for (size_t j = 0; j < 5; j++) {
        const auto& [numlabel, probability] = batchlabelVal[i][j];
        INFO("" << j + 1 << " : " << labels[numlabel] << " " << probability);
      }
    }
  } else {
    int64_t imgFoundAt = -1;
    for (size_t i = 0; i < fileRawPixels.size(); i++) {
      if (fileRawPixels[i].first.generic_string().find(imgName) != std::string::npos) {
        imgFoundAt = i;
      }
    }
    if (imgFoundAt != -1) {
      for (size_t i = 0; i < batchSize; i++) {
        INFO("Batch element " << i + 1 << " --> with image: " << fileRawPixels[imgFoundAt].first);
        for (size_t j = 0; j < 5; j++) {
          const auto& [numlabel, probability] = batchlabelVal[i][j];
          INFO("" << j + 1 << " : " << labels[numlabel] << " " << probability);
        }
      }
    }
  }
}

int compareResults(size_t batchSize, std::vector<std::string>& labels, std::vector<Ort::Value>& outputTensor,
                   std::vector<Ort::Value>& outputTensorEt) {

  std::vector<std::vector<std::pair<size_t, float>>> batchlabelVal(batchSize,
                                                                   std::vector<std::pair<size_t, float>>(1000));
  std::vector<std::vector<std::pair<size_t, float>>> batchlabelValEt(batchSize,
                                                                     std::vector<std::pair<size_t, float>>(1000));

  sortResults(batchSize, *outputTensor.data(), batchlabelVal);
  sortResults(batchSize, *outputTensorEt.data(), batchlabelValEt);

  INFO("");
  INFO("*********************************************************************");
  INFO("**************** Compare results between providers ******************");
  INFO("*********************************************************************");
  INFO("********** Cpu and etGlowProvider with batch " << batchSize << " have: **********");
  INFO("*********************************************************************");
  // Check the 5 first entries labels are the same for cpu and etglowprovider.
  for (size_t i = 0; i < batchSize; i++) {
    INFO("Batch element " << i + 1);
    for (size_t j = 0; j < 5; j++) {
      const auto& [numlabel, probability] = batchlabelVal[i][j];
      const auto& [numlabelEt, probabilityEt] = batchlabelValEt[i][j];
      if (labels[numlabel] != labels[numlabelEt]) {
        INFO("Label at position " << i << " are NOT MATCHING ");
        INFO("Cpu result is " << labels[numlabel] << " whereas EtGlowProvider is " << labels[probabilityEt]);
        return -1;
      } else {
        INFO("" << j << ": " << labels[numlabel]);
      }
    }
  }
  INFO("*********************************************************************");
  INFO("");
  return 0;
}

int compareAsyncResults(int64_t ndx, const Ort::Value* cpuValue, const Ort::Value* etValue) {

  std::vector<std::vector<std::pair<size_t, float>>> cpuLabel(1, std::vector<std::pair<size_t, float>>(1000));
  std::vector<std::vector<std::pair<size_t, float>>> etLabel(1, std::vector<std::pair<size_t, float>>(1000));

  sortResults(1, *cpuValue, cpuLabel);
  sortResults(1, *etValue, etLabel);

  bool success = true;
  if (cpuLabel[0][0].first == etLabel[0][0].first) {
    INFO("Inferences " << ndx << " Matches between CPU and EtProvider in async run.");
  } else {
    INFO("*********************************************************************");
    INFO("Inference " << ndx << " NOT MATCHES between CPU and EtProvider in async run.");
    INFO("CPU " << cpuLabel[0][0].first << " ET prov " << etLabel[0][0].first);
    INFO("*********************************************************************");
    success = false;
  }

  cpuLabel.clear();
  etLabel.clear();
  return success;
}

void AsyncCallback(void* userData, [[maybe_unused]] OrtValue** outputs, [[maybe_unused]] size_t numOutputs,
                   OrtStatusPtr statusPtr) {

  Ort::Status status(statusPtr);
  int64_t iterValue = *(reinterpret_cast<int64_t*>(userData));

  VERBOSE("AsyncCallback -> " << iterValue);

  if (status.IsOK()) {
    // do something not necessary in this callback
  }

  atomicTotalAsyncCalls++;
  atomic_wait.store(true);
}

void AsyncCallbackEt(void* userData, [[maybe_unused]] OrtValue** outputs, [[maybe_unused]] size_t numOutputs,
                     OrtStatusPtr statusPtr) {

  Ort::Status status(statusPtr);
  std::unique_ptr<userDataHelper> userDataResources(reinterpret_cast<userDataHelper*>(userData));

  bool results_match = true;
  if (status.IsOK()) {
    results_match = compareAsyncResults(userDataResources->position, userDataResources->cpuOuts->data(),
                                        userDataResources->outs.data());
  }
  if (!status.IsOK() || !results_match) {
    all_async_runs_successtul = false;
  }

  atomicTotalAsyncCallsEt++;
  atomic_waitEt.store(true);
}

int main(int argc, char** argv) {

  gflags::SetUsageMessage("Usage: example-function [OPTION] ...");
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  if (FLAGS_help || FLAGS_h) { // if your help or h is true, make FLAGS_help off and FLAGS_helpshort on
    FLAGS_help = false;
    FLAGS_helpshort = true;
  }

  gflags::HandleCommandLineHelpFlags(); // add you handler back

  // Update verbosity
  g_verbose = FLAGS_verbose;

  VERBOSE("Artifacts folder placed at: " << FLAGS_artifact_folder);

  // Currently these resnet50 models fails see https://esperantotech.atlassian.net/browse/SW-20984
  // auto modelPath = FLAGS_artifact_folder / fs::path("models/resnet50_onnx/") / fs::path(modelOnnxName);
  // auto modelPath = FLAGS_artifact_folder / fs::path("models/resnet50-v2-7/") / fs::path(modelOnnxName);

  auto modelPath = FLAGS_artifact_folder / fs::path("models/resnet50_denso_onnx/") / fs::path(modelOnnxName);
  auto imagesPath = FLAGS_artifact_folder / fs::path("input_data/images/imagenet/images");
  auto labelFile = FLAGS_artifact_folder / fs::path("input_data/images/imagenet") / fs::path("labels.txt");

  // check providers
  auto providers = Ort::GetAvailableProviders();
  INFO("Available ORT providers: ");
  for (auto provider : providers) {
    INFO("" << provider);
  }

  // load labels
  std::vector<std::string> labels = loadLabels(labelFile);
  if (labels.empty()) {
    INFO("FAILED to load labels from: " << labelFile);
    return -1;
  }

  // Get a list of files images names and raw data
  const auto& imgFilesPath = directoryFiles(imagesPath, "*.png");
  std::vector<std::pair<std::filesystem::path, std::vector<float>>> fileRawPixels;
  readRawImageFiles(imgFilesPath, fileRawPixels);

  VERBOSE("Readed images file names");

  for (const auto& fraw : fileRawPixels) {
    VERBOSE("FileName: " << fraw.first);
  }

  size_t batchSize = 1;
  if (!FLAGS_asyncMode) {
    batchSize = FLAGS_batchSize;
  }

  std::string imgName;
  if (!gflags::GetCommandLineFlagInfoOrDie("image").is_default) {
    VERBOSE("Image provider is " << imgName);
    imgName = FLAGS_image;
    if (!checkImgExists(imgName, imgFilesPath)) {
      INFO("Image provided is not found at folder " << imagesPath);
      INFO("Execution TERMINATED");
      return -1;
    }
  }
  struct timespec rem, req = {0, TEN_MILI_SECONDS};

  RESNET resnet(batchSize, EP::CPU);
  RESNET resnetEt(batchSize, EP::ETGLOW, FLAGS_iobinding);

  resnet.createSession(modelPath);
  resnetEt.createSession(modelPath);

  std::vector<std::vector<int64_t>> inputDims;
  std::vector<int64_t> inputNodeDims = resnet.session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

  inputNodeDims[0] = batchSize;           // N
  inputNodeDims[1] = RESNET::numChannels; // num channels
  inputNodeDims[2] = RESNET::width;
  inputNodeDims[3] = RESNET::height;

  inputDims.push_back(inputNodeDims);

  size_t inputNodeSize = inputNodeDims[0] * inputNodeDims[1] * inputNodeDims[2] * inputNodeDims[3];

  int result = 0;
  if (FLAGS_asyncMode) {

    VERBOSE("RESNET Async execution start.");
    size_t totalInferences = FLAGS_totalInferences;
    VERBOSE("Total inferences to do " << totalInferences << ". BatchSize is fixed to " << batchSize);
    std::vector<std::vector<Ort::Value>> inputTensorPool(3);
    std::vector<std::vector<Ort::Value>> outputTensorPool(totalInferences);
    std::vector<std::vector<Ort::Value>> outputTensorPoolEt(totalInferences);

    std::vector<std::vector<float>> inputTensorValuesPool(3, std::vector<float>(inputNodeSize));
    std::vector<std::vector<float>> outputTensorValuesPool(totalInferences, std::vector<float>(1000));
    std::vector<std::vector<float>> outputTensorValuesPoolEt(totalInferences, std::vector<float>(1000));

    fillInputTensorWithDataImages(1, "1_cat", fileRawPixels, inputTensorValuesPool[0]);
    fillInputTensorWithDataImages(1, "2_dog", fileRawPixels, inputTensorValuesPool[1]);
    fillInputTensorWithDataImages(1, "3_zeb", fileRawPixels, inputTensorValuesPool[2]);

    std::array<int64_t, 2> outputShape = {1, 1000};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    for (size_t i = 0; i < inputTensorValuesPool.size(); i++) {
      inputTensorPool[i].emplace_back(Ort::Value::CreateTensor<float>(memory_info, inputTensorValuesPool[i].data(),
                                                                      inputTensorValuesPool[i].size(),
                                                                      inputNodeDims.data(), inputNodeDims.size()));
    }

    for (size_t i = 0; i < outputTensorValuesPool.size(); i++) {
      outputTensorPool[i].emplace_back(Ort::Value::CreateTensor<float>(memory_info, outputTensorValuesPool[i].data(),
                                                                       outputTensorValuesPool[i].size(),
                                                                       outputShape.data(), outputShape.size()));
    }

    VERBOSE("Calling runasync");

    std::vector<int64_t> position(totalInferences);
    std::iota(position.begin(), position.end(), 0);

    for (size_t i = 0; i < totalInferences; i++) {
      auto ndx = (i % inputTensorPool.size());
      resnet.runasync(inputTensorPool[ndx].data(), outputTensorPool[i].data(), AsyncCallback, &position[i]);
    }

    while (atomicTotalAsyncCalls != static_cast<int64_t>(totalInferences)) {
      nanosleep(&req, &rem);
    }
    atomicTotalAsyncCalls = 0;

    auto start = std::chrono::high_resolution_clock::now();
    if (FLAGS_deleteAsyncTensors) {
      for (size_t i = 0; i < totalInferences; i++) {
        std::unique_ptr<userDataHelper> userDataResource = std::make_unique<userDataHelper>();
        userDataResource->position = position[i];
        userDataResource->outs.emplace_back(Ort::Value{nullptr});
        userDataResource->cpuOuts = &outputTensorPool[i];
        auto ndx = (i % inputTensorPool.size());
        // If output_value are preallocated OrtValue* for each inference on the fly, It doesn't work fine.
        // If output_value is initialize by a null pointer it will be filled with an OrtValue* by onnxruntime.
        resnetEt.runasync(inputTensorPool[ndx].data(), userDataResource->outs.data(), AsyncCallbackEt,
                          userDataResource.get());
        userDataResource.release();
      }
    } else {
      for (size_t i = 0; i < outputTensorValuesPoolEt.size(); i++) {
        outputTensorPoolEt[i].emplace_back(
          Ort::Value::CreateTensor<float>(memory_info, outputTensorValuesPoolEt[i].data(),
                                          outputTensorValuesPoolEt[i].size(), outputShape.data(), outputShape.size()));
      }
      for (size_t i = 0; i < totalInferences; i++) {
        auto ndx = (i % inputTensorPool.size());
        resnetEt.runasync(inputTensorPool[ndx].data(), outputTensorPoolEt[i].data(), AsyncCallback, &position[i]);
      }
    }

    while ((FLAGS_deleteAsyncTensors ? atomicTotalAsyncCallsEt : atomicTotalAsyncCalls) !=
           static_cast<int64_t>(totalInferences)) {
      nanosleep(&req, &rem);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto elaps = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    INFO("**************************************************************");
    INFO("EtGlowProvider execution in ASYNC_MODE (launch+wait) " << totalInferences << " takes: " << elaps.count());
    INFO("**************************************************************");
    VERBOSE("Total asynccallback calls were :" << atomicTotalAsyncCalls);

    if (!FLAGS_deleteAsyncTensors) {
      for (size_t i = 0; i < outputTensorPool.size(); i++) {
        compareAsyncResults(i, outputTensorPool[i].data(), outputTensorPoolEt[i].data());
      }
    }
    if (!all_async_runs_successtul) {
      result = -1;
    }
  } else {

    std::vector<size_t> inputSizes;
    inputSizes.push_back(inputNodeSize);

    std::vector<float> inputTensorValues(inputNodeSize);

    fillInputTensorWithDataImages(batchSize, imgName, fileRawPixels, inputTensorValues);

    std::vector<Ort::Value> inputTensor;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, inputTensorValues.data(), inputTensorValues.size(), inputNodeDims.data(), inputNodeDims.size()));

    auto start = std::chrono::high_resolution_clock::now();
    auto outputTensor = resnet.run(inputTensor);
    assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());
    auto stop = std::chrono::high_resolution_clock::now();
    auto elaps = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    long double count = elaps.count(); // precision: long double
    INFO("**************************************************");
    INFO("CPU execution inference time: " << count);
    INFO("**************************************************");

    // Here It creates explicitly the expected output tensor and call resnet run.
    // std::vector<float> results;
    // results.resize(FLAGS_batchSize*1000);
    // std::array<int64_t, 2> outputShape = {FLAGS_batchSize, 1000};
    // std::array<int64_t, 2> outputShape = {1, FLAGS_batchSize*1000};
    // std::vector<Ort::Value> outputTensor;
    // outputTensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info,
    //                                                           results.data(),
    //                                                           results.size(),
    //                                                           outputShape.data(),
    //                                                           outputShape.size()));
    // resnet.run(inputTensor, outputTensor);

    showResults(batchSize, imgName, fileRawPixels, labels, outputTensor);

    std::vector<Ort::Value> outputTensorEt;
    start = std::chrono::high_resolution_clock::now();
    if (resnetEt.isBinding()) {
      resnetEt.bindInput(*inputTensor.data());
      Ort::MemoryInfo output_mem_info("ProviderEt", OrtDeviceAllocator, 0, OrtMemTypeDefault);
      resnetEt.bindOutput(output_mem_info);

      // The output tensor is not allocated before binding it, rather an Ort::MemoryInfo is bound as output.
      // This is an effective way to let the session allocate the tensor depending on the needed shapes.
      // Especially for data dependent shapes or dynamic shapes this can be a great solution
      // to get the right allocation. However in case the output shape is known and
      // the output tensor should be reused it is beneficial to bind an Ort::Value to the output as well.
      // This can be allocated using the session allocator or external memory.
      /*
      Ort::Allocator providerEt_allocator(resnetEt.session_, output_mem_info);
      auto outTensor = Ort::Value::CreateTensor<float>(providerEt_allocator,
                   outputShape.data(),
                   outputShape.size(),
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      resnetEt.bindOutput(outTensor);
      */
      resnetEt.runBinding();
      outputTensorEt = resnetEt.bindGetOutputValues();
    } else {
      outputTensorEt = resnetEt.run(inputTensor);
    }
    stop = std::chrono::high_resolution_clock::now();
    elaps = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    count = elaps.count(); // precision: long double
    INFO("**************************************************");
    INFO("EtGlowProvider execution inference time: " << count);
    INFO("**************************************************");
    showResults(batchSize, imgName, fileRawPixels, labels, outputTensorEt);

    result = compareResults(batchSize, labels, outputTensor, outputTensorEt);
  }

  VERBOSE("RESNET execution ends.");

  return result;
}
