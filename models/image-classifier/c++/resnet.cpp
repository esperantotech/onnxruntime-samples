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
#include <chrono>
#include <cmath>
#include <cstddef> // byte, size_t
#include <filesystem>
#include <fmt/format.h>
#include <gflags/gflags.h>
#include <iostream>
#include <iterator>
#include <optional>
#include <string_view>

#include <onnxruntime/core/providers/etglow/etglow_provider_options.h>
#include <onnxruntime/onnxruntime_cxx_api.h>

#include "HelperFunctions.hpp"

DEFINE_bool(h, false, "help"); // if you wanna to use '-h'
DECLARE_bool(help);            // make sure you cna access FALGS_help
DECLARE_bool(helpshort);       // make sure you can access FLAGS_hlepshort

DEFINE_string(artifact_folder, "../../../DownloadArtifactory", "Absolute folder where the models and datasets are");
DEFINE_bool(verbose, false, "Verbose output");
DEFINE_uint64(batchSize, 1, "Number of images per batch");

static bool ValidateBatchSize(const char* flagname, uint64_t value) {
  if (value >= 1 && value < 32768) {
    return true;
  }
  INFO("Invalid value for " << flagname << "with value " << value);
  return false;
}

DEFINE_validator(batchSize, &ValidateBatchSize);

// Flag that defines the output verbosity.
bool g_verbose = false;

enum class EP { CPU = 0, ETGLOW = 1 };

constexpr std::string_view modelOnnxName = "model.onnx";

// This is the structure to interface with the RESNET model
// After instantiation, set the input_image data to be the 2246x224 pixel image of the number to recognize
// Then call Run() to fill in the results_ data with the probabilities of each
// result_ holds the index with highest probability (aka the number the model thinks is in the image)
class RESNET {
public:
  explicit RESNET(EP deviceProvider)
    : typeOfProvider_(deviceProvider) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    inputOrtTensor_ =
      Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape_.data(), inputShape_.size());
    outputOrtTensor_ = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape_.data(),
                                                       outputShape_.size());
  }

  int createSession(std::filesystem::path modelPath) {
    try {
      if (typeOfProvider_ == EP::CPU) {
        // Model path is const wchar_t*
        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
      } else if (typeOfProvider_ == EP::ETGLOW) {
        // fill et_options if need
        // WatchOut! each resnet50 model has specific onxx_symbols.
        etOptions_.et_onnx_symbols = "N=1;batch=1;height=224;width=224";
        etOptions_.device_id = 0;
        sessionOptions_.AppendExecutionProvider_EtGlow(etOptions_);
        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
      } else {
        return -1;
      }
    } catch (Ort::Exception& oe) {
      std::cout << fmt::format("ONNX exception caught: {0} . Code {1}\n", oe.what(), oe.GetOrtErrorCode());
      return -1;
    }

    // define names
    Ort::AllocatorWithDefaultOptions allocator;
    inputName_.emplace(session_.GetInputNameAllocated(session_.GetInputCount() - 1, allocator));
    outputName_.emplace(session_.GetOutputNameAllocated(session_.GetOutputCount() - 1, allocator));

    return 0;
  }

  int run() {

    const char* inputNames[] = {inputName_->get()};
    char* outputNames[] = {outputName_->get()};

    Ort::RunOptions runOptions;

    try {
      session_.Run(runOptions, inputNames, &inputOrtTensor_, 1, outputNames, &outputOrtTensor_, 1);
    } catch (const Ort::Exception& oe) {
      std::cout << fmt::format("ONNX exception caught: {0} . Code {1}\n", oe.what(), oe.GetOrtErrorCode());
      return -1;
    }
    return 0;
  }

  void copyImageData(const RawImage& imgRaw) {
    // copy and conditioning image data to input array

    // Mean and stddev normalization values of input tensors.
    // These are standard normalization factors for imagenet, adjusted for
    // normalizing values in the 0to255 range instead of 0to1, as seen at:
    // https://github.com/pytorch/examples/blob/master/imagenet/main.py
    const std::vector<float> mean{0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    const std::vector<float> stddev{0.229f, 0.224f, 0.225f};
    const auto* pPixel = imgRaw.data();
    std::vector<float> vec;
    constexpr auto scale = float(1.0 / 255);

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
    std::copy(outputT.begin(), outputT.end(), input.begin());
  }

  void sortResult() {
    // sort results
    for (size_t i = 0; i < results.size(); ++i) {
      indexValuePairs_.emplace_back(i, results[i]);
    }
    std::sort(indexValuePairs_.begin(), indexValuePairs_.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
  }

  void cleanResults() {
    std::fill(std::begin(results), std::end(results), 0);
    std::fill(std::begin(input), std::end(input), 0);
    indexValuePairs_.clear();
  }

  void showResult(std::vector<std::string>& labels) {
    sortResult();
    // show result. top 5
    for (size_t i = 0; i < 5; ++i) {
      const auto& result = indexValuePairs_[i];
      INFO("" << i + 1 << ": " << labels[result.first] << " " << result.second);
    }
  }

  auto getIndexValuePairs() -> std::pair<size_t, float>* {
    return indexValuePairs_.data();
  }

  static constexpr const int64_t numChannels = 3;
  static constexpr const int64_t width = 224;
  static constexpr const int64_t height = 224;
  static constexpr const size_t numInputElements = numChannels * height * width;
  static constexpr const int64_t numClasses = 1000;
  // define array
  std::array<float, numInputElements> input;
  std::array<float, numClasses> results;

private:
  EP typeOfProvider_;
  Ort::Env env_{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default"};
  Ort::SessionOptions sessionOptions_;
  Ort::Session session_{nullptr};
  OrtEtGlowProviderOptions etOptions_;
  Ort::Value inputOrtTensor_{nullptr};
  Ort::Value outputOrtTensor_{nullptr};
  std::optional<Ort::AllocatedStringPtr> inputName_;
  std::optional<Ort::AllocatedStringPtr> outputName_;
  // define shape
  const std::array<int64_t, 4> inputShape_ = {1, numChannels, height, width};
  const std::array<int64_t, 2> outputShape_ = {1, numClasses};
  std::vector<std::pair<size_t, float>> indexValuePairs_;
};

int compareProvidersResults(std::vector<std::string> labels, const std::pair<size_t, float>* cpuResult,
                            const std::pair<size_t, float>* etResult) {

  INFO("");
  INFO("*********************************************");
  INFO("***** Compare results between providers *****");
  INFO("*********************************************");
  INFO("Cpu and etGlowProvider have same results:    ");
  // Check the 5 first entries labels are the same for cpu and etglowprovider.
  for (size_t i = 0; i < 5; ++i, ++cpuResult, ++etResult) {
    if (labels[cpuResult->first] != labels[etResult->first]) {
      INFO("Label at position " << i << " are NOT MATCHING ");
      INFO("Cpu result is " << labels[cpuResult->first] << " whereas EtGlowProvider is " << labels[etResult->first]);
      return -1;
    } else {
      INFO("" << i << ": " << labels[cpuResult->first]);
    }
  }
  INFO("*********************************************");
  INFO("*********************************************");
  INFO("");
  return 0;
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
  for (auto provider : providers) {
    INFO("" << provider);
  }

  // load labels
  std::vector<std::string> labels = loadLabels(labelFile);
  if (labels.empty()) {
    INFO("FAILED to load labels: " << labelFile);
    return -1;
  }

  VERBOSE("Working model: RESNET50");

  RESNET resnet(EP::CPU);
  RESNET resnetEt(EP::ETGLOW);

  resnet.createSession(modelPath);
  resnetEt.createSession(modelPath);

  // Get a list of images files to process
  const auto& imgFilesPath = directoryFiles(imagesPath, "*.png");

  for (const auto& filename : imgFilesPath) {
    VERBOSE("Processing image file: " << filename.filename());
    auto imgRaw = loadImage(filename);

    resnet.copyImageData(imgRaw);
    resnetEt.copyImageData(imgRaw);

    auto start = std::chrono::high_resolution_clock::now();
    resnet.run();
    auto stop = std::chrono::high_resolution_clock::now();
    auto elaps = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    long double count = elaps.count(); // precision: long double
    INFO("CPU execution inference time: " << count);
    resnet.showResult(labels);

    start = std::chrono::high_resolution_clock::now();
    resnetEt.run();
    stop = std::chrono::high_resolution_clock::now();
    elaps = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    count = elaps.count(); // precision: long double
    INFO("EtGlowProvider execution inference time: " << count);
    resnetEt.showResult(labels);

    compareProvidersResults(labels, resnet.getIndexValuePairs(), resnetEt.getIndexValuePairs());

    resnet.cleanResults();
    resnetEt.cleanResults();
  }

  VERBOSE("RESNET execution ends.");

  return 0;
}