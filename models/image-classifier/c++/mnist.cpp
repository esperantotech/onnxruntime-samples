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

// Flag that defines the output verbosity.
bool g_verbose = false;

enum class EP {
    CPU    = 0,
    ETGLOW = 1
};

constexpr std::string_view modelOnnxName = "model.onnx";

template <typename T>
static void softmax(T& input) {
  float rowmax = *std::max_element(input.begin(), input.end());
  std::vector<float> y(input.size());
  float sum = 0.0f;
  for (size_t i = 0; i != input.size(); ++i) {
    sum += y[i] = std::exp(input[i] - rowmax);
  }
  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = y[i] / sum;
  }
}

// This is the structure to interface with the MNIST model
// After instantiation, set the input data to be the 28x28 pixel image of the number to recognize
// Then call Run() to fill in the results_ data with the probabilities of each
// result_ holds the index with highest probability (aka the number the model thinks is in the image)
class MNIST {
 public:
   explicit MNIST(EP deviceProvider)
     : typeOfProvider_(deviceProvider) {
     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
     inputOrtTensor_ = Ort::Value::CreateTensor<float>(memory_info, input_.data(), input_.size(), inputShape_.data(),
                                                       inputShape_.size());
     outputOrtTensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                                        outputShape_.data(), outputShape_.size());
   }

  int createSession(std::filesystem::path modelPath) {

    try {
      if (typeOfProvider_ == EP::CPU) {
        // Model path is const wchar_t*
        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
      }
      else if (typeOfProvider_ == EP::ETGLOW) {
        //fill et_options if need
        sessionOptions_.AppendExecutionProvider_EtGlow(etOptions_);
        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
      } else {
        return -1;
      }
    } catch (Ort::Exception &oe) {
      std::cout << fmt::format("ONNX exception caught: {0} . Code {1}\n", oe.what(), oe.GetOrtErrorCode());
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

  int run() {

    try {
      session_.Run(runOptions_, inputNodeNames_.data(), &inputOrtTensor_, 1, outputNodeNames_.data(), &outputOrtTensor_,
                   1);
    } catch (Ort::Exception& oe) {
      std::cout << fmt::format("ONNX excetion caught: {0} . Code {1}\n", oe.what(), oe.GetOrtErrorCode());
      return -1;
    }
    return 0;
  }

  std::ptrdiff_t softMaxResult() {
    softmax(results_);
    result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
    return result_;
  }

  void showResult() {
    auto value = softMaxResult();
    if (typeOfProvider_ == EP::CPU) {
      INFO("CPU            result --> " << value);
    } else {
      INFO("EtGlowProvider result --> " << value);
    }
  }

  void copyImageData(const RawImage& imgRaw) {
    // copy and contitioning image data to input array

    const auto *pPixel = imgRaw.data();
    std::vector<float> vec;

    for (size_t i = 0; i < (imgRaw.height() * imgRaw.rowBytes()); i = i + imgRaw.bytesPerPixel()) {
      vec.emplace_back(int(pPixel[i]) == 0 ? 0.0f : 1.0f);
    }
    std::copy(vec.begin(), vec.end(), input_.begin());
  }

  auto getResult() const -> int64_t {
    return result_;
  }

  static constexpr const int width_ = 28;
  static constexpr const int height_ = 28;
  // define array
  std::array<float, width_ * height_> input_{};
  std::array<float, 10> results_{};
  int64_t result_{0};

 private:
  EP typeOfProvider_;
  Ort::Env env_{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default"};
  Ort::SessionOptions sessionOptions_;
  Ort::Session session_{nullptr};
  OrtEtGlowProviderOptions etOptions_;
  Ort::Value inputOrtTensor_{nullptr};
  Ort::Value outputOrtTensor_{nullptr};
  Ort::RunOptions runOptions_;
  size_t numInputNodes_;
  size_t numOutputNodes_;
  std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
  std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
  std::vector<const char*> inputNodeNames_;
  std::vector<char*> outputNodeNames_;

  // define shape
  std::array<int64_t, 4> inputShape_{1, 1, width_, height_};
  std::array<int64_t, 2> outputShape_{1, 10};
};

int compareProvidersResults(const std::filesystem::path& filename, int64_t cpuResult, int64_t etResult) {

  INFO("");
  INFO("*********************************************");
  INFO("***** Compare results between providers *****");
  INFO("*********************************************");
  INFO("Cpu and etGlowProvider have same results:    ");
  // Check the 5 first entries labels are the same for cpu and etglowprovider.
  if (cpuResult != etResult) {
    INFO("Result for " << filename << " are NOT MATCHING ");
    INFO("Cpu result is " << cpuResult << " whereas EtGlowProvider is " << etResult);
    return -1;
  } else {
    INFO("Image: " << filename);
    INFO("CPU and EtGlowProvider matches result is : " << cpuResult);
  }
  INFO("*********************************************");
  INFO("*********************************************");
  INFO("");
  return 0;
}

int main(int argc, char *argv[]) {

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

  auto modelPath = FLAGS_artifact_folder / fs::path("models/mnist") / fs::path(modelOnnxName);
  auto imagesPath = FLAGS_artifact_folder / fs::path("input_data/images/images_1-1-28-28");

  //check providers
  auto providers = Ort::GetAvailableProviders();
  INFO("Available ORT providers: ");
  for (auto provider : providers) {
    INFO("" << provider);
  }

  VERBOSE("Working model: MINST");

  MNIST mnist(EP::CPU);
  MNIST mnistEt(EP::ETGLOW);

  mnist.createSession(modelPath);
  mnistEt.createSession(modelPath);
  
  // Get a list of image files to process
  const auto &imgFilesPath = directoryFiles(imagesPath, "*.png");

  for (const auto &filename : imgFilesPath) {
    VERBOSE("processing image file: " << filename);

    auto imgRaw = loadImage(filename);
    mnist.copyImageData(imgRaw);
    mnistEt.copyImageData(imgRaw);

    auto start = std::chrono::high_resolution_clock::now();
    mnist.run();
    auto stop = std::chrono::high_resolution_clock::now();
    auto elaps = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    long double count = elaps.count(); // precision: long double.
    VERBOSE("CPU execution inference time: "<< count);
    mnist.showResult();

    start = std::chrono::high_resolution_clock::now();
    mnistEt.run();
    stop = std::chrono::high_resolution_clock::now();
    elaps = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    count = elaps.count(); // precision: long double.
    VERBOSE("EtGlowProvider execution inference time: "<< count);
    mnistEt.showResult();

    compareProvidersResults(filename.filename(), mnist.getResult(), mnistEt.getResult());
  }

  VERBOSE("MINST execution ends.");

  return 0;
}