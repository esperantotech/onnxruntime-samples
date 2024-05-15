
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/etglow/etglow_provider_options.h>

// POSIX
#include <fnmatch.h>

//STD
#include <array>
#include <cmath>
#include <algorithm>
#include <cstdint> 
#include <iostream>
#include <string>
#include <filesystem>
#include <functional> // invoke()
#include <fmt/format.h>
#include <string_view>
#include <variant>
#include <map>
#include <iterator>
#include <iostream>
#include <chrono>


#include "CommandLine.hpp"
#include "Config.h"
#include "ErrorMng.hpp"
#include "PreprocessImageClassifier.hpp"
#include "Utils.hpp"
#include "nlohmann/json.hpp"
#include "RawImage.hpp"
#include "Types.hpp"

// Some namespace and type names simplifications
namespace fs = std::filesystem;
using json = nlohmann::json;

#define INFO(args_) \
  do { \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) */ \
    std::cout << timestamp() << args_ << std::endl; \
  } while (false)

#define VERBOSE(args_) \
  if (g_verbose) { \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) */ \
    std::cout << timestamp() << args_ << std::endl; \
  } else \
    ((void)0)

// Flag that defines the output verbosity.
bool g_verbose = false;

enum class EP {
    CPU    = 0,
    ETGLOW = 1
};

struct ExecParams final {
  std::filesystem::path artifactsFolder;  
  fs::path modelInfoFilename;  
  nlohmann::json modelInfo;
  bool verbose = false;
  size_t batchSize;  
};

auto loadJSONFile(const std::string &filename) -> nlohmann::json {
  std::ifstream f(filename);
  if (not f) {
    exitWithErrorMsg("Cannot open file '" + filename + "'");
  }
  try {
    return json::parse(f);
  } catch (const std::exception &ex) {
    exitWithErrorMsg("Bad JSON file '" + filename + "'\n" + ex.what());
  }
}

void printUsage(const CommandLineParseResult& cmd) {
  std::cout <<  fmt::format("ONNXRUNTIME SAMPLES {0}.{1}.{2}, Copyright (C) 2023 Esperanto Technologies, Inc.\n",
                             VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)                 
            << "\n"
            << fmt::format("Usage: {0} [OPTION]...\n",std::filesystem::path(cmd.name).filename().generic_string())
            << "Options:\n"
               "    -h, --help                          Print this help message\n"
               "\n"
               "    -a, --artifacts-folder              Absolute folder where the models and datasets are (e.g.: "
               "/home/et/SW-artifacts)\n"               
               "Mandatory options:\n"
               "    -mf, --model-info-file <filename>   JSON file with the model description (optional "
               "if FILE is provided)\n"
               "\n"
               "Optional options:\n"
               "    -b, --batch-size <N>                Number of images per batch (default: "      
               "    -v, --verbose                       Verbose output\n\n";
}

auto execParamsFromCommandLine(const CommandLineParseResult& cmd) -> ExecParams {
  // Helper function that exits the program with an error message when an option value is invalid.
  const auto invalidOptionValue = [&cmd](std::string_view optionName, std::string_view reason) {
    printUsage(cmd);
    exitWithErrorMsg(fmt::format("invalid value of option '{0}': {1}.", optionName, reason));
  };
  // Helper function that exits the program with an error message when an option value is missing.
  const auto checkOptionHasRequiredValue = [&invalidOptionValue](std::string_view optionName, std::string_view value) {
    if (value.empty()) {
      invalidOptionValue(optionName, "missing argument value");
    }
  };
  // Helper function that returns the bool value corresponding to the input param string-like
  // representation.
  const auto parseBoolValue = [](const auto &v) { return v != "false"; };
  // Helper function that returns if the given container contains the given value.
  const auto contains = [](const std::vector<const char *> &c, std::string_view v) {
    return c.end() != std::find_if(c.begin(), c.end(), [&](const char *e) {
      return std::string_view(e) == v;
    });
  };

  ExecParams result;
  //
  for (const auto& [key, val] : cmd.options) {
    if (contains({"-h", "--help"}, key)) {
      printUsage(cmd);
      exit(EXIT_SUCCESS);      
    }  else if (contains({"-v", "--verbose"}, key)) {
      result.verbose = parseBoolValue(val);
    } else if (contains({"-a", "--artifacts-folder"}, key)) {
      checkOptionHasRequiredValue(key, val);
      auto v = fs::path(val);
      if (not fs::exists(v)) {
        invalidOptionValue(key, "file not exists");        
      }
      result.artifactsFolder = std::move(v);
    } else if (contains({"-b", "--batch-size"}, key)) {    
      checkOptionHasRequiredValue(key, val);
      const auto v = parsePositiveValue(val);
      if (not v.has_value()) {
        invalidOptionValue(key, "value must be a positive number");
      }
      result.batchSize = v.value();
    } else if (contains({"-mf", "--model-info-file"}, key)) {
      checkOptionHasRequiredValue(key, val);
      auto v = fs::path(val);
      if (not fs::exists(v)) {
        invalidOptionValue(key, "file not exists");        
      }
      result.modelInfoFilename = std::move(v);
      result.modelInfo = loadJSONFile(result.modelInfoFilename);
    } else {
      const auto hint = "\nhint: use option -h or --help to list available options.";
      exitWithErrorMsg(fmt::format("unknown option '{0}'{1}", key, hint));      
    }
  } 

  if (result.artifactsFolder.empty()) {
    result.artifactsFolder = "../../../DownloadArtifactory";
  } 

  return result;
}

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
// After instantiation, set the input_image_ data to be the 28x28 pixel image of the number to recognize
// Then call Run() to fill in the results_ data with the probabilities of each
// result_ holds the index with highest probability (aka the number the model thinks is in the image)
class MNIST {
 public:

  MNIST(EP deviceProvider): typeOfProvider_(deviceProvider) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
                                                    input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                                     output_shape_.data(), output_shape_.size());
  }

  int createSession(std::filesystem::path modelPath) {

    try {
      if (typeOfProvider_ == EP::CPU) {
        // Model path is const wchar_t*
        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
      }
      else if (typeOfProvider_ == EP::ETGLOW) {
        //fill et_options if need
        sessionOptions_.AppendExecutionProvider_EtGlow(et_options_);
        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
      } else {
        return -1;
      }
    } catch (Ort::Exception &oe) {
      std::cout << fmt::format("ONNX exception caught: {0} . Code {1}\n", oe.what(), oe.GetOrtErrorCode());
      return -1;
    }
    return 0;
  }

  std::ptrdiff_t Run() {
    const char* input_names[] = {"Input3"};
    const char* output_names[] = {"Plus214_Output_0"};

    Ort::RunOptions run_options;
    try {
      session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    } catch (Ort::Exception &oe) {
      std::cout << fmt::format("ONNX excetion caught: {0} . Code {1}\n", oe.what(), oe.GetOrtErrorCode());
      return -1;
    }

    softmax(results_);
    result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
    return result_;
  }

  void getImage(RawImage imgRaw) {
    
    float* output = input_image_.data();
    std::fill(input_image_.begin(), input_image_.end(), 0.f);
    const auto *pPixel = imgRaw.data();

    for (size_t x=0; x < (imgRaw.height() * imgRaw.width()); x++) {
      output[x] += int(pPixel[x]) == 0 ? 0.0f : 0.1f;
    }
  } 

  static constexpr const int width_ = 28;
  static constexpr const int height_ = 28;

  std::array<float, width_ * height_> input_image_{};
  std::array<float, 10> results_{};
  int64_t result_{0};

 private:
  EP typeOfProvider_;
  Ort::Env env_{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default"};
  Ort::SessionOptions sessionOptions_;
  Ort::Session session_{nullptr};
  
  OrtEtGlowProviderOptions et_options_;

  Ort::Value input_tensor_{nullptr};
  std::array<int64_t, 4> input_shape_{1, 1, width_, height_};

  Ort::Value output_tensor_{nullptr};
  std::array<int64_t, 2> output_shape_{1, 10};  
};

auto directoryFiles(const std::filesystem::path &path, const char *pattern) {
  std::vector<std::string> result; //filesystem::path> result;
  for (const auto &dirEntry : fs::directory_iterator(path)) {
    const auto &p = dirEntry.path();
    if (not fs::is_regular_file(p)) {
      continue;
    }
    if (0 == fnmatch(pattern, p.filename().string().c_str(), FNM_PATHNAME | FNM_PERIOD)) {
      result.emplace_back(p);
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

int main(int argc, char *argv[]) {

  // Parse command line arguments
  const auto& cmdParams = parseCommandLine(argc, argv);
  auto execParams = execParamsFromCommandLine(cmdParams);

  // Update verbosity
  g_verbose = execParams.verbose;
  
  std::cout << fmt::format("artifacts folder placed on {0}\n", execParams.artifactsFolder.generic_string());

  auto modelPath = execParams.artifactsFolder / fs::path("models/mnist") / fs::path(execParams.modelInfo["model_file"].get<std::string>());
  VERBOSE("  ONNX model file: " << modelPath);
  auto imagesPath = execParams.artifactsFolder / fs::path("input_data/images/images_1-1-28-28");
  VERBOSE("  Images to be load from: " << imagesPath);
  
  //check providers
  auto providers = Ort::GetAvailableProviders();
  for (auto provider : providers) {
    std::cout << provider << std::endl;
  }

  //model to use
  VERBOSE("Working model: " << ((execParams.modelInfo.contains("name") > 0) ?
                execParams.modelInfo["name"].get<std::string>() :
                std::string("(unnamed)")));
  const std::string modelFormat = execParams.modelInfo["format"];
  
  MNIST mnist(EP::CPU);
  MNIST mnistEt(EP::ETGLOW);
  mnist.createSession(modelPath);
  mnistEt.createSession(modelPath);
  
  // Get a list of image files to process
  const auto &imgFilesPath = directoryFiles(imagesPath, "*.png");

  for (const auto &filename : imgFilesPath) {
    VERBOSE("processing image file: " << filename);
    auto img = ImageClassifierPreprocessor::loadImage(filename);
    mnist.getImage(img);
    mnistEt.getImage(img);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = mnist.Run();
    auto stop = std::chrono::high_resolution_clock::now();
    auto elaps = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    long double count = elaps.count(); // precision: long double.
    VERBOSE("CPU execution inference time: "<< count);

    start = std::chrono::high_resolution_clock::now();    
    auto result2 = mnistEt.Run();
    stop = std::chrono::high_resolution_clock::now();
    elaps = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    count = elaps.count(); // precision: long double.
    VERBOSE("EtGlowProvider execution inference time: "<< count);
    
    VERBOSE("CPU        result --> "<<result);
    VERBOSE("EtProvicer result --> "<<result2);
    if (result == result2) {
      INFO("CPU and EtGlowProvider matches the result: " << result << "  for image: " << filename);
    }
  }

  VERBOSE("End excution of model" << execParams.modelInfo.contains("name"));

  return EXIT_SUCCESS; 
}