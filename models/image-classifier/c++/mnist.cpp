

#include <iostream>
#include <array>
#include <cstdint> 
#include <onnxruntime/onnxruntime_cxx_api.h>

int main(void) {


  static constexpr const int width_ = 28;
  static constexpr const int height_ = 28;

  Ort::Env env;
  //Ort::Session session_{env, L"../mnist/mnist-12.onnx", Ort::SessionOptions{nullptr}};    

  Ort::Value input_tensor_{nullptr};
  std::array<int64_t, 4> input_shape_{1, 1, width_, height_};

  Ort::Value output_tensor_{nullptr};
  std::array<int64_t, 2> output_shape_{1, 10};

  std::cout << "end c++ api mnist" << std::endl;

  return 0; 
}