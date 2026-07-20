#include "visionops_runtime/rknn_runner.hpp"

#include <rknn_api.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace visionops::runtime {

namespace {

class RknnRunnerReal final : public RknnRunner {
 public:
  explicit RknnRunnerReal(std::string task_type) : task_type_(std::move(task_type)) {}
  ~RknnRunnerReal() override { release(); }

  bool load_model(const std::string& path, const RunnerModelConfig& config) override {
    release();
    task_type_ = config.task_type;
    input_width_ = config.input_width;
    input_height_ = config.input_height;
    dump_io_ = config.dump_io;

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
      last_error_ = "无法读取 RKNN 模型: " + path;
      return false;
    }
    const auto size = input.tellg();
    if (size <= 0) {
      last_error_ = "RKNN 模型为空: " + path;
      return false;
    }
    model_data_.resize(static_cast<std::size_t>(size));
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char*>(model_data_.data()), size)) {
      last_error_ = "RKNN 模型读取失败: " + path;
      model_data_.clear();
      return false;
    }

    const int init_result = rknn_init(
        &context_,
        model_data_.data(),
        static_cast<std::uint32_t>(model_data_.size()),
        0,
        nullptr);
    if (init_result < 0) {
      last_error_ = "rknn_init 失败，错误码 " + std::to_string(init_result);
      context_ = 0;
      model_data_.clear();
      return false;
    }

    rknn_input_output_num io_count{};
    const int query_result = rknn_query(
        context_, RKNN_QUERY_IN_OUT_NUM, &io_count, sizeof(io_count));
    if (query_result < 0 || io_count.n_input == 0) {
      last_error_ = "RKNN 输入输出查询失败，错误码 " + std::to_string(query_result);
      release();
      return false;
    }
    input_count_ = io_count.n_input;
    output_count_ = io_count.n_output;
    input_infos_.clear();
    input_infos_.reserve(input_count_);
    for (std::uint32_t index = 0; index < input_count_; ++index) {
      rknn_tensor_attr attribute{};
      attribute.index = index;
      const int attribute_result = rknn_query(
          context_, RKNN_QUERY_INPUT_ATTR, &attribute, sizeof(attribute));
      if (attribute_result < 0) {
        last_error_ = "RKNN 输入属性查询失败，索引 " + std::to_string(index) +
            "，错误码 " + std::to_string(attribute_result);
        release();
        return false;
      }
      TensorInfo info;
      info.name = attribute.name;
      info.data_type = std::to_string(static_cast<int>(attribute.type));
      info.layout = std::to_string(static_cast<int>(attribute.fmt));
      info.byte_size = attribute.size;
      for (std::uint32_t dimension = 0; dimension < attribute.n_dims; ++dimension) {
        info.dimensions.push_back(attribute.dims[dimension]);
      }
      input_infos_.push_back(std::move(info));
    }
    output_infos_.clear();
    output_attrs_.clear();
    output_infos_.reserve(output_count_);
    output_attrs_.reserve(output_count_);
    for (std::uint32_t index = 0; index < output_count_; ++index) {
      rknn_tensor_attr attribute{};
      attribute.index = index;
      const int attribute_result = rknn_query(
          context_, RKNN_QUERY_OUTPUT_ATTR, &attribute, sizeof(attribute));
      if (attribute_result < 0) {
        last_error_ = "RKNN 输出属性查询失败，索引 " + std::to_string(index) +
            "，错误码 " + std::to_string(attribute_result);
        release();
        return false;
      }
      TensorInfo info;
      info.name = attribute.name;
      info.data_type = std::to_string(static_cast<int>(attribute.type));
      info.layout = std::to_string(static_cast<int>(attribute.fmt));
      const std::size_t float_bytes = static_cast<std::size_t>(attribute.n_elems) * sizeof(float);
      info.byte_size = float_bytes > 0 ? float_bytes : attribute.size;
      for (std::uint32_t dimension = 0; dimension < attribute.n_dims; ++dimension) {
        info.dimensions.push_back(attribute.dims[dimension]);
      }
      output_infos_.push_back(std::move(info));
      output_attrs_.push_back(attribute);
    }
    output_float_buffers_.clear();
    reusable_outputs_.clear();
    output_float_buffers_.resize(output_count_);
    reusable_outputs_.resize(output_count_);
    for (std::uint32_t index = 0; index < output_count_; ++index) {
      std::size_t elements = static_cast<std::size_t>(output_attrs_[index].n_elems);
      if (elements == 0) {
        elements = 1;
        for (const auto dimension : output_infos_[index].dimensions) elements *= dimension;
      }
      output_float_buffers_[index].resize(elements);
      reusable_outputs_[index].index = index;
      reusable_outputs_[index].want_float = 1;
      reusable_outputs_[index].is_prealloc = 1;
      reusable_outputs_[index].buf = output_float_buffers_[index].data();
      reusable_outputs_[index].size = static_cast<std::uint32_t>(elements * sizeof(float));
    }
    if (dump_io_) {
      std::cout << "RKNN IO: inputs=" << input_count_ << " outputs=" << output_count_ << '\n';
      for (std::size_t index = 0; index < output_infos_.size(); ++index) {
        std::cout << "  output[" << index << "] " << output_infos_[index].name << " dims=";
        for (const auto dimension : output_infos_[index].dimensions) std::cout << dimension << 'x';
        std::cout << " bytes=" << output_infos_[index].byte_size << '\n';
      }
    }
    loaded_ = true;
    last_error_.clear();
    return true;
  }

  bool is_loaded() const override { return loaded_; }
  std::string backend_name() const override { return "rknn"; }

  RknnOutput infer(const RknnInput& input) override {
    std::lock_guard<std::mutex> lock(mutex_);
    RknnOutput result;
    result.runner_called = true;
    result.task_type = task_type_;
    if (!loaded_) {
      result.error = last_error_.empty() ? "RKNN 模型尚未加载" : last_error_;
      return result;
    }
    if (input.data == nullptr || input.size == 0) {
      result.error = "RKNN 输入数据为空";
      last_error_ = result.error;
      return result;
    }

    rknn_input rknn_input_value{};
    rknn_input_value.index = 0;
    rknn_input_value.buf = const_cast<std::uint8_t*>(input.data);
    rknn_input_value.size = static_cast<std::uint32_t>(input.size);
    rknn_input_value.pass_through = 0;
    rknn_input_value.type = RKNN_TENSOR_UINT8;
    rknn_input_value.fmt = RKNN_TENSOR_NHWC;

    const auto set_input_started = std::chrono::steady_clock::now();
    int code = rknn_inputs_set(context_, 1, &rknn_input_value);
    result.set_input_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - set_input_started).count();
    if (code < 0) {
      result.error = "rknn_inputs_set 失败，错误码 " + std::to_string(code);
      last_error_ = result.error;
      return result;
    }
    const auto run_started = std::chrono::steady_clock::now();
    code = rknn_run(context_, nullptr);
    result.run_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - run_started).count();
    if (code < 0) {
      result.error = "rknn_run 失败，错误码 " + std::to_string(code);
      last_error_ = result.error;
      return result;
    }

    bool used_preallocated_outputs = output_preallocation_supported_;
    std::vector<rknn_output> dynamic_outputs;
    const auto get_output_started = std::chrono::steady_clock::now();
    if (used_preallocated_outputs) {
      for (std::uint32_t index = 0; index < output_count_; ++index) {
        reusable_outputs_[index].index = index;
        reusable_outputs_[index].want_float = 1;
        reusable_outputs_[index].is_prealloc = 1;
        reusable_outputs_[index].buf = output_float_buffers_[index].data();
        reusable_outputs_[index].size = static_cast<std::uint32_t>(
            output_float_buffers_[index].size() * sizeof(float));
      }
      code = rknn_outputs_get(context_, output_count_, reusable_outputs_.data(), nullptr);
      if (code < 0) {
        // Some older runtime/driver combinations reject preallocated float
        // outputs. Fall back to the legacy allocation path and keep the model
        // operational; subsequent frames skip the failed preallocation attempt.
        output_preallocation_supported_ = false;
        used_preallocated_outputs = false;
        std::cerr << "[WARN] RKNN preallocated float outputs are unsupported by the current "
                  << "runtime/driver; falling back to dynamic output buffers, code=" << code << '\n';
      }
    }
    if (!used_preallocated_outputs) {
      dynamic_outputs.resize(output_count_);
      for (std::uint32_t index = 0; index < output_count_; ++index) {
        dynamic_outputs[index].index = index;
        dynamic_outputs[index].want_float = 1;
        dynamic_outputs[index].is_prealloc = 0;
      }
      code = rknn_outputs_get(context_, output_count_, dynamic_outputs.data(), nullptr);
    }
    if (code < 0) {
      result.get_output_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - get_output_started).count();
      result.error = "rknn_outputs_get 失败，错误码 " + std::to_string(code);
      last_error_ = result.error;
      return result;
    }

    result.tensors.reserve(output_count_);
    result.host_input_copy_avoided = true;
    result.output_buffers_preallocated = used_preallocated_outputs;
    for (std::uint32_t index = 0; index < output_count_; ++index) {
      RuntimeTensor tensor;
      tensor.info = output_infos_[index];
      if (used_preallocated_outputs) {
        const std::size_t capacity = output_float_buffers_[index].size() * sizeof(float);
        std::size_t actual_size = reusable_outputs_[index].size;
        if (actual_size == 0 || actual_size > capacity) actual_size = capacity;
        tensor.info.byte_size = actual_size;
        tensor.set_data_view(output_float_buffers_[index].data(), actual_size);
        result.output_view_bytes += actual_size;
      } else {
        tensor.info.byte_size = dynamic_outputs[index].size;
        if (dynamic_outputs[index].buf != nullptr && dynamic_outputs[index].size > 0) {
          const auto* begin = static_cast<const std::uint8_t*>(dynamic_outputs[index].buf);
          tensor.data.assign(begin, begin + dynamic_outputs[index].size);
        }
      }
      result.tensors.push_back(std::move(tensor));
    }
    if (!used_preallocated_outputs) {
      rknn_outputs_release(context_, output_count_, dynamic_outputs.data());
    }
    result.get_output_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - get_output_started).count();

    result.success = true;
    result.inference_ms = result.set_input_ms + result.run_ms + result.get_output_ms;
    last_error_.clear();
    return result;
  }

  std::string last_error() const override { return last_error_; }
  std::uint32_t input_count() const override { return input_count_; }
  std::uint32_t output_count() const override { return output_count_; }
  std::vector<TensorInfo> input_infos() const override { return input_infos_; }
  std::vector<TensorInfo> output_infos() const override { return output_infos_; }

 private:
  void release() {
    if (context_ != 0) {
      rknn_destroy(context_);
      context_ = 0;
    }
    loaded_ = false;
    input_count_ = 0;
    output_count_ = 0;
    output_infos_.clear();
    output_attrs_.clear();
    output_float_buffers_.clear();
    reusable_outputs_.clear();
    output_preallocation_supported_ = true;
    model_data_.clear();
  }

  std::string task_type_;
  std::string last_error_;
  int input_width_{640};
  int input_height_{640};
  bool loaded_{false};
  bool dump_io_{false};
  rknn_context context_{0};
  std::uint32_t input_count_{0};
  std::uint32_t output_count_{0};
  std::vector<std::uint8_t> model_data_;
  std::vector<TensorInfo> output_infos_;
  std::vector<TensorInfo> input_infos_;
  std::vector<rknn_tensor_attr> output_attrs_;
  std::vector<std::vector<float>> output_float_buffers_;
  std::vector<rknn_output> reusable_outputs_;
  bool output_preallocation_supported_{true};
  std::mutex mutex_;
};

}  // namespace

std::unique_ptr<RknnRunner> make_real_rknn_runner(const std::string& task_type) {
  return std::make_unique<RknnRunnerReal>(task_type);
}

}  // namespace visionops::runtime
