#include <torch/csrc/inductor/aoti_torch/utils.h>

namespace torch_mlu::aot_inductor {

// Utility function to convert a pointer to an optional list of values
template <class T, class U>
inline std::optional<std::vector<T>> pointer_to_optional_list(
    U** ptr,
    int64_t len) {
  return ptr ? std::make_optional<std::vector<T>>(
                   torch::aot_inductor::pointer_to_list<T>(*ptr, len))
             : std::nullopt;
}

inline void new_tensor_list_handle(
    std::vector<at::Tensor>&& tensor_list,
    AtenTensorHandle** ret) {
  for (uint64_t i = 0; i < tensor_list.size(); i++) {
    *(ret[i]) =
        torch::aot_inductor::new_tensor_handle(std::move(tensor_list[i]));
  }
}

} // namespace torch_mlu::aot_inductor
