#include <torch/torch.h>
#include <cmath>
#include <vector>

at::Tensor DepthWiseConvForwardLaucher(const at::Tensor input, const at::Tensor weight, const at::Tensor bias,
                                       const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h, const int stride_w,
                                       const int pad_l, const int pad_h, const int pad_w, const bool bias_term_);

std::vector<at::Tensor> DepthWiseConvBackwarddLaucher(const at::Tensor output_grad, const at::Tensor input, const at::Tensor weight, const at::Tensor bias,
                                                      const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h, const int stride_w,
                                                      const int pad_l, const int pad_h, const int pad_w, const bool bias_term_, const int alg_id);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

at::Tensor depthwise_conv_forward_cuda(const at::Tensor input, const at::Tensor weight, const at::Tensor bias,
                                       const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h, const int stride_w,
                                       const int pad_l, const int pad_h, const int pad_w, const bool bias_term_){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    return DepthWiseConvForwardLaucher(input, weight, bias, kernel_l, kernel_h, kernel_w, stride_l, stride_h, stride_w,
                                       pad_l, pad_h, pad_w, bias_term_);
}

std::vector<at::Tensor> depthwise_conv_backward_cuda(const at::Tensor output_grad, const at::Tensor input, const at::Tensor weight, const at::Tensor bias,
                                                      const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h, const int stride_w,
                                       const int pad_l, const int pad_h, const int pad_w, const bool bias_term_, const int alg_id){
  CHECK_INPUT(output_grad);
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  if(bias_term_){
    CHECK_INPUT(bias);
  }
  return DepthWiseConvBackwarddLaucher(output_grad, input, weight, bias, kernel_l, kernel_h, kernel_w, stride_l, stride_h, stride_w,
                                       pad_l, pad_h, pad_w, bias_term_, alg_id);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &depthwise_conv_forward_cuda, "Depthwise_Conv forward (CUDA)");
  m.def("backward", &depthwise_conv_backward_cuda, "Depthwise_Conv backward (CUDA)");
}
