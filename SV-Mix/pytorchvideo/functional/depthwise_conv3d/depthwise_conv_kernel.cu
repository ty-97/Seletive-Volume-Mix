#include <torch/torch.h>
#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include "caffe_cuda_macro.h"

using namespace at;

template <typename scalar_t>
__global__ void ConvForward(const int nthreads,
    const scalar_t* const bottom_data, const int num, const int channels,
    const int length, const int height, const int width, const int conved_length, const int conved_height,
    const int conved_width, const int kernel_l, const int kernel_h, const int kernel_w,
    const int stride_l, const int stride_h, const int stride_w, const int pad_l, const int pad_h, const int pad_w,
    scalar_t* const top_data, const scalar_t* const weight, const scalar_t* const bias, const bool bias_term_) {
    CUDA_KERNEL_LOOP(index, nthreads) {

        const int pw = index % conved_width;
        const int ph = (index / conved_width) % conved_height;
        const int pl = (index / conved_width / conved_height) % conved_length;
        const int c = (index / conved_width / conved_height / conved_length) % channels;
        const int n = index / conved_width / conved_height / conved_length / channels;
        int lstart = pl * stride_l - pad_l;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int lend = min(lstart + kernel_l, length);
        int hend = min(hstart + kernel_h, height);
        int wend = min(wstart + kernel_w, width);
        lstart = max(lstart, 0);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        scalar_t aveval = 0;
        const scalar_t* const bottom_slice =
            bottom_data + (n * channels + c) * length * height * width;
        const scalar_t* const weight_slice =
            weight + c * kernel_l * kernel_h * kernel_w;

        int klstart = lend<kernel_l ? kernel_l - lend : 0;
        int khstart = hend<kernel_h ? kernel_h - hend : 0;
        int kwstart = wend<kernel_w ? kernel_w - wend : 0;
        for (int l = lstart; l < lend; ++l) {
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {

                    aveval += bottom_slice[(l * height + h) * width + w] * weight_slice[((klstart + l - lstart) * kernel_h + (khstart + h - hstart)) * kernel_w + (kwstart + w - wstart)];

                }
            }
        }
        if (bias_term_) {
            aveval += bias[c];
        }
        top_data[index] = aveval;
    }
}

// depthwise 1x3x3, pad 0x0x0
template <typename scalar_t>
__global__ void ConvForward2(const int nthreads,
    const scalar_t* const bottom_data, const int num, const int channels,
    const int length, const int height, const int width, const int conved_length, const int conved_height,
    const int conved_width, const int kernel_l, const int kernel_h, const int kernel_w,
    const int stride_l, const int stride_h, const int stride_w, const int pad_l, const int pad_h, const int pad_w,
    scalar_t* const top_data, const scalar_t* const weight, const scalar_t* const bias, const bool bias_term_) {
    CUDA_KERNEL_LOOP(index, nthreads) {

        const int pw = index % conved_width;
        const int ph = (index / conved_width) % conved_height;
        const int pl = (index / conved_width / conved_height) % conved_length;
        const int c = (index / conved_width / conved_height / conved_length) % channels;
        const int n = index / conved_width / conved_height / conved_length / channels;

		    int lstart = pl * stride_l;
		    int hstart = ph * stride_h;
		    int wstart = pw * stride_w;

        scalar_t aveval = 0;
        const scalar_t* const bottom_slice =
            bottom_data + (((n * channels + c) * length + lstart) * height + hstart) * width + wstart;
        const scalar_t* const weight_slice =
            weight + c * kernel_l * kernel_h * kernel_w;

        aveval += bottom_slice[0] * weight_slice[0];
        aveval += bottom_slice[1] * weight_slice[1];
        aveval += bottom_slice[2] * weight_slice[2];
        aveval += bottom_slice[width] * weight_slice[3];
        aveval += bottom_slice[width + 1] * weight_slice[4];
        aveval += bottom_slice[width + 2] * weight_slice[5];
        aveval += bottom_slice[width + width] * weight_slice[6];
        aveval += bottom_slice[width + width + 1] * weight_slice[7];
        aveval += bottom_slice[width + width + 2] * weight_slice[8];

        if (bias_term_) {
            aveval += bias[c];
        }
        top_data[index] = aveval;
    }
}

// depthwise 1x5x5, pad 0x0x0
template <typename scalar_t>
__global__ void ConvForward3(const int nthreads,
    const scalar_t* const bottom_data, const int num, const int channels,
    const int length, const int height, const int width, const int conved_length, const int conved_height,
    const int conved_width, const int kernel_l, const int kernel_h, const int kernel_w,
    const int stride_l, const int stride_h, const int stride_w, const int pad_l, const int pad_h, const int pad_w,
    scalar_t* const top_data, const scalar_t* const weight, const scalar_t* const bias, const bool bias_term_) {
    CUDA_KERNEL_LOOP(index, nthreads) {

        const int pw = index % conved_width;
        const int ph = (index / conved_width) % conved_height;
        const int pl = (index / conved_width / conved_height) % conved_length;
        const int c = (index / conved_width / conved_height / conved_length) % channels;
        const int n = index / conved_width / conved_height / conved_length / channels;

		    int lstart = pl * stride_l;
		    int hstart = ph * stride_h;
		    int wstart = pw * stride_w;

        scalar_t aveval = 0;
        const scalar_t* const bottom_slice =
            bottom_data + (((n * channels + c) * length + lstart) * height + hstart) * width + wstart;
        const scalar_t* const weight_slice =
            weight + c * kernel_l * kernel_h * kernel_w;

        aveval += bottom_slice[0] * weight_slice[0];
        aveval += bottom_slice[1] * weight_slice[1];
        aveval += bottom_slice[2] * weight_slice[2];
        aveval += bottom_slice[3] * weight_slice[3];
        aveval += bottom_slice[4] * weight_slice[4];
        aveval += bottom_slice[width] * weight_slice[5];
        aveval += bottom_slice[width + 1] * weight_slice[6];
        aveval += bottom_slice[width + 2] * weight_slice[7];
        aveval += bottom_slice[width + 3] * weight_slice[8];
        aveval += bottom_slice[width + 4] * weight_slice[9];
        aveval += bottom_slice[width + width] * weight_slice[10];
        aveval += bottom_slice[width + width + 1] * weight_slice[11];
        aveval += bottom_slice[width + width + 2] * weight_slice[12];
        aveval += bottom_slice[width + width + 3] * weight_slice[13];
        aveval += bottom_slice[width + width + 4] * weight_slice[14];
        aveval += bottom_slice[width + width + width] * weight_slice[15];
        aveval += bottom_slice[width + width + width + 1] * weight_slice[16];
        aveval += bottom_slice[width + width + width + 2] * weight_slice[17];
        aveval += bottom_slice[width + width + width + 3] * weight_slice[18];
        aveval += bottom_slice[width + width + width + 4] * weight_slice[19];
        aveval += bottom_slice[width + width + width + width] * weight_slice[20];
        aveval += bottom_slice[width + width + width + width + 1] * weight_slice[21];
        aveval += bottom_slice[width + width + width + width + 2] * weight_slice[22];
        aveval += bottom_slice[width + width + width + width + 3] * weight_slice[23];
        aveval += bottom_slice[width + width + width + width + 4] * weight_slice[24];
        
        if (bias_term_) {
            aveval += bias[c];
        }
        top_data[index] = aveval;
    }
}


at::Tensor DepthWiseConvForwardLaucher(const at::Tensor input, const at::Tensor weight, const at::Tensor bias,
                                       const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h, const int stride_w,
                                       const int pad_l, const int pad_h, const int pad_w, const bool bias_term_){
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_length = input.size(2);
    const auto input_height = input.size(3);
    const auto input_width = input.size(4);

    const auto kernal_extent_l = /* dilation_l * */ (kernel_l - 1) + 1;
    const auto conved_length = (input_length + 2 * pad_l - kernal_extent_l) / stride_l + 1;
    
    const auto kernal_extent_h = /* dilation_h * */ (kernel_h - 1) + 1;
    const auto conved_height = (input_height + 2 * pad_h - kernal_extent_h) / stride_h + 1;

    const auto kernal_extent_w = /* dilation_w * */ (kernel_w - 1) + 1;
    const auto conved_width = (input_width + 2 * pad_w - kernal_extent_w) / stride_w + 1;

    IntList size = {batch_size, channels, conved_length, conved_height, conved_width};
    auto output = at::zeros(size, input.options());
    const auto count = batch_size * channels * conved_length * conved_height * conved_width;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(), "ConvLaucherForward",
        ([&]{
            const scalar_t *bottom_data = input.data<scalar_t>();
            scalar_t *top_data = output.data<scalar_t>();
            const scalar_t *depthwise_weight = weight.data<scalar_t>();

            if (bias_term_)
            {
                const scalar_t *depthwise_bias = bias.data<scalar_t>();
                ConvForward<scalar_t><<<GET_BLOCKS(count), THREADS_PER_BLOCK>>>(count, bottom_data, batch_size,
                    channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
                    stride_w, pad_l, pad_h, pad_w, top_data, depthwise_weight, depthwise_bias, bias_term_);
//                if ((kernel_l == 1) && (kernel_h == 5) && (kernel_w == 5) && (pad_l == 0) && (pad_h == 0) && (pad_w == 0))
//                {
//                    ConvForward3<scalar_t><<<GET_BLOCKS(count), THREADS_PER_BLOCK>>>(count, bottom_data, batch_size,
//                        channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
//                        stride_w, pad_l, pad_h, pad_w, top_data, depthwise_weight, depthwise_bias, bias_term_);
//                }
//                else
//                {
//                    if ((kernel_l == 1) && (kernel_h == 3) && (kernel_w == 3) && (pad_l == 0) && (pad_h == 0) && (pad_w == 0))
//                    {
//                        ConvForward2<scalar_t><<<GET_BLOCKS(count), THREADS_PER_BLOCK>>>(count, bottom_data, batch_size,
//                            channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
//                            stride_w, pad_l, pad_h, pad_w, top_data, depthwise_weight, depthwise_bias, bias_term_);
//    				        }
//    				        else
//    				        {
//                        ConvForward<scalar_t><<<GET_BLOCKS(count), THREADS_PER_BLOCK>>>(count, bottom_data, batch_size,
//                            channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
//                            stride_w, pad_l, pad_h, pad_w, top_data, depthwise_weight, depthwise_bias, bias_term_);
//                    }
//                }
            }
            else
            {
                ConvForward<scalar_t><<<GET_BLOCKS(count), THREADS_PER_BLOCK>>>(count, bottom_data, batch_size,
                    channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
                    stride_w, pad_l, pad_h, pad_w, top_data, depthwise_weight, 0, bias_term_);
//                if ((kernel_l == 1) && (kernel_h == 5) && (kernel_w == 5) && (pad_l == 0) && (pad_h == 0) && (pad_w == 0))
//                {
//                    ConvForward3<scalar_t><<<GET_BLOCKS(count), THREADS_PER_BLOCK>>>(count, bottom_data, batch_size,
//                        channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
//                        stride_w, pad_l, pad_h, pad_w, top_data, depthwise_weight, 0, bias_term_);
//                }
//                else
//                {
//                    if ((kernel_l == 1) && (kernel_h == 3) && (kernel_w == 3) && (pad_l == 0) && (pad_h == 0) && (pad_w == 0))
//                    {
//                        ConvForward2<scalar_t><<<GET_BLOCKS(count), THREADS_PER_BLOCK>>>(count, bottom_data, batch_size,
//                            channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
//                            stride_w, pad_l, pad_h, pad_w, top_data, depthwise_weight, 0, bias_term_);
//                    }
//                    else
//                    {
//                        ConvForward<scalar_t><<<GET_BLOCKS(count), THREADS_PER_BLOCK>>>(count, bottom_data, batch_size,
//                            channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
//                            stride_w, pad_l, pad_h, pad_w, top_data, depthwise_weight, 0, bias_term_);
//                    }
//                }
            }
        }));
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return output;
}

template <typename scalar_t>
__global__ void ConvBackward(const int nthreads,
    const scalar_t* const top_diff,
    const int num, const int channels, const int length, const int height,
    const int width, const int conved_length, const int conved_height, const int conved_width,
    const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h,
    const int stride_w, const int pad_l, const int pad_h, const int pad_w,
    scalar_t* const bottom_diff,
    const scalar_t* const weight) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int w = index % width + pad_w;
        const int h = (index / width) % height + pad_h;
        const int l = (index / width / height) % length + pad_l;
        const int c = (index / width / height / length) % channels;
        const int n = index / width / height / length / channels;

        const int plstart = (l < kernel_l) ? 0 : (l - kernel_l) / stride_l + 1;
        const int plend = min(l / stride_l + 1, conved_length);
        const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
        const int phend = min(h / stride_h + 1, conved_height);
        const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
        const int pwend = min(w / stride_w + 1, conved_width);

        const int klstart = (l >= kernel_l) ? ((l - kernel_l) % stride_l) + (kernel_l - stride_l) : l;
        const int khstart = (h >= kernel_h) ? ((h - kernel_h) % stride_h) + (kernel_h - stride_h) : h;
        const int kwstart = (w >= kernel_w) ? ((w - kernel_w) % stride_w) + (kernel_w - stride_w) : w;

        scalar_t gradient = 0;
        const scalar_t* const top_diff_slice =
            top_diff + (n * channels + c) * conved_length * conved_height * conved_width;

        const scalar_t* const weight_slice = weight + c * kernel_l * kernel_h * kernel_w;

        for (int pl = plstart; pl < plend; ++pl) {
            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    int kl = klstart - (pl - plstart)*stride_l;
                    int kh = khstart - (ph - phstart)*stride_h;
                    int kw = kwstart - (pw - pwstart)*stride_w;
                    gradient += top_diff_slice[(pl * conved_height + ph) * conved_width + pw] * weight_slice[(kl * kernel_h + kh)*kernel_w + kw];
                }
            }
        }
        bottom_diff[index] = gradient;
    }
}

#define DIVIDE_CEIL(a,b) a/b+((a/b*b)<a)


template <typename scalar_t>
__global__ void ConvBackwardWeight(const int nthreads,
    const scalar_t* const top_diff,
    const int num, const int channels, const int length, const int height,
    const int width, const int conved_length, const int conved_height, const int conved_width,
    const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h,
    const int stride_w, const int pad_l, const int pad_h, const int pad_w,
    scalar_t* const weight_diff,
    const scalar_t* const bottom_data) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int kw = index % kernel_w;
        const int kh = (index / kernel_w) % kernel_h;
        const int kl = (index / kernel_w / kernel_h) % kernel_l;
        const int c = index / kernel_w / kernel_h / kernel_l;

        scalar_t gradient = 0;
        for (int n = 0; n<num; n++) {

            const scalar_t* const top_diff_slice = top_diff + (n * channels + c) * conved_length * conved_height * conved_width;
            const scalar_t* const bottom_data_slice = bottom_data + (n * channels + c) * length * height * width;

            const int plstart = max(DIVIDE_CEIL((pad_l - kl), stride_l), 0);
            const int plend = min(DIVIDE_CEIL((length + pad_l - kl), stride_l), conved_length);

            const int phstart = max(DIVIDE_CEIL((pad_h - kh), stride_h), 0);
            const int phend = min(DIVIDE_CEIL((height + pad_h - kh), stride_h), conved_height);

            const int pwstart = max(DIVIDE_CEIL((pad_w - kw), stride_w), 0);
            const int pwend = min(DIVIDE_CEIL((width + pad_w - kw), stride_w), conved_width);
            for (int pl = plstart; pl < plend; pl++) {
                for (int ph = phstart; ph < phend; ph++) {
                    for (int pw = pwstart; pw < pwend; pw++) {
                        const int l = pl*stride_l + kl - pad_l;
                        const int h = ph*stride_h + kh - pad_h;
                        const int w = pw*stride_w + kw - pad_w;
                        gradient += top_diff_slice[(pl * conved_height + ph) * conved_width + pw] * bottom_data_slice[(l * height + h)*width + w];
                    }
                }
            }
        }
        weight_diff[((c * kernel_l + kl) * kernel_h + kh) * kernel_w + kw] += gradient;
    }
}

template <typename scalar_t>
__global__ void ConvBackwardWeight2(
    const scalar_t* const top_diff,
    const int num, const int channels, const int length, const int height,
    const int width, const int conved_length, const int conved_height, const int conved_width,
    const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h,
    const int stride_w, const int pad_l, const int pad_h, const int pad_w,
    scalar_t* const weight_diff,
    const scalar_t* const bottom_data) {

    __shared__ scalar_t buffer[CAFFE_CUDA_NUM_THREADS];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int kw = bid % kernel_w;
    const int kh = (bid / kernel_w) % kernel_h;
    const int kl = (bid / kernel_w / kernel_h) % kernel_l;
    const int c = bid / kernel_w / kernel_h / kernel_l;

    buffer[tid] = 0;

    for (int index = tid; index < num * conved_length * conved_height * conved_width; index += blockDim.x)
    {
        const int pw = index % conved_width;
        const int ph = (index / conved_width) % conved_height;
        const int pl = (index / conved_width / conved_height) % conved_length;
        const int n = index / conved_width / conved_height / conved_length;
        const int lstart = pl * stride_l - pad_l;
        const int hstart = ph * stride_h - pad_h;
        const int wstart = pw * stride_w - pad_w;
        const int l = lstart + kl;
        const int h = hstart + kh;
        const int w = wstart + kw;
        if (l < 0 || h < 0 || w < 0 || l >= length || h >= height || w >= width)
        {
            continue;
        }

        buffer[tid] += bottom_data[(((n * channels + c) * length + l) * height + h) * width + w]
            * top_diff[(((n * channels + c) * conved_length + pl) * conved_height + ph) * conved_width + pw];
    }
    __syncthreads();
    // do tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            buffer[tid] += buffer[tid + s];
        }
        __syncthreads();
    }
    // save the result back
    if (tid == 0) {
        weight_diff[bid] += buffer[0];
    }
}

template <typename scalar_t>
__global__ void ConvBackwardWeight3(
    const scalar_t* const top_diff,
    const int num, const int channels, const int length, const int height,
    const int width, const int conved_length, const int conved_height, const int conved_width,
    const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h,
    const int stride_w, const int pad_l, const int pad_h, const int pad_w,
    scalar_t* const weight_diff,
    const scalar_t* const bottom_data) {

    __shared__ scalar_t buffer[CAFFE_CUDA_NUM_THREADS];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int kw = bid % kernel_w;
    const int kh = (bid / kernel_w) % kernel_h;
    const int kl = (bid / kernel_w / kernel_h) % kernel_l;
    const int c = bid / kernel_w / kernel_h / kernel_l;

    buffer[tid] = 0;

    for (int index = tid; index < num * conved_length * conved_height; index += blockDim.x)
    {
        const int ph = index % conved_height;
        const int pl = (index / conved_height) % conved_length;
        const int n = index / conved_height / conved_length;
        const int lstart = pl * stride_l - pad_l;
        const int hstart = ph * stride_h - pad_h;
        const int l = lstart + kl;
        const int h = hstart + kh;

        if (l < 0 || h < 0 || l >= length || h >= height)
        {
            continue;
        }

        const scalar_t* const bottom_slice =
            bottom_data + (((n * channels + c) * length + l) * height + h) * width;

        const scalar_t* const top_slice = top_diff + (((n * channels + c) * conved_length + pl) * conved_height + ph) * conved_width;

        for (int pw = 0; pw < conved_width; ++pw)
        {
            const int wstart = pw * stride_w - pad_w;
            const int w = wstart + kw;
            if (w < 0 || w >= width)
            {
                continue;
            }

            buffer[tid] += bottom_slice[w] * top_slice[pw];
        }
    }
    __syncthreads();
    // do tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            buffer[tid] += buffer[tid + s];
        }
        __syncthreads();
    }
    // save the result back
    if (tid == 0) {
        weight_diff[bid] += buffer[0];
    }
}

template <typename scalar_t>
__global__ void ConvBackwardWeight4(const int nthreads,
    const scalar_t* const top_diff,
    const int num, const int channels, const int length, const int height,
    const int width, const int conved_length, const int conved_height, const int conved_width,
    const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h,
    const int stride_w, const int pad_l, const int pad_h, const int pad_w,
    scalar_t* const weight_diff,
    const scalar_t* const bottom_data) {

    __shared__ scalar_t buffer[CAFFE_CUDA_NUM_THREADS];
    const int tid = threadIdx.x;
    const int part = tid % 8;
    const int part_w = part % 2;
    const int part_h = (part / 2) % 2;
    const int part_l = (part / 2 / 2) % 2;
    const int id = tid / 8;

    for (int index = blockIdx.x * blockDim.x / 8 + id; index < nthreads; index += blockDim.x / 8 * gridDim.x)
    {
        const int kw = index % kernel_w;
        const int kh = (index / kernel_w) % kernel_h;
        const int kl = (index / kernel_w / kernel_h) % kernel_l;
        const int c = index / kernel_w / kernel_h / kernel_l;

        const int plstart = max(DIVIDE_CEIL((pad_l - kl + (length / 2) * part_l), stride_l), 0);
        const int plend = min(DIVIDE_CEIL((length / 2 + pad_l - kl + (length - length / 2) * part_l), stride_l), conved_length);

        const int phstart = max(DIVIDE_CEIL((pad_h - kh + (height / 2) * part_h), stride_h), 0);
        const int phend = min(DIVIDE_CEIL((height / 2 + pad_h - kh + (height - height / 2) * part_h), stride_h), conved_height);

        const int pwstart = max(DIVIDE_CEIL((pad_w - kw + (width / 2) * part_w), stride_w), 0);
        const int pwend = min(DIVIDE_CEIL((width / 2 + pad_w - kw + (width - width / 2) * part_w), stride_w), conved_width);

        scalar_t gradient = 0;
        for (int n = 0; n<num; n++) {
            const scalar_t* const top_diff_slice = top_diff + (n * channels + c) * conved_length * conved_height * conved_width;
            const scalar_t* const bottom_data_slice = bottom_data + (n * channels + c) * length * height * width;

            for (int pl = plstart; pl < plend; pl++) {
                for (int ph = phstart; ph < phend; ph++) {
                    for (int pw = pwstart; pw < pwend; pw++) {
                        const int l = pl*stride_l + kl - pad_l;
                        const int h = ph*stride_h + kh - pad_h;
                        const int w = pw*stride_w + kw - pad_w;
                        gradient += top_diff_slice[(pl * conved_height + ph) * conved_width + pw] * bottom_data_slice[(l * height + h)*width + w];
                    }
                }
            }
        }
        buffer[tid] = gradient;

        __syncthreads();
        // do tree reduction
        for (int s = 4; s > 0; s >>= 1) {
            if (part < s) {
                buffer[tid] += buffer[tid + s];
            }
            __syncthreads();
        }
        // save the result back
        if (part == 0) {
            weight_diff[((c * kernel_l + kl) * kernel_h + kh) * kernel_w + kw] += buffer[tid];
        }
    }
}


template <typename scalar_t>
__global__ void ConvBackwardBias(const int nthreads,
    const scalar_t* const top_diff,
    const int num, const int channels, const int length, const int height,
    const int width, const int conved_length, const int conved_height, const int conved_width,
    const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h,
    const int stride_w, const int pad_l, const int pad_h, const int pad_w,
    scalar_t* const bias_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int c = index;
        scalar_t gradient = 0;
        for (int n = 0; n<num; n++) {
            const scalar_t* const top_diff_slice =
                top_diff + (n * channels + c) * conved_length * conved_height * conved_width;
            for (int pl = 0; pl < conved_length; pl++) {
                for (int ph = 0; ph < conved_height; ph++) {
                    for (int pw = 0; pw < conved_width; pw++) {
                        gradient += top_diff_slice[(pl * conved_height + ph) * conved_width + pw];
                    }
                }
            }
        }
        bias_diff[c] += gradient;
    }
}

std::vector<at::Tensor> DepthWiseConvBackwarddLaucher(const at::Tensor output_grad, const at::Tensor input, const at::Tensor weight, const at::Tensor bias,
                                                      const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h, const int stride_w,
                                                      const int pad_l, const int pad_h, const int pad_w, const bool bias_term_, const int alg_id){
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_length = input.size(2);
    const auto input_height = input.size(3);
    const auto input_width = input.size(4);

    const auto kernal_extent_l = /* dilation_l * */ (kernel_l - 1) + 1;
    const auto conved_length = (input_length + 2 * pad_l - kernal_extent_l) / stride_l + 1;
    const auto kernal_extent_h = /* dilation_h * */ (kernel_h - 1) + 1;
    const auto conved_height = (input_height + 2 * pad_h - kernal_extent_h) / stride_h + 1;
    const auto kernal_extent_w = /* dilation_w * */ (kernel_w - 1) + 1;
    const auto conved_width = (input_width + 2 * pad_w - kernal_extent_w) / stride_w + 1;

    const int count_weight = channels * kernel_l * kernel_h * kernel_w;
    const int count_input = batch_size * channels * input_length * input_height * input_width;

    auto weight_diff = at::zeros_like(weight);
    auto bottom_diff = at::zeros_like(input);
    at::Tensor bias_diff;
    int count_bias = 0;

    if (bias_term_){
        count_bias = channels;
        bias_diff = at::zeros_like(bias);
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        output_grad.type(), "ConvLaucherBackward",
        ([&]{
            const scalar_t *bottom_data = input.data<scalar_t>();
            const scalar_t *depthwise_weight = weight.data<scalar_t>();
            const scalar_t *top_diff = output_grad.data<scalar_t>();
            scalar_t *depthwise_weight_diff = weight_diff.data<scalar_t>();
            scalar_t *depthwise_bottom_diff = bottom_diff.data<scalar_t>();

            if (bias_term_){
                scalar_t *depthwise_bias_diff = bias_diff.data<scalar_t>();
                ConvBackwardBias<scalar_t><<<GET_BLOCKS(count_bias), THREADS_PER_BLOCK>>>(count_bias, top_diff, batch_size,
                    channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
                    stride_w, pad_l, pad_h, pad_w, depthwise_bias_diff);
            }
            switch (alg_id)
				    {
				    case 0:
                ConvBackwardWeight<scalar_t><<<GET_BLOCKS(count_weight), THREADS_PER_BLOCK>>>(count_weight, top_diff, batch_size,
                    channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
                    stride_w, pad_l, pad_h, pad_w, depthwise_weight_diff, bottom_data);
                break;
            case 1:
                ConvBackwardWeight2<scalar_t> << <count_weight, THREADS_PER_BLOCK >> >(
						        top_diff, batch_size, channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
                    stride_w, pad_l, pad_h, pad_w, depthwise_weight_diff, bottom_data);
                break;
            case 2:
                ConvBackwardWeight3<scalar_t> << <count_weight, THREADS_PER_BLOCK >> >(
						        top_diff, batch_size, channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
                    stride_w, pad_l, pad_h, pad_w, depthwise_weight_diff, bottom_data);
                break;
            case 3:
                ConvBackwardWeight4<scalar_t> << <GET_BLOCKS(count_weight * 8), THREADS_PER_BLOCK >> >(
    						    count_weight, top_diff, batch_size, channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
                    stride_w, pad_l, pad_h, pad_w, depthwise_weight_diff, bottom_data);
                break;
            }
                
            ConvBackward<scalar_t><<<GET_BLOCKS(count_input), THREADS_PER_BLOCK>>>(count_input, top_diff, batch_size,
                channels, input_length, input_height, input_width, conved_length, conved_height, conved_width, kernel_l, kernel_h, kernel_w, stride_l, stride_h,
                stride_w, pad_l, pad_h, pad_w, depthwise_bottom_diff, depthwise_weight);

        }));
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    if (bias_term_){
        return {bottom_diff, weight_diff, bias_diff};
    }
    else{
        return {bottom_diff, weight_diff};
    }
}

