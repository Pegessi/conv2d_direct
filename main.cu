#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <unistd.h>
#include <cudnn.h>

#include "dense_help_func.cu"
#include "conv.cu"

// #define DEBUG
#define size_matrix int
#define element_type float
#define CUDNN
// #define CPU_SERIAL
#define GPU_PARALLEL

void generate_matrix(element_type *mat, int m, int n);
void generate_filter(element_type *mat, int size);

int main()
{
    // 初始化CPU数据 32 16 30 14
    const int N = 1;    // batch size
    const int inC = 3; // inChannel >15会出错？
    const int inH = 1920;
    const int inW = 2400;
    const int outC = 3; // outChannel 每个都与不同的卷积核运算 之后再分别放到outChannel中
    const int kernelH = 6;
    const int kernelW = 6;
    const int outH = inH - kernelH + 1;
    const int outW = inW - kernelW + 1;

    // cudaSetDevice(7);

    element_type *inputs, *outputs, *kernel;
    int in_size = N * inC * inH * inW,
        out_size = N * outC * outH * outW,
        filter_size = outC * inC * kernelH * kernelW;
    inputs = (element_type *)malloc(in_size * sizeof(element_type));
    outputs = (element_type *)malloc(out_size * sizeof(element_type));
    kernel = (element_type *)malloc(filter_size * sizeof(element_type));
    for (int i = 0; i < in_size; i++)
    {
        inputs[i] = rand() % 10;
    }
    for (int i = 0; i < filter_size; i += 3)
    {
        kernel[i] = -1;
        kernel[i + 1] = 0;
        kernel[i + 2] = 1;
        // kernel[i + 3] = 1;
    }
    for (int i = 0; i < out_size; i++)
    {
        outputs[i] = 0;
    }
    // 计时数据
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int iters = 100, warmup = 10;
    float msecTotal = 0;
    double msecPerMatrixMul[2] = {0, 0}, gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = out_size * inC * kernelH * kernelW;

#ifdef CUDNN
    /* ---- CUDNN CONV BEGIN ----*/
    // 初始化cudnn及相关Tensor描述符
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               N, inC, inH, inW);

    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               N, outC, outH, outW);

    cudnnFilterDescriptor_t kernel_desc;
    cudnnCreateFilterDescriptor(&kernel_desc);
    cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               outC, inC, kernelH, kernelW); // k-outputChannel c-inputChannel h w

    // 卷积描述符
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    // pad_h-padding height pad_w u-vertical filter stride v-horizontal filter stride
    cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1,
                                    CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
    // 运算空间
    size_t space_size = 0;
    cudnnConvolutionFwdAlgo_t alg_kind = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnStatus_t Error =
        cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc,
                                                kernel_desc, conv_desc, output_desc,
                                                alg_kind,
                                                &space_size);

    if (Error != CUDNN_STATUS_SUCCESS)
    {
        fprintf(stderr, "calc spacesize failed!\n");
    }

    void *workspace = nullptr;
    cudaMalloc(&workspace, space_size);

    // printf("space size: %ld\n", space_size); // 打印出是0？

    // 初始化GPU数据
    auto alpha = 1.0f;
    auto beta = 0.0f;
    size_t input_size = N * inC * inH * inW * sizeof(float);
    size_t kernel_size = outC * inC * kernelH * kernelW * sizeof(float);
    size_t output_size = N * outC * outH * outW * sizeof(float);

    void *dev_input = nullptr;
    cudaMalloc(&dev_input, input_size);
    cudaMemcpy(dev_input, inputs, input_size, cudaMemcpyHostToDevice);
    void *dev_kernel = nullptr;
    cudaMalloc(&dev_kernel, kernel_size);
    cudaMemcpy(dev_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);
    void *dev_output = nullptr;
    cudaMalloc(&dev_output, output_size);

    // 检查核函数错误
    // cudaError_t err = cudaSetDevice(0);
    // if (err != cudaSuccess)
    // {
    //     errorHandler(err, __FILE__, __LINE__);
    // }
    // cudaEventRecord(start);
    for (int run = 0; run < iters + warmup; run++)
    {
        if (run == warmup)
            cudaEventRecord(start);
        Error = cudnnConvolutionForward(handle,
                                        &alpha, input_desc,
                                        dev_input,
                                        kernel_desc, dev_kernel,
                                        conv_desc,
                                        alg_kind,
                                        workspace,
                                        space_size,
                                        &beta,
                                        output_desc,
                                        dev_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(outputs, dev_output, output_size, cudaMemcpyDeviceToHost);
    printf("cudnn cost time: %f\n", msecTotal / (iters));
    msecPerMatrixMul[0] = msecTotal / (iters);
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);

    if (Error != CUDNN_STATUS_SUCCESS)
    {
        fprintf(stderr, "cudnn forward failed!\n");
    }

    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);
    /* ---- CUDNN CONV END ---- */
#endif

#ifdef CPU_SERIAL
    /* ---- CPU serial BEGIN ---- */
    float *self_outputs;
    self_outputs = (element_type *)malloc(out_size * sizeof(element_type));
    for (int i = 0; i < out_size; i++)
    {
        self_outputs[i] = 0;
    }
    serial_convolution(inputs, self_outputs, kernel, N, inC, inH, inW, outC, outH, outW, kernelH, kernelW);
    for (int i = 0; i < outH * outW; i++)
    {
        printf("%.2f|%.2f\n", outputs[i], self_outputs[i]);
    }
    /* ---- CPU serial END ---- */
#endif

#ifdef GPU_PARALLEL
    /* ---- SELF CUDA BEGIN ---- */
    // 初始化GPU数据
    float *self_outputs;
    self_outputs = (element_type *)malloc(out_size * sizeof(element_type));
    element_type *self_dev_input, *self_dev_kernel, *self_dev_output;
    cudaMalloc(&self_dev_input, input_size);
    cudaMalloc(&self_dev_kernel, kernel_size);
    cudaMalloc(&self_dev_output, output_size);
    cudaMemcpy(self_dev_input, inputs, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(self_dev_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(self_dev_output, self_outputs, output_size, cudaMemcpyHostToDevice);

    const int THREAD_HEIGHT = 1, THREAD_WIDTH = 1,                                         // 一个线程负责的元素数
        KERNEL_HEIGHT = kernelH, KERNEL_WIDTH = kernelW,                                   // 卷积核大小
        BLOCK_HEIGHT = 8, BLOCK_WIDTH = 4,                                                 // 分块大小
        MALLOC_KERNEL_HEIGHT = KERNEL_HEIGHT % 2 == 0 ? KERNEL_HEIGHT : KERNEL_HEIGHT + 1, // 用于kernel在SMEM的修正尺寸 奇数尺寸无法分配空间
        MALLOC_KERNEL_WIDTH = KERNEL_WIDTH % 2 == 0 ? KERNEL_WIDTH : KERNEL_WIDTH + 1,     // 用于kernel在SMEM的修正尺寸
        MALLOC_BLOCK_HEIGHT = (BLOCK_HEIGHT + KERNEL_HEIGHT) * 2,                          // 用于block在SMEM的修正尺寸
        MALLOC_BLOCK_WIDTH = (BLOCK_WIDTH + KERNEL_WIDTH) * 2,                             // 用于block在SMEM的修正尺寸
        MALLOC_TEMP_SIZE = outC * 4;                                                       // 用于计算时暂存计算结果的寄存器大小

    // printf("%d %d %d %d %d\n",KERNEL_HEIGHT,KERNEL_WIDTH,MALLOC_BLOCK_HEIGHT,MALLOC_BLOCK_WIDTH,MALLOC_TEMP_SIZE);

    // 第一个参数是x轴范围，第二个是y轴
    dim3 dimGrid(outW / BLOCK_WIDTH, outH / BLOCK_HEIGHT);
    dim3 dimBlock(BLOCK_WIDTH / THREAD_WIDTH, BLOCK_HEIGHT / THREAD_HEIGHT);

    // cudaEventRecord(start);

    for (int run = 0; run < iters + warmup; run++)
    {
        if (run == warmup)
            cudaEventRecord(start);
        v1_convolution<BLOCK_HEIGHT, BLOCK_WIDTH, KERNEL_HEIGHT, KERNEL_WIDTH, MALLOC_TEMP_SIZE,
                       MALLOC_KERNEL_HEIGHT, MALLOC_KERNEL_WIDTH, MALLOC_BLOCK_HEIGHT, MALLOC_BLOCK_WIDTH>
            <<<dimGrid, dimBlock>>>(self_dev_input, self_dev_output, self_dev_kernel,
                                    N, inC, inH, inW, outC, outH, outW, kernelH, kernelW);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(self_outputs, self_dev_output, output_size, cudaMemcpyDeviceToHost);

    printf("my conv cost time: %f\n", msecTotal / (iters));
    msecPerMatrixMul[1] = msecTotal / (iters);
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    for (int i = 0; i < outC * outH * outW; i++)
    {
        if (outputs[i] != -self_outputs[i])
        {
            printf("WRONG VALUE: %.2f|%.2f at %d\n", outputs[i], -self_outputs[i], i);
            break;
        }
    }
    /* ---- SELF CUDA END ---- */
#endif

    // _exit(0);
    cudaFree(dev_input);
    cudaFree(dev_kernel);
    cudaFree(dev_output);
#ifdef GPU_PARALLEL
    cudaFree(self_dev_input);
    cudaFree(self_dev_kernel);
    cudaFree(self_dev_output);
#endif
    free(self_outputs);
    free(inputs);
    free(outputs);
    free(kernel);
    return 0;
}

__host__ void generate_matrix(element_type *mat, int m, int n)
{

    for (int i = 0; i < m * n; i++)
    {
        // printf("total %d\n", m * n);
        mat[i] = rand() % 10;
        // printf("1\n");
        // int row = (i / n), col = (i % n);                         // 行数与列数
        // int row_block = row / block_m, col_block = col / block_n; // 块行号与列号
        // if ((row_block * k_block + col_block) % stride == 0)
        // {
        //     mat[i] = 1;
        // }
        // else
        // {
        //     mat[i] = 0;
        // }
    }
}

__host__ void generate_filter(element_type *mat, int size)
{
    if (size == 3)
    {
        for (int i = 0; i <= 6; i += 3)
        {
            mat[i] = 1;
            mat[i + 1] = 0;
            mat[i + 2] = -1;
        }
    }
}
