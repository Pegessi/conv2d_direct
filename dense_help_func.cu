#ifndef DENSE_HELP_FUNC
#define DENSE_HELP_FUNC

#include <cublas_v2.h>
#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

static const char *_cuBlasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}


#define checkCuBlasErrors(func)				\
{									\
    cublasStatus_t e = (func);			\
    if(e != CUBLAS_STATUS_SUCCESS)						                \
        printf ("%s %d CuBlas: %s", __FILE__,  __LINE__, _cuBlasGetErrorEnum(e));		\
}



#endif

// void errorHandler(cudaError_t error, const char *file, const int line)
// {
//     printf("CUDA error %d at %s:%d\n", error, file, line);
//     exit(EXIT_FAILURE);
// }

// #define CHECK_CUDNN_ERROR(err) \
//     if (err != CUDNN_STATUS_SUCCESS) { \
//         fprintf(stderr, "CUDNN error: %s\n", cudnnGetErrorString(err)); \
//         exit(EXIT_FAILURE); \
//     }
