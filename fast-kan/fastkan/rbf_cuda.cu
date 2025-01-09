#include <torch/extension.h>
#include <vector>
#include <torch/torch.h>



template <typename scalar_t>
__global__ void rbf_kernel_variable(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ grid,
    scalar_t* __restrict__ output,
    const scalar_t inv_denom,
    const int num_elements,
    const int N,
    const int G
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;      // Index in the last dimension of x
    int grid_idx = blockIdx.y * blockDim.y + threadIdx.y; // Index in the grid
    int batch_idx = blockIdx.z;                           // Index in the "non-last" dimensions of x

    if (idx < N && grid_idx < G) {
        // Compute offset for the input and output tensors
        int input_offset = batch_idx * N + idx;
        int output_offset = batch_idx * (N * G) + idx * G + grid_idx;

        // Compute the difference and RBF output
        scalar_t diff = inv_denom * (x[input_offset] - grid[grid_idx]);
        output[output_offset] = expf(-diff * diff);
    }
}



torch::Tensor radial_basis_function_cuda(
    torch::Tensor x,
    torch::Tensor grid,
    float denominator
) {
    // Ensure tensors are contiguous and on CUDA
    TORCH_CHECK(x.is_contiguous() && grid.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(x.is_cuda() && grid.is_cuda(), "Tensors must be on CUDA");

    int num_dims = x.dim();
    TORCH_CHECK(num_dims >= 1, "Input tensor must have at least one dimension");

    // Get sizes
    int N = x.size(-1);      // Last dimension size
    int G = grid.size(0);    // Grid size
    int num_elements = x.numel() / N; // Product of "non-last" dimensions
    float inv_denom_float = 1.0f / denominator;

    // Prepare output tensor
    std::vector<int64_t> output_shape = x.sizes().vec();
    output_shape.push_back(G);  // Append grid size to output shape
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor output = torch::zeros(output_shape, options);

    // Configure CUDA kernel
    dim3 block(16, 16, 1);
    dim3 grid_dim((N + block.x - 1) / block.x, (G + block.y - 1) / block.y, num_elements);

    // Dispatch the kernel with AT_DISPATCH
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rbf_kernel_variable", ([&] {
        scalar_t inv_denom = static_cast<scalar_t>(inv_denom_float);
        rbf_kernel_variable<scalar_t><<<grid_dim, block>>>(
            x.data_ptr<scalar_t>(),
            grid.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            inv_denom,
            num_elements,
            N,
            G
        );
    }));

    return output;
}


//BACKWARD PASS:
extern "C" __global__ void rbf_kernel_variable_back(
    const float* __restrict__ x,
    const float* __restrict__ grid,
    float* __restrict__ output,
    const float inv_denom_sq,
    const int num_elements, // Number of elements in the "non-last" dimensions of x
    const int N,            // Size of the last dimension of x
    const int G             // Size of grid
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Index in the last dimension of x
    int grid_idx = blockIdx.y * blockDim.y + threadIdx.y; // Index in the grid
    int batch_idx = blockIdx.z; // Index in the "non-last" dimensions of x
    
    if (idx < N && grid_idx < G) {
        // Compute offset for the input and output tensors
        int input_offset = batch_idx * N + idx;
        int output_offset = batch_idx * (N * G) + idx * G + grid_idx;
        
        float diff = (x[input_offset] - grid[grid_idx]);
        float diff_denom =  diff * inv_denom_sq;

        output[output_offset] = -2.0 * diff_denom * expf(- diff_denom * diff);
        // output[output_offset] = diff;
    }
}


#include <torch/extension.h>
#include <vector>

// Launch BACKWARD CUDA kernel
torch::Tensor radial_basis_function_cuda_back(
    torch::Tensor x,
    torch::Tensor grid,
    float denominator
) {
    // Ensure tensors are contiguous and on CUDA
    TORCH_CHECK(x.is_contiguous() && grid.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(x.is_cuda() && grid.is_cuda(), "Tensors must be on CUDA");

    int num_dims = x.dim();
    TORCH_CHECK(num_dims >= 1, "Input tensor must have at least one dimension");

    // Get sizes
    int N = x.size(-1);  // Last dimension size
    int G = grid.size(0);  // Grid size
    int num_elements = x.numel() / N;  // Product of "non-last" dimensions
    const float inv_denom_sq = 1.0/(denominator*denominator);

    // Prepare output tensor
    std::vector<int64_t> output_shape = x.sizes().vec();
    output_shape.push_back(G);  // Append grid size to output shape
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor output = torch::zeros(output_shape, options);
    
    
    // Configure CUDA kernel
    dim3 block(16, 16, 1);
    dim3 grid_dim((N + block.x - 1) / block.x, (G + block.y - 1) / block.y, num_elements);

    // Launch kernel
    rbf_kernel_variable_back<<<grid_dim, block>>>(
        x.data_ptr<float>(),
        grid.data_ptr<float>(),
        output.data_ptr<float>(),
        inv_denom_sq,
        num_elements,
        N,
        G
    );

    return output;
}


extern "C" __global__ void rbf_grad_x_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ grads,
    float* __restrict__ grad_x,
    int num_elements,  // Product of all leading dimensions
    int G              // Grid size (last dimension)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Flattened index for leading dimensions

    if (idx < num_elements) {
        float sum = 0.0f;

        for (int g = 0; g < G; g++) {
            int offset = idx * G + g;  // Offset for accessing the G dimension
            sum += grad_output[offset] * grads[offset];
        }

        grad_x[idx] = sum;
    }
}


void launch_rbf_grad_x_kernel(
    torch::Tensor grad_output,
    torch::Tensor grads,
    torch::Tensor grad_x
) {
    // Ensure tensors are contiguous and on CUDA
    TORCH_CHECK(grad_output.is_contiguous() && grads.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(grad_output.is_cuda() && grads.is_cuda(), "Tensors must be on CUDA");

    // Get tensor shapes
    int G = grad_output.size(-1);  // Grid size
    int num_elements = grad_output.numel() / G;  // Product of leading dimensions

    // Configure CUDA kernel
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    // Launch the kernel
    rbf_grad_x_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        grads.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        num_elements,
        G
    );
}


// Combined CUDA kernel
// CUDA kernel using scalar_t


template <typename scalar_t>
__global__ void rbf_kernel_combined(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ grid,
    scalar_t* __restrict__ grad_output,
    float* __restrict__ grad_x,
    const scalar_t inv_denom,
    const int num_elements,
    const int N,
    const int G
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Index in the last dimension of x
    int grid_idx = blockIdx.y * blockDim.y + threadIdx.y; // Index in the grid
    int batch_idx = blockIdx.z; // Index in the "non-last" dimensions of x

    // Shared memory for the grid
    __shared__ scalar_t shared_grid[32]; // Ensure this is >= G. Adjust if G is larger.

    // Load grid into shared memory (one thread per grid element)
    if (threadIdx.y == 0 && blockIdx.y == 0) {
        for (int g = 0; g < G; ++g) {
            shared_grid[g] = grid[g];
        }
    }
    __syncthreads(); // Synchronize threads to ensure shared memory is loaded

    // Main computation
    if (idx < N && grid_idx < G) {
        int input_offset = batch_idx * N + idx;
        int output_offset = batch_idx * (N * G) + idx * G + grid_idx;

        // Compute difference and gradient component
        scalar_t diff = inv_denom * (x[input_offset] - shared_grid[grid_idx]);
        scalar_t gradient_comp = static_cast<scalar_t>(-2.0) * diff * inv_denom * expf(-diff * diff);
        float product = static_cast<float>(gradient_comp) * static_cast<float>(grad_output[output_offset]);

        // Accumulate grad_x
        atomicAdd(&grad_x[input_offset], product);
    }
}


// template <typename scalar_t>
// Launch combined CUDA kernel
torch::Tensor radial_basis_function_cuda_combined(
    torch::Tensor x,
    torch::Tensor grid,
    float denominator,
    torch::Tensor grad_output,
    int x_block_dim,
    int y_block_dim,
    int z_block_dim
) {
    // Ensure tensors are contiguous and on CUDA
    TORCH_CHECK(x.is_contiguous() && grid.is_contiguous() && grad_output.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(x.is_cuda() && grid.is_cuda() && grad_output.is_cuda(), "Tensors must be on CUDA");

    int num_dims = x.dim();
    TORCH_CHECK(num_dims >= 1, "Input tensor must have at least one dimension");

    // Get sizes
    int N = x.size(-1);
    int G = grid.size(0);
    int num_elements = x.numel() / N;

    std::vector<int64_t> output_shape = x.sizes().vec();
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor grad_x = torch::zeros(output_shape, options);

    // Calculate inverse denominator
    float inv_denom = 1.0f / denominator;

    // Configure CUDA kernel launch
    dim3 block(x_block_dim, y_block_dim, z_block_dim);
    dim3 grid_dim((N + block.x - 1) / block.x, (G + block.y - 1) / block.y, num_elements);

    // Dispatch the kernel using AT_DISPATCH
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rbf_kernel_combined", ([&] {
        rbf_kernel_combined<scalar_t><<<grid_dim, block>>>(
            x.data_ptr<scalar_t>(),
            grid.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            grad_x.data_ptr<float>(),
            static_cast<scalar_t>(inv_denom),
            num_elements,
            N,
            G
        );
    }));

    return grad_x;
}




extern "C" __global__ void rbf_kernel_variable_weighted(
    const float* __restrict__ x,
    const float* __restrict__ grid,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const float denominator,
    const int num_elements,  // Number of elements in the "non-last" dimensions of x
    const int N,             // Size of the last dimension of x
    const int G              // Size of grid
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;     // Index in the last dimension of x
    int grid_idx = blockIdx.y * blockDim.y + threadIdx.y; // Index in the grid
    int batch_idx = blockIdx.z;                          // Index in the "non-last" dimensions of x

    if (idx < N && grid_idx < G) {
        // Compute offsets for input, grid, weight, and output tensors
        int input_offset = batch_idx * N + idx;
        int output_offset = batch_idx * (N * G) + idx * G + grid_idx;
        int weight_offset = idx * G + grid_idx;

        // Compute the RBF value
        float diff = (x[input_offset] - grid[grid_idx]) / denominator;
        float rbf_value = expf(-diff * diff);

        // Apply the weight
        output[output_offset] = rbf_value * weights[weight_offset];
    }
}


torch::Tensor radial_basis_function_cuda_weighted(
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor weights,
    float denominator
) {
    // Ensure tensors are contiguous and on CUDA
    TORCH_CHECK(x.is_contiguous() && grid.is_contiguous() && weights.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(x.is_cuda() && grid.is_cuda() && weights.is_cuda(), "Tensors must be on CUDA");

    int num_dims = x.dim();
    TORCH_CHECK(num_dims >= 1, "Input tensor must have at least one dimension");

    // Get sizes
    int N = x.size(-1);  // Last dimension size
    int G = grid.size(0);  // Grid size
    int num_elements = x.numel() / N;  // Product of "non-last" dimensions

    // Prepare output tensor
    std::vector<int64_t> output_shape = x.sizes().vec();
    output_shape.push_back(G);  // Append grid size to output shape
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor output = torch::zeros(output_shape, options);

    // Configure CUDA kernel
    dim3 block(16, 16, 1);
    dim3 grid_dim((N + block.x - 1) / block.x, (G + block.y - 1) / block.y, num_elements);

    // Launch kernel
    rbf_kernel_variable_weighted<<<grid_dim, block>>>(
        x.data_ptr<float>(),
        grid.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        denominator,
        num_elements,
        N,
        G
    );

    return output;
}


extern "C" __global__ void rbf_kernel_variable_weighted_back(
    const float* __restrict__ x,
    const float* __restrict__ grid,
    const float* __restrict__ weights,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_x,
    float* __restrict__ grad_weights,
    const float inv_denom,
    const int num_elements,  // Number of elements in the "non-last" dimensions of x
    const int N,             // Size of the last dimension of x
    const int G              // Size of the grid
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;      // Index in the last dimension of x
    int grid_idx = blockIdx.y * blockDim.y + threadIdx.y; // Index in the grid
    int batch_idx = blockIdx.z;                           // Index in the "non-last" dimensions of x

    if (idx < N && grid_idx < G) {
        // Compute offsets for input, grid, weight, and output tensors
        int input_offset = batch_idx * N + idx;
        int output_offset = batch_idx * (N * G) + idx * G + grid_idx;
        int weight_offset = idx * G + grid_idx;

        // Compute the RBF value and its gradient
        float diff = inv_denom * (x[input_offset] - grid[grid_idx]);
        float rbf_value = expf(-diff * diff);
        float gradient_comp = -2.0 * diff * inv_denom * rbf_value;

        // Compute and accumulate grad_x
        atomicAdd(&grad_x[input_offset], gradient_comp * weights[weight_offset] * grad_output[output_offset]);

        // Compute and accumulate grad_weights
        atomicAdd(&grad_weights[weight_offset], rbf_value * grad_output[output_offset]);
    }
}


std::vector<torch::Tensor>  radial_basis_function_cuda_weighted_back(
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor weights,
    float denominator,
    torch::Tensor grad_output,
    int x_grid_dim,
    int y_grid_dim,
    int z_grid_dim
) {
    // Ensure tensors are contiguous and on CUDA
    TORCH_CHECK(x.is_contiguous() && grid.is_contiguous() && weights.is_contiguous() && grad_output.is_contiguous(),
                "Tensors must be contiguous");
    TORCH_CHECK(x.is_cuda() && grid.is_cuda() && weights.is_cuda() && grad_output.is_cuda(), "Tensors must be on CUDA");

    int num_dims = x.dim();
    TORCH_CHECK(num_dims >= 1, "Input tensor must have at least one dimension");

    // Get sizes
    int N = x.size(-1);  // Last dimension size
    int G = grid.size(0);  // Grid size
    int num_elements = x.numel() / N;  // Product of "non-last" dimensions

    // Prepare output tensors
    std::vector<int64_t> output_shape = x.sizes().vec();
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor grad_x = torch::zeros(output_shape, options);
    torch::Tensor grad_weights = torch::zeros({N, G}, options);

    // Calculate inv_denom
    float inv_denom = 1.0f / denominator;

    // Configure CUDA kernel
    dim3 block(x_grid_dim, y_grid_dim, z_grid_dim);
    dim3 grid_dim((N + block.x - 1) / block.x, (G + block.y - 1) / block.y, num_elements);

    // Launch kernel
    rbf_kernel_variable_weighted_back<<<grid_dim, block>>>(
        x.data_ptr<float>(),
        grid.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        inv_denom,
        num_elements,
        N,
        G
    );

    return {grad_x, grad_weights};
}





extern "C" __global__ void rbf_kernel_variable_weighted_back_optim(
    const float* __restrict__ x,
    const float* __restrict__ grid,
    const float* __restrict__ weights,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_x,
    float* __restrict__ grad_weights,
    const float inv_denom,
    const int num_elements,
    const int N,
    const int G
) {
    // Calculate thread indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension
    int batch_idx = blockIdx.y;  // num_elements dimension

    // Shared memory for grid
    __shared__ float shared_grid[8];  // G is small (always 8)

    // Load grid into shared memory
    if (threadIdx.x < G) {
        shared_grid[threadIdx.x] = grid[threadIdx.x];
    }
    __syncthreads();

    // Thread-local accumulation for grad_x
    if (idx < N && batch_idx < num_elements) {
        int input_offset = batch_idx * N + idx;
        float x_val = x[input_offset];
        float grad_x_val = 0.0f;

        for (int grid_idx = 0; grid_idx < G; ++grid_idx) {
            int output_offset = batch_idx * (N * G) + idx * G + grid_idx;
            int weight_offset = idx * G + grid_idx;

            float diff = inv_denom * (x_val - shared_grid[grid_idx]);
            float rbf_value = expf(-diff * diff);
            float gradient_comp = -2.0f * diff * inv_denom * rbf_value;

            // Accumulate gradient contributions
            grad_x_val += gradient_comp * weights[weight_offset] * grad_output[output_offset];
            atomicAdd(&grad_weights[weight_offset], rbf_value * grad_output[output_offset]);
        }

        // Write accumulated grad_x
        grad_x[input_offset] += grad_x_val;
    }
}




std::vector<torch::Tensor> radial_basis_function_cuda_weighted_back_optim(
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor weights,
    float denominator,
    torch::Tensor grad_output
) {
    // Ensure tensors are contiguous and on CUDA
    TORCH_CHECK(x.is_contiguous() && grid.is_contiguous() && weights.is_contiguous() && grad_output.is_contiguous(),
                "Tensors must be contiguous");
    TORCH_CHECK(x.is_cuda() && grid.is_cuda() && weights.is_cuda() && grad_output.is_cuda(), "Tensors must be on CUDA");

    // Get tensor dimensions
    int N = x.size(-1);  // Last dimension size
    int G = grid.size(0);  // Grid size
    int num_elements = x.numel() / N;  // Product of "non-last" dimensions

    // Prepare output tensors
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor grad_x = torch::zeros_like(x, options);
    torch::Tensor grad_weights = torch::zeros({N, G}, options);

    // Calculate inverse denominator
    float inv_denom = 1.0f / denominator;

    // Kernel configuration
    dim3 block(128, 1, 1);  // 256 threads per block
    dim3 grid_new((N + block.x - 1) / block.x, num_elements, 1);

    // Launch kernel
    rbf_kernel_variable_weighted_back_optim<<<grid_new, block>>>(
        x.data_ptr<float>(),
        grid.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        inv_denom,
        num_elements,
        N,
        G
    );

    return {grad_x, grad_weights};
}


extern "C" __global__ void rbf_kernel_combined_optimized(
    const float* __restrict__ x,
    const float* __restrict__ grid,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_x,
    const float inv_denom,
    const int num_elements,
    const int N,
    const int G
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Index in the last dimension of x
    int grid_idx = blockIdx.y * blockDim.y + threadIdx.y; // Index in the grid
    int batch_idx = blockIdx.z; // Index in the "non-last" dimensions of x

    if (idx < N && grid_idx < G) {
        // Compute offset for the input and output tensors
        int input_offset = batch_idx * N + idx;
        int output_offset = batch_idx * (N * G) + idx * G + grid_idx;

        float diff = inv_denom * (x[input_offset] - grid[grid_idx]);
        float gradient_comp = -2.0 * diff * inv_denom * expf(-diff * diff);

        // Compute the gradient contribution
        float local_grad = gradient_comp * grad_output[output_offset];

        // Use shared memory for block-level reduction
        __shared__ float shared_grad[256];  // Example shared memory size (adjust based on block size)
        int local_idx = threadIdx.x;

        shared_grad[local_idx] = local_grad;
        __syncthreads();

        // Perform block reduction
        if (local_idx == 0) {
            float sum = 0.0;
            for (int i = 0; i < blockDim.x; ++i) {
                sum += shared_grad[i];
            }
            // Atomic write the block result to global memory
            atomicAdd(&grad_x[batch_idx * N + idx], sum);
        }
    }
}


// Launch the kernel
torch::Tensor radial_basis_function_cuda_combined_optimized(
    torch::Tensor x,
    torch::Tensor grid,
    float denominator,
    torch::Tensor grad_output
) {
    // Ensure tensors are contiguous and on CUDA
    TORCH_CHECK(x.is_contiguous() && grid.is_contiguous() && grad_output.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(x.is_cuda() && grid.is_cuda() && grad_output.is_cuda(), "Tensors must be on CUDA");

    int num_dims = x.dim();
    TORCH_CHECK(num_dims >= 1, "Input tensor must have at least one dimension");

    // Get sizes
    int N = x.size(-1);  // Last dimension size
    int G = grid.size(0);  // Grid size
    int num_elements = x.numel() / N;  // Product of "non-last" dimensions

    std::vector<int64_t> output_shape = x.sizes().vec();
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor grad_x = torch::zeros(output_shape, options);
    
    // Calculate inv_denom
    float inv_denom = 1.0f / denominator;

    // Configure CUDA kernel
    dim3 block(16, 16, 1); // Adjust block size for better occupancy (based on A100 specs)
    dim3 grid_dim((N + block.x - 1) / block.x, (G + block.y - 1) / block.y, num_elements);

    // Launch kernel
    rbf_kernel_combined_optimized<<<grid_dim, block>>>(
        x.data_ptr<float>(),
        grid.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        inv_denom,
        num_elements,
        N,
        G
    );

    return grad_x;
}

#include <torch/extension.h>
#include <vector>


// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("radial_basis_function_cuda", &radial_basis_function_cuda, "RBF CUDA");
    m.def("radial_basis_function_cuda_back", &radial_basis_function_cuda_back, "RBF CUDA BACK");
    m.def("launch_rbf_grad_x_kernel", &launch_rbf_grad_x_kernel, "RBF Grad X Kernel (CUDA)");
    m.def("radial_basis_function_cuda_combined", &radial_basis_function_cuda_combined, "Combined Backward");
    m.def("radial_basis_function_cuda_weighted", &radial_basis_function_cuda_weighted, "Weighted Forward RBF");
    m.def("radial_basis_function_cuda_weighted_back", &radial_basis_function_cuda_weighted_back, "Weighted Backward RBF");
m.def("radial_basis_function_cuda_combined_optimized", &radial_basis_function_cuda_combined_optimized, "Weighted Backward RBF");
m.def("radial_basis_function_cuda_weighted_back_optim", &radial_basis_function_cuda_weighted_back_optim, "Weighted Backward RBF   xxxx");

}