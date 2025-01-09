from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rbf_cuda',
    ext_modules=[
        CUDAExtension(
            'rbf_cuda',
            ['rbf_cuda.cu'],
            extra_compile_args={'nvcc': ['-lineinfo', '-DTORCH_USE_CUDA_DSA']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
