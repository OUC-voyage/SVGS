from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='scaffold_gs_pro',
    ext_modules=[
        CUDAExtension('scaffold_gs_pro',
            include_dirs=['/home/ouc/anaconda3/envs/scaffold_gs_pro1/include/opencv4', '/usr/local/cuda-11.7/include', '.'],
            library_dirs=['/home/ouc/anaconda3/envs/scaffold_gs_pro1/lib'],  
            libraries=['opencv_core', 'opencv_imgproc', 'opencv_highgui', 'opencv_imgcodecs'],  
            sources=[
                'PatchMatch.cpp', 
                'Propagation.cu',
                'pro.cpp'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3',
                    '-gencode=arch=compute_86,code=sm_86',
                ]
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
