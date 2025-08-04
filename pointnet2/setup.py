# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
_ext_src_root = "_ext_src"

import shutil
# 删除旧编译文件
build_dir = os.path.join(ROOT, "build")
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)

setup(
    name='pointnet2',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext_src',
            sources=[
                *glob.glob(f"{_ext_src_root}/src/*.cpp"),
                *glob.glob(f"{_ext_src_root}/src/*.cu")
            ],
            extra_compile_args={
                "cxx": ["-O2", f"-I{ROOT}/{_ext_src_root}/include"],
                "nvcc": ["-O2", f"-I{ROOT}/{_ext_src_root}/include"],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    package_data={'pointnet2._ext_src': ['*.so']},
    # 添加显式包目录声明
    package_dir={'pointnet2': './', 'pointnet2._ext_src': 'pointnet2/_ext_src'},
)
