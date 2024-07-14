#!/usr/bin/env python
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved
import setuptools
import torch
import os
import subprocess

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.8"

setuptools.setup(
    name="zero",
    version="0.1.0",
    author="congtou",
    python_requires=">=3.6",
    long_description="ZeroAI Core Module",
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    packages=setuptools.find_namespace_packages(),
)

setuptools.setup(
    name="business",
    version="0.1.0",
    author="congtou",
    python_requires=">=3.6",
    long_description="Insight Face Module",
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    packages=setuptools.find_namespace_packages(),
)