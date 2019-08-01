#!/usr/bin/env python3
try:
        from setuptools import setup, find_packages
except ImportError:
        from distutils.core import setup, find_packages

setup(
    name='SingleAnalyst',
    version='0.7',
    description='scRNA-seq analysis platform, featured with reference index build and search',
    author="yuyifei",
    author_email="yuyifei@tongji.edu.cn",

    install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'faiss',
        'matplotlib',
        'seaborn',
        'pandas',
        'feather',
        'h5py'
    ]
)
