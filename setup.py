#!/usr/bin/env python3
"""
Setup script for SlideCheck library

Install in development mode:
    pip install -e .

Install with dependencies:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="slidecheck",
    version="1.0.0",
    description="A library for weakly-supervised pathology image analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SlideCheck Team",
    url="https://github.com/lingxitong/SlideCheck",
    license="Apache-2.0",
    python_requires=">=3.8",
    packages=find_packages(include=["slidecheck", "slidecheck.*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "h5py>=3.8.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "timm": [
            "timm>=0.9.0",  # For MobileNetV3 and other vision models
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
