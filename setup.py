from os import path

from setuptools import find_packages, setup

working_directory = path.abspath(path.dirname(__file__))

with open(path.join(working_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="agdiff",
    version="0.0.1",
    description="AGDIFF_chi: stereochemistry-aware all-atom diffusion for molecular 3D structure generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dizhou Wu and Yike Zou",
    url="https://github.com/yikezougroup/AGDIFF_chi",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["agdiff", "diffusion models", "generative models", "conformer", "stereochemistry"],
)
