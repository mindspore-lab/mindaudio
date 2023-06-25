#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md", mode="r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", mode="r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

exec(open("mindaudio/version.py").read())

setup(
    name="mindaudio",
    author="MindSpore Lab",
    author_email="mindspore-lab@example.com",
    version=__version__,
    url="https://github.com/mindspore-lab/mindaudio",
    project_urls={
        "Sources": "https://github.com/mindspore-lab/mindaudio",
        "Issue Tracker": "https://github.com/mindspore-lab/mindaudio/issues",
    },
    description="A toolbox of audio models and algorithms based on MindSpore.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache Software License 2.0",
    include_package_data=True,
    packages=find_packages(include=["mindaudio", "mindaudio.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pyyaml",
        "tqdm",
        "scipy",
        "sentencepiece",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha"
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
