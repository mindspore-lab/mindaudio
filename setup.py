#!/usr/bin/env python

from setuptools import find_packages, setup

exec(open("mindaudio/version.py").read())


def read_requirements(fps):
    reqs = []
    for fp in fps:
        with open(fp) as f:
            reqs.extend(f.readlines())

    return reqs


setup(
    name="mindaudio",
    author="MindSpore Lab",
    author_email="mindspore-lab@example.com",
    url="https://github.com/mindspore-lab/mindaudio",
    project_urls={
        "Sources": "https://github.com/mindspore-lab/mindaudio",
        "Issue Tracker": "https://github.com/mindspore-lab/mindaudio/issues",
    },
    description="A toolbox of audio models and algorithms based on MindSpore.",
    license="Apache Software License 2.0",
    include_package_data=True,
    packages=find_packages(include=["mindaudio", "mindaudio.*"]),
    python_requires=">=3.7",
    install_requires=[
        "Pillow==8.1.0",
        "numpy",
        "pyyaml",
        "tqdm",
        "scipy",
        "sentencepiece",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    tests_require=[
        "pytest",
    ],
    version=__version__,
    zip_safe=False,
)
