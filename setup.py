# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from setuptools import setup, find_packages
import os


SETUP_PATH = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_PATH = os.path.join(SETUP_PATH, "requirements.txt")

def read_requirements(file_path):
    requirements = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            requirements.append(line)
    return requirements

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = read_requirements(REQUIREMENTS_PATH)

setup(
    name='RealMirror',
    version='0.1.0',
    packages=find_packages(),
    author='RealMirror Team',
    maintainer='RealMirror Team',
    description='A Comprehensive, Open-Source Vision-Language-Action Platform for Embodied AI',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://terminators2025.github.io/RealMirror.github.io',
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.0.0",
    ],
    python_requires='>=3.11',
)
