#!/bin/bash
# make sure your python virtual env is already sourced and active.
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
export FORCE_CMAKE=1
pip install --upgrade llama-cpp-python[server]

