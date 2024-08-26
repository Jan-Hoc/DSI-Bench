#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# check if conda is available
if ! command -v conda &> /dev/null
then
    echo "conda could not be found"
    exit 1
fi

if test -f $SCRIPT_DIR/environment.yml; then
    conda env create -y -n ${1:-benchmark} -f $SCRIPT_DIR/environment.yml
    eval "$(conda shell.bash hook)" &> /dev/null
    conda activate ${1:-benchmark}
    pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-tf-plugin-cuda120
    pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7
    cd $SCRIPT_DIR/..
    poetry install
    conda deactivate
    conda clean -aqy
else
    echo "environment.yml could not be found"
    exit 1
fi