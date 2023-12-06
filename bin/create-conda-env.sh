#!/bin/bash --login

# create a new conda environment
PROJECT_DIR="$PWD"
ENV_PREFIX="$PROJECT_DIR/env"
conda create --prefix $ENV_PREFIX python=3.9