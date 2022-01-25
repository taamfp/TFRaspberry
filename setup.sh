### Adapted from https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi ###

#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi

# Install Python dependencies
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt