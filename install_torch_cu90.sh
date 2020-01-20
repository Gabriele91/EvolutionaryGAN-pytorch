#!/bin/bash

python3 -m pip uninstall torch
python3 -m pip install \
https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-linux_x86_64.whl \
https://download.pytorch.org/whl/cu90/torchvision-0.3.0-cp36-cp36m-manylinux1_x86_64.whl \
https://download.pytorch.org/whl/torchaudio-0.3.0-cp36-cp36m-manylinux1_x86_64.whl
python3 -m uninstall Pillow
python3 -m install Pillow==6.0.0
