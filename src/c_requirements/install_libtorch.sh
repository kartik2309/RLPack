#!/bin/zsh

echo "Attempting to download built libtorch"
if [ "$1" -eq 1 ]; then
  curl https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu102.zip -o libtorch.zip
elif [ "$1" -eq 2 ]; then
  curl https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.12.1.zip -o libtorch.zip
fi
unzip libtorch.zip
