#!/bin/zsh

# kineto Installation.
CPUS=$(getconf _NPROCESSORS_ONLN)
echo "Cloning git repository"
cd "$1" || exit
git clone --recursive https://github.com/pytorch/kineto.git
cd "$1"/kineto/libkineto || exit
echo "This may take a while ..."
cmake -S . -B build -DCMAKE_INSTALL_LIBDIR="$1" -DCUDA_SOURCE_DIR="$2"
cmake --build build -j "${CPUS}"
