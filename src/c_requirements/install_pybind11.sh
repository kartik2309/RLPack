#!/bin/zsh

# pybind11 Installation.
CPUS=$(getconf _NPROCESSORS_ONLN)
echo "Cloning git repository"
git clone https://github.com/pybind/pybind11.git
cd pybind11 || exit
echo "This may take a while ..."
cmake -S . -B build -DDOWNLOAD_CATCH=ON -DCMAKE_CXX_FLAGS="-Wall -std=c++17 -lstdc++fs"
cmake --build build -j "${CPUS}"
cmake --install build
