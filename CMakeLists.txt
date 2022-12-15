cmake_minimum_required(VERSION 3.22)
project(RLPack CXX)
# ----------------------------------- Includes for CMAKE ----------------------------------- #
add_subdirectory(binaries/memory)
add_subdirectory(binaries/grad_accumulator)
# ----------------------------- Set Global Variables for CMAKE ----------------------------- #
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(Python_VIRTUALENV FIRST)

