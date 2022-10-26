cmake_minimum_required(VERSION 3.22)
project(RLPack)
# ----------------------------------- Includes for CMAKE ----------------------------------- #

add_subdirectory(binaries/memory)
# ----------------------------- Set Global Variables for CMAKE ----------------------------- #
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(Python_VIRTUALENV FIRST)

