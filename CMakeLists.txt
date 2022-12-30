cmake_minimum_required(VERSION 3.22)
project(RLPack CXX)
# ----------------------------------- Includes for CMAKE ----------------------------------- #
add_subdirectory(binaries/replay_buffer)
add_subdirectory(binaries/grad_accumulator)
add_subdirectory(binaries/rollout_buffer)
# ----------------------------- Set Global Variables for CMAKE ----------------------------- #
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(Python_VIRTUALENV FIRST)

