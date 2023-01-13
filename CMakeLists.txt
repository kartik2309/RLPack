cmake_minimum_required(VERSION 3.22)
project(RLPack CXX)
# ----------------------------- Set Global Variables for CMAKE ----------------------------- #
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(Python_VIRTUALENV FIRST)
# ----------------------------- Preliminary check for packages ---------------------------- #
if (NOT CUDAToolkit_FOUND)
    message("-- CUDA was not found.")
else ()
    add_compile_definitions(__CUDA_AVAILABLE__)
    message("-- CUDA was found.")
    enable_language(CUDA)
endif()
if (NOT EXISTS "${Torch_PACKAGE_DIR}/kineto/")
    execute_process(
            COMMAND sh "${CMAKE_SOURCE_DIR}/install_cpp_requirements.sh"
            "${Torch_PACKAGE_DIR}" "${CUDAToolkit_TARGET_DIR}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/"
            COMMAND_ECHO STDOUT
    )
endif ()
set(kineto_LIBRARY "${Torch_PACKAGE_DIR}/kineto/libkineto/build/libkineto.a")
# ----------------------------------- Includes for CMAKE ----------------------------------- #
add_subdirectory(binaries/replay_buffer)
add_subdirectory(binaries/grad_accumulator)
add_subdirectory(binaries/rollout_buffer)
add_subdirectory(binaries/stl_bindings)
