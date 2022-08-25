cmake_minimum_required(VERSION 3.22)
project(C_Memory)
# ----------------------------------- Includes for CMAKE ----------------------------------- #
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ----------------------------- Set Global Variables for CMAKE ----------------------------- #
set(CMAKE_CXX_STANDARD 17)
set(Python_VIRTUALENV FIRST)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif ()
# ----------------------------------- Package Dependencies ---------------------------------- #
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    execute_process(
            COMMAND sh ${CMAKE_SOURCE_DIR}/src/c_requirements/install_pybind11.sh
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/"
            COMMAND_ECHO STDOUT
    )
    if("${CMAKE_PREFIX_PATH}" STREQUAL "")
        set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/pybind11")
    else()
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CMAKE_SOURCE_DIR}/pybind11")
    endif()
    find_package(pybind11 REQUIRED)
endif()

find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(CUDAToolkit QUIET)
if(NOT CUDA_FOUND)
    set(CUDA_SOURCE_DIR "")
endif()
if(NOT EXISTS "${Torch_PACKAGE_DIR}/kineto/")
    execute_process(
            COMMAND sh "${CMAKE_SOURCE_DIR}/src/c_requirements/install_kineto.sh" "${Torch_PACKAGE_DIR}" "${CUDAToolkit_TARGET_DIR}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/"
            COMMAND_ECHO STDOUT
    )
endif()
set(kineto_LIBRARY "${Torch_PACKAGE_DIR}/kineto/libkineto/build/libkineto.a")
find_package(Torch QUIET)
if(NOT Torch_FOUND)
    message(FATAL "Cannot find a valid PyTorch installation!")
endif()
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(TORCH_USE_RTLD_GLOBAL YES)

pybind11_add_module(
        C_Memory
        C_Memory.cpp
        C_Memory.h
        binding.cpp
)

add_dependencies(C_Memory pybind11::headers)
target_link_libraries(C_Memory PRIVATE Python::Python)
target_link_libraries(C_Memory PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(C_Memory PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(C_Memory PRIVATE "${TORCH_PYTHON_LIBRARY}")
# --------------------------------------- Installation --------------------------------------- #
install(
        TARGETS C_Memory
        EXPORT C_Memory
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


if (CALL_FROM_SETUP_PY)
    set(RLPack_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
else ()
    set(RLPack_INSTALL_PREFIX ${Python3_SITELIB})
endif ()

set_target_properties(
        C_Memory PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN TRUE
)
