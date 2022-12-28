cmake_minimum_required(VERSION 3.22)
project(C_GradAccumulator CXX)
# ----------------------------------- Includes for CMAKE ----------------------------------- #
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ----------------------------- Set Global Variables for CMAKE ----------------------------- #
set(CMAKE_CXX_STANDARD 17)
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    message("${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "Old Version of GCC was found. Please install GCC 9 or above")
endif ()
set(Python_VIRTUALENV FIRST)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif ()

# ----------------------------------- Package Dependencies ---------------------------------- #

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
find_package(Torch QUIET)
if (NOT Torch_FOUND)
    message(FATAL_ERROR "Cannot find a valid PyTorch installation!")
endif ()
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(TORCH_USE_RTLD_GLOBAL YES)

# ----------------------------------------- Linking ---------------------------------------- #

pybind11_add_module(
        C_GradAccumulator
        C_GradAccumulator.cpp
        C_GradAccumulator.h
        Binding.cpp
)

add_dependencies(C_GradAccumulator pybind11::headers)
target_link_libraries(C_GradAccumulator PRIVATE Python::Python)
target_link_libraries(C_GradAccumulator PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(C_GradAccumulator PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(C_GradAccumulator PRIVATE "${TORCH_PYTHON_LIBRARY}")
# --------------------------------------- Installation --------------------------------------- #
install(
        TARGETS C_GradAccumulator
        EXPORT C_GradAccumulator
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
        C_GradAccumulator PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN TRUE
)
