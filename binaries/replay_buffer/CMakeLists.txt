cmake_minimum_required(VERSION 3.22)
project(C_ReplayBuffer CXX)
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
find_package(CUDAToolkit QUIET)
if (NOT CUDAToolkit_FOUND)
    message("-- CUDA was not found.")
    set_source_files_properties(C_ReplayBuffer.cu PROPERTIES LANGUAGE CXX)
    set_source_files_properties(Binding.cu PROPERTIES LANGUAGE CXX)
else ()
    add_compile_definitions(__CUDA_AVAILABLE__)
    message("-- CUDA was found.")
    enable_language(CUDA)

endif ()
if (NOT EXISTS "${Torch_PACKAGE_DIR}/kineto/")
    execute_process(
            COMMAND sh "${CMAKE_SOURCE_DIR}/install_cpp_requirements.sh"
            "${Torch_PACKAGE_DIR}" "${CUDAToolkit_TARGET_DIR}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/"
            COMMAND_ECHO STDOUT
    )
endif ()
set(kineto_LIBRARY "${Torch_PACKAGE_DIR}/kineto/libkineto/build/libkineto.a")
find_package(Torch QUIET)
if (NOT Torch_FOUND)
    message(FATAL_ERROR "Cannot find a valid PyTorch installation!")
endif ()
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(TORCH_USE_RTLD_GLOBAL YES)

# ----------------------------------------- Linking ---------------------------------------- #

pybind11_add_module(
        C_ReplayBuffer
        sumtree_node/SumTreeNode.cpp
        sumtree_node/SumTreeNode.h
        sumtree/SumTree.cpp
        sumtree/SumTree.h
        ${CMAKE_SOURCE_DIR}/binaries/utils/ops/arg_mergesort.cuh
        ${CMAKE_SOURCE_DIR}/binaries/utils/maps.h
        utils/CudaOffload.cuh
        utils/HostOffload.tpp
        utils/Utils.h
        C_ReplayBuffer.cu
        C_ReplayBuffer.cuh

        Binding.cu
)

add_dependencies(C_ReplayBuffer pybind11::headers)
target_link_libraries(C_ReplayBuffer PRIVATE Python::Python)
target_link_libraries(C_ReplayBuffer PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(C_ReplayBuffer PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(C_ReplayBuffer PRIVATE "${TORCH_PYTHON_LIBRARY}")
if (CUDAToolkit_FOUND)
    target_link_libraries(C_ReplayBuffer PRIVATE ${CUDA_LIBRARIES})
    target_link_libraries(C_ReplayBuffer PRIVATE ${CMAKE_CUDA_RUNTIME_LIBRARY})
    target_include_directories(C_ReplayBuffer PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_directories(C_ReplayBuffer PRIVATE ${CUDAToolkit_LIBRARY_DIR})
    target_compile_options(C_ReplayBuffer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler "${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS}" -Xptxas -v,
            --use_fast_math
            >)
else ()
    set_target_properties(C_ReplayBuffer PROPERTIES LINKER_LANGUAGE CXX)
endif ()

# --------------------------------------- Installation --------------------------------------- #
install(
        TARGETS C_ReplayBuffer
        EXPORT C_ReplayBuffer
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
        C_ReplayBuffer PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN TRUE
)
