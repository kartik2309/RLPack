cmake_minimum_required(VERSION 3.8)
project(RLPack)

# ----------------------------------- Includes for CMAKE ----------------------------------- #

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ----------------------------------- Set Global Variables for CMAKE ----------------------------------- #

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
set_property(GLOBAL PROPERTY USE_FOLDERS YES)
set(METAL_CPP_DIR "${CMAKE_SOURCE_DIR}/utils/Metal/metal-cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
# ----------------------------------- Package Dependencies ----------------------------------- #

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(PyBind11 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif ()

if (APPLE)
    find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
    find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
    find_library(APPLE_FWK_METAL Metal REQUIRED)
    find_library(APPLE_FWK_METAL_PERFORMANCE_SHADERS MetalPerformanceShaders REQUIRED)
    find_library(APPLE_FWK_METAL_PERFORMANCE_SHADERSGRAPH MetalPerformanceShadersGraph REQUIRED)
endif ()

find_package(Boost COMPONENTS program_options log log_setup mpi serialization random REQUIRED)

# ----------------------------------- Add Modules and Link Libraries ----------------------------------- #

pybind11_add_module(
        RLPack

        bindings/pybind11_helpers/Dqn/GetDqnAgent.h
        bindings/pybind11_helpers/Dqn/GetDqnAgent.tpp
        bindings/pybind11_helpers/dqnBinder.cpp

        src/Dqn/Agent.h
        src/Dqn/Agent.tpp
        src/Dqn/DqnBase.hpp

        src/Dqn/Dcqn1d/Dcqn1d.h
        src/Dqn/Dcqn1d/Dcqn1d.tpp

        src/Dqn/Dlqn1d/Dlqn1d.h
        src/Dqn/Dlqn1d/Dlqn1d.tpp

        src/utils/utils.hpp
        src/utils/Normalization/Normalization.h
        src/utils/Normalization/Normalization.cpp
        src/utils/Memory/Memory.h
        src/utils/Memory/Memory.cpp

)


if (APPLE)
    target_include_directories(RLPack SYSTEM PRIVATE ${METAL_CPP_DIR})
    target_link_libraries(
            RLPack PRIVATE
            ${APPLE_FWK_FOUNDATION}
            ${APPLE_FWK_QUARTZ_CORE}
            ${APPLE_FWK_METAL}
            ${APPLE_FWK_METAL_PERFORMANCE_SHADERS}
            ${APPLE_FWK_METAL_PERFORMANCE_SHADERSGRAPH}
    )
endif ()

add_dependencies(RLPack pybind11::headers)

target_link_libraries(RLPack PRIVATE Python::Python)
target_link_libraries(RLPack PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(RLPack PRIVATE MPI::MPI_CXX)
target_link_libraries(RLPack PRIVATE "${OpenCV_LIBRARIES}")
target_link_libraries(RLPack PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(RLPack PRIVATE ${Boost_LIBRARIES})

# ----------------------------------- Installation ----------------------------------- #
install(
        TARGETS RLPack
        EXPORT RLPack
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
        RLPack PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN TRUE
)