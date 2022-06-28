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
# ----------------------------------- Package Dependencies ----------------------------------- #
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(PyBind11 REQUIRED)
find_package(OpenMP REQUIRED)
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
endif ()

find_package(Boost COMPONENTS program_options log log_setup mpi serialization REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DBOOST_LOG_DYN_LINK}")

# ----------------------------------- Add Modules and Link Libraries ----------------------------------- #

pybind11_add_module(
        RLPack

        bindings/pybind11_helpers/Dqn/GetDqnAgent.h
        bindings/pybind11_helpers/Dqn/GetDqnAgent.tpp
        bindings/pybind11_helpers/dqnBinder.cpp

        src/Dqn/Agent.h
        src/Dqn/Agent.tpp
        src/Dqn/DqnImpl.hpp

        src/Dqn/Dqn1d/Dqn1d.tpp
        src/Dqn/Dqn1d/Dqn1d.h

        utils/Metal/MetalComputeWrapperImpl.cpp
        utils/Metal/MetalComputeWrapperImpl.h
)

IF (APPLE)

    FILE(GLOB METAL_FILES "utils/Metal/MetalFiles/*.metal")
    FOREACH (METAL_FILE ${METAL_FILES})
        add_custom_command(
                TARGET RLPack

                COMMAND
                xcrun
                -sdk macosx metal
                -c "utils/Metal/MetalFiles/${METAL_FILE}.metal"
                -o "${METAL_BUILD_DIR}/${METAL_FILE}.air"

                COMMAND
                xcrun
                -sdk macosx metallib "${METAL_BUILD_DIR}/${METAL_FILE}.air"
                -o "${METAL_BUILD_DIR}/${METAL_FILE}.metallib"
        )
    ENDFOREACH ()
    target_include_directories(RLPack SYSTEM PRIVATE ${METAL_CPP_DIR})
    target_link_libraries(RLPack PRIVATE ${APPLE_FWK_FOUNDATION}
            ${APPLE_FWK_QUARTZ_CORE} ${APPLE_FWK_METAL})

ENDIF ()

add_dependencies(RLPack pybind11::headers)

target_link_libraries(RLPack PRIVATE Python::Python)
target_link_libraries(RLPack PRIVATE "${OpenCV_LIBRARIES}")
target_link_libraries(RLPack PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(RLPack PRIVATE ${Boost_LOG_LIBRARY})
target_link_libraries(RLPack PRIVATE ${Boost_LOG_SETUP_LIBRARY})
target_link_libraries(RLPack PRIVATE ${Boost_LIBRARIES})
target_link_libraries(RLPack PRIVATE OpenMP::OpenMP_CXX)

# ----------------------------------- Installation ----------------------------------- #
install(
        TARGETS RLPack
        EXPORT RLPack
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


IF (CALL_FROM_SETUP_PY)
    set(RLPack_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
ELSE ()
    set(RLPack_INSTALL_PREFIX ${Python3_SITELIB})
ENDIF ()

set_target_properties(
        RLPack PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN TRUE
)