import setuptools
import cmake_build_extension

import inspect
import os
import sys
from pathlib import Path

__version__ = '0.0.1'

init_py = inspect.cleandoc(
    """
    from .lib import RLPack
    """
)

CIBW_CMAKE_OPTIONS = []
if "CIBUILDWHEEL" in os.environ and os.environ["CIBUILDWHEEL"] == "1":
    if sys.platform == "linux":
        CIBW_CMAKE_OPTIONS += ["-DCMAKE_INSTALL_LIBDIR=lib"]

setuptools.setup(
    name="RLPack",
    version=__version__,
    author="Kartik Rajeshwaran",
    author_email="kartik.rajeshwaran@gmail.com",
    description="Implementation of RL Algorithms",
    long_description="Implementation of RL Algorithms with C++ backend and made available to Gym Frontend "
                     "with Python Bindings",
    long_description_content_type="text/markdown",
    url="https://github.com/kartik2309/RLPack",
    packages=['rlpack'],
    package_dir={'rlpack': 'bindings/rlpack'},
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": ["RLPack_entry = RLPack.bin.__main__:main"],
    },
    ext_modules=[
        cmake_build_extension.CMakeExtension(
            name="BuildAndInstall",
            install_prefix="rlpack",
            cmake_depends_on=["pybind11"],
            expose_binaries=["bin/rlpack"],
            write_top_level_init=init_py,
            source_dir=str(Path(__file__).parent.absolute()),
            cmake_configure_options=[
                                        f"-DPython3_ROOT_DIR={Path(sys.prefix)}",
                                        "-DBUILD_SHARED_LIBS:BOOL=TRUE",
                                        f"-DMETAL_BUILD_DIR={os.environ.get('METAL_BUILD_DIR', None)}",
                                        f"-DMETAL_CPP_DIR={os.environ.get('METAL_CPP_DIR', None)}",
                                        f"-DCALL_FROM_SETUP_PY=TRUE",
                                    ] + CIBW_CMAKE_OPTIONS,
        ),
    ],
    cmdclass={
        "build_ext": cmake_build_extension.BuildExtension,
        "sdist": cmake_build_extension.GitSdistFolder
    }
)
