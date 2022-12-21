import glob
import os
import shutil
import sys
from pathlib import Path
from site import getsitepackages
from typing import Any

from pybind11 import get_cmake_dir
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from torch.utils import cmake_prefix_path

__version__ = "1.0.1"


class CMakeExtension(Extension):
    """
    Helper class to build the CMake files for C++ Backend.
    """

    def __init__(self, shell_script: str) -> None:
        """
        :param: shell_script: str: The shell script to be passed to be run in the Extension.
        """
        super().__init__(shell_script, sources=[])


class BuildExternal(build_ext):
    """
    Helper tool to build external files with custom commands.
    """

    def run(self) -> None:
        """
        Runs the provided method `build_cmake` to build the extension.
        """
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext: Any) -> None:
        """
        Builds the binaries for RLPack backend with C++.
        :param ext: Any: The external extensions to be used to build the cmake.
        """
        cwd = Path().absolute()

        build_temp = Path(self.build_temp)
        build_lib = Path(self.build_lib)
        build_temp.mkdir(parents=True, exist_ok=True)
        external_dir = Path(self.get_ext_fullpath(ext.name))
        external_dir.mkdir(parents=True, exist_ok=True)
        lib_directories = ["binaries/memory", "binaries/grad_accumulator"]

        config = "Debug" if self.debug else "Release"
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE:STRING={config}",
            f"-DPython_ROOT_DIR={Path(sys.prefix)}",
            "-DBUILD_SHARED_LIBS:BOOL=TRUE",
            "-DCALL_FROM_SETUP_PY:BOOL=TRUE",
            f"-DTorch_DIR={cmake_prefix_path}/Torch",
            f"-DTorch_PACKAGE_DIR={getsitepackages()[0]}/torch",
            f"-Dpybind11_DIR={get_cmake_dir()}",
        ]

        build_args = ["--config", config, f"-j {os.cpu_count()}"]
        os.makedirs(os.path.join(str(build_lib), "rlpack", "lib"), exist_ok=True)

        os.chdir(str(build_temp))
        self.spawn(["cmake", str(cwd)] + cmake_args)
        self.spawn(["cmake", "--build", "."] + build_args)
        os.chdir(str(cwd))
        for directory in lib_directories:
            libs = glob.glob(f"{str(build_temp)}/{directory}/*.so")
            for lib in libs:
                print(os.path.join(str(build_lib), "rlpack", "lib"))
                shutil.copy2(lib, os.path.join(str(build_lib), "rlpack", "lib"))


# Call to setup method to build the package.
setup(
    name="RLPack",
    version=__version__,
    author="Kartik Rajeshwaran",
    author_email="kartik.rajeshwaran@gmail.com",
    description="Implementation of RL Algorithms",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/kartik2309/RLPack",
    packages=[
        "rlpack",
        "rlpack.dqn",
        "rlpack.actor_critic",
        "rlpack.models",
        "rlpack.environments",
        "rlpack.utils",
        "rlpack.utils.base",
        "rlpack._C",
    ],
    package_data={
        "license_files": ["LICENSE.md"],
    },
    platforms="posix",
    include_package_data=True,
    package_dir={
        "rlpack": "rlpack",
        "rlpack.dqn": "rlpack/dqn",
        "rlpack.actor_critic": "rlpack/actor_critic",
        "rlpack.models": "rlpack/models",
        "rlpack.environments": "rlpack/environments",
        "rlpack.utils": "rlpack/utils",
        "rlpack.utils.base": "rlpack/utils/base",
        "rlpack._C": "rlpack/_C",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
    ],
    license="MIT",
    license_file="LICENSE.md",
    python_requires=">=3.7",
    scripts=["rlpack/bin/simulator"],
    entry_points={
        "console_scripts": ["rlpack_entry = rlpack.bin.simulator.__main__:main"],
    },
    ext_modules=[CMakeExtension(f"{Path().absolute()}")],
    cmdclass={
        "build_ext": BuildExternal,
    },
)
