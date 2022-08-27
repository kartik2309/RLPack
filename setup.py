import glob
import os
import shutil
import sys
from pathlib import Path
from site import getsitepackages

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from torch.utils import cmake_prefix_path

__version__ = "0.0.1"


class CMakeExtension(Extension):
    def __init__(self, shell_script):
        super().__init__(shell_script, sources=[])


class BuildExternal(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = Path().absolute()

        build_temp = Path(self.build_temp)
        build_lib = Path(self.build_lib)
        build_temp.mkdir(parents=True, exist_ok=True)
        external_dir = Path(self.get_ext_fullpath(ext.name))
        external_dir.mkdir(parents=True, exist_ok=True)
        lib_directories = ["src/memory"]

        config = "Debug" if self.debug else "Release"
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE:STRING={config}",
            f"-DPython_ROOT_DIR={Path(sys.prefix)}",
            "-DBUILD_SHARED_LIBS:BOOL=TRUE",
            "-DCALL_FROM_SETUP_PY:BOOL=TRUE",
            f"-DTorch_DIR={cmake_prefix_path}/Torch",
            f"-DTorch_PACKAGE_DIR={getsitepackages()[0]}/torch",
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


setup(
    name="RLPack",
    version=__version__,
    author="Kartik Rajeshwaran",
    author_email="kartik.rajeshwaran@gmail.com",
    description="Implementation of RL Algorithms",
    long_description="Implementation of RL Algorithms with PyTorch and optimization with C++.",
    long_description_content_type="text/markdown",
    url="https://github.com/kartik2309/RLPack",
    packages=[
        "rlpack",
        "rlpack.dqn",
        "rlpack.models",
        "rlpack.environments",
        "rlpack.utils",
        "rlpack.utils.base",
        "rlpack._C",
    ],
    package_data={
        "rlpack/environments/configs": ["rlpack/environments/configs/dlqn1d.yaml"]
    },
    platforms="posix",
    include_package_data=True,
    package_dir={
        "rlpack": "rlpack",
        "rlpack.dqn": "rlpack/dqn",
        "rlpack.models": "rlpack/models",
        "rlpack.environments": "rlpack/environments",
        "rlpack.environments.configs": "rlpack/environments/configs",
        "rlpack.utils": "rlpack/utils",
        "rlpack.utils.base": "rlpack/utils/base",
        "rlpack._C": "rlpack/_C",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX",
    ],
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
