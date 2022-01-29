from distutils.core import setup, Extension

INTERFACE_module = Extension("INTERFACE",
                             sources=["interface.cpp"])

setup(name="INTERFACE",
      version="0.1",
      description="Interface for the project-chps21 neural network.",
      ext_modules=[INTERFACE_module])
