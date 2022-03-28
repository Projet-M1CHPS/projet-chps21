# Breast cancer detection through machine learning

This project is part of the first year M1CHPS (High Performance Computing Degree) curriculum.

The end goal is to implement in C++ a tool capable of cancer detection using various machine learning (ML) algorithms.
Those algorithms will be implemented natively in C++, and no ML external libraries will be used.

In the end, the user will have the choice between various training methods, and will be able to interact with the tool
through a Python api.

# Requirements

Manual install :

* cmake >= 3.17
* gcovr
* doxygen (Optionnal)

For versions < 1.0.0

* blas

For versions >= 1.0.0

* OpenCL

Imported by the project :

* GoogleTest (No need to install it, it is imported by the project)
* TSCL (No need to install it, it is imported by the project)

# Building the project

To build the project, run the following command in the root directory of the project:

```sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release  ..

# Build the project
make -j 
```

We also recommend you run the tests to ensure everything is working properly

```sh
make test
```

# Usage

If everything worked so far, the project is now runnable. Versions <= 1.0 do not provide a python interface, but instead
an executable for loading a dataset and training a model.

```sh
# Build the project if not done already
cd build
./bin/main <path todataset> <output path>
```

Raw data will be outputted to <output path>/(eval or optimize).

# About this project

## First part

During the first semester, we will focus on exploring and implementing various machine learning algorithms. The main
objective is to have a functional prototype, and a good understanding of the various challenges of this subject. We will
also profile the application performance for further work.

## Second part

This second part will focus on the project optimization and new QoL features. No further training algorithm should be
implemented, except if they provide a significant improvement in the results or performance.

For this, we will use OpenCL and OpenMP to optimize the application in a portable way.

## Features

* A simple and intuitive API (Work in progress)
* Powerful machine learning algorithms, designed with maximum accuracy and performance in mind
* A plethora of training algorithms to choose from
* A clear report of the results, to provide the end-user feedback on his work

# Credits