# KRess: Breast cancer detection through machine learning

[![Run Test](https://github.com/Thukisdo/projet-chps21/actions/workflows/cmake.yml/badge.svg)](https://github.com/Thukisdo/projet-chps21/actions/workflows/cmake.yml)

This project is done in the context of the Master Calcul Haute Performance et Simulation at the Université de Versailles St-Quentin.

The goal of this project is to implement a tool for breast cancer detection, and optimize it accordingly.

While the first semester focused on the sequential implementation (Versions <= 0.5), the second semester sees us switching to a GPU backend for maximum performance and scalability.

Furthermore, we also this project as an opportunity to explore software architecture on a real scale.

# Build: Dependencies

### DNF (Yum)

```
sudo dnf install ocl-icd-devel \
                 opencl-headers \
                 clblast-devel \
                 boost-devel \
                 blas-devel \
```

You will also need a blas implementation, and an OpenCL runtime (we recommend POCL for starter)
Note that the Cuda toolkit already provides the necessary OpenCL files.

```sh
sudo dnf install pocl \
                 openblas-devel
```

# Build:

```sh
git clone https://github.com/Thukisdo/projet-chps21/ kress-chps21
cd kress-chps21 && mkdir build
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
make -j
```

We advise you to run the tests to ensure everything went well:

```
cd build && make test -j
```

# Usage

```sh
# Build the project if not done already
cd build
./bin/kress <path todataset> <output path>
```

Raw data will be outputted to \<output path\>/.

# Authors
  
  [Benjamin Lozes](https://github.com/byjtew)  \
  [Ugo Battiston](https://github.com/johnkyky)  \
  [Mathys JAM](https://github.com/Thukisdo)

# Thanks to
  [Mohammed-Salah Ibnamar](https://github.com/yaspr), for supervising us during this project \
  Thomas Dufaud, for providing us access to a GPU cluster
