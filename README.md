# KataGo-Outer-Open-Gomoku
using KataGo method to train Outer-Open Gomoku.

## Features
* Outer-Open Gomoku Gomoku
* Multi-threading Tree/Root Parallelization with Virtual Loss and LibTorch
* Gomoku, MCTS and Network Infer are written in C++
* SWIG for Python C++ extension

## Args
Edit config.py

## Packages
* Python 3.6+
* PyGame 1.9+
* CUDA 11+
* [PyTorch 1.1+](https://pytorch.org/get-started/locally/)
* [LibTorch 1.3+ (Pre-cxx11 ABI)](https://pytorch.org/get-started/locally/)
* [SWIG 3.0.12+](https://sourceforge.net/projects/swig/files/)
* CMake 3.8+
* MSVC14.0+ / GCC6.0+

## Run
```
# Compile Python extension
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=path/to/libtorch -DPYTHON_EXECUTABLE=path/to/python -DCMAKE_BUILD_TYPE=Release
make -j10

# Run
cd ../test
python learner_test.py train # train model
python learner_test.py play  # play with human

#Ludii
cd test
python server.py
// to open server, and you also need to open the Ludii client
```
