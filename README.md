# Ground up ANN for MNIST number set

## Initial Setup

Egien Files downloaded as a git submodule<br>
When cloning the repo, use:
```
git submodule update --init
```

Initial project configuration:
```
cmake -S . -B build
```

## Build project:

From the root directory, run:
```
cmake --build build
```

The binary will be specified in the output message of CMake<br>
Default for me was /build/Debug/
