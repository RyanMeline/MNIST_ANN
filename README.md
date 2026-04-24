# Ground up ANN for MNIST number set

Initial Setup
---
In the root directory:<br>
Download Eigen git submodule:<br>
```
git submodule update --init
```
Initial build file configuration:
```
cmake -S . -B build
```
---

Build project:
---
From the root directory, run:
```
cmake --build build
```
Or from /build/ run:
```
cmake --build .
```

The binary will be specified in the output message of CMake<br>
Default for me was /build/Debug/

---