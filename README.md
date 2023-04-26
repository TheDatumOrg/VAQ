# Fast Adaptive Similarity Search through Variance‑Aware Quantization

# Requirements
- CMake 3.13 or newer
- g++ 7.5 or newer

# Build
```
# With testing
mkdir build && cd build
cmake .. -DBUILD_TESTING=ON
make

# With debug
cmake .. -D=DCMAKE_BUILD_TYPE=Debug
make

# With optimization level option {O2: full, O3: aggressive (default)}
cmake .. -DOPTIMIZATION_LEVEL=full
make

# Without optimization
cmake .. -DOPTIMIZATION_LEVEL=generic
make
```

# Run
```
# Testing
make test

# Driver (temporary)
./main
```
