# LibMatchstick

LibMatchstick is a lightweight C++ / CUDA neural network library.

It is designed as a small self-contained framework for learning, experimentation, and course projects. The library currently focuses on basic neural network components, especially MLP and CNN workflows, while keeping the API relatively simple and explicit.

---

## Features

- Matrix container
- 3D tensor container
- 4D tensor container
- Basic matrix operations
- Basic tensor operations
- CUDA-backed computation in the implementation
- Common activation functions
  - ReLU
  - Leaky ReLU
  - Sigmoid
  - Tanh
  - Identity
  - Softmax for matrix output
- Common loss functions
  - Mean Squared Error
  - Mean Absolute Error
  - Cross Entropy
- Fully connected layer
- Convolution layer
- MLP network wrapper
- CNN network wrapper
- Parameter save/load interfaces for weights, biases, and kernels

---

## Project Status

LibMatchstick is currently a small experimental neural network library.

The current API is suitable for:

- Simple MLP experiments
- Small CNN experiments
- Classification tasks
- Educational neural network implementation
- CUDA learning and demonstration
- Course project demonstration

It is not intended to replace mature machine learning frameworks such as PyTorch, TensorFlow, or ONNX Runtime.

---

## Repository Layout

Recommended layout:

```text
LibMatchstick/
├── include/
│   ├── matrix.h
│   ├── tensor_3d.h
│   ├── tensor_4d.h
│   ├── activation.h
│   ├── losses.h
│   ├── layer.h
│   └── network.h
├── src/
├── docs/
│   └── API.md
├── LICENSE
├── COPYING
├── COPYING.LESSER
├── CMakeLists.txt
└── README.md
```

---

## Requirements

The exact requirements depend on how the library is built on your system.

Common requirements:

- CMake
- CUDA Toolkit
- A CUDA-capable GPU
- A compiler supported by your CUDA Toolkit

On Linux, typical compilers include:

- GCC
- Clang
- NVCC with a supported host compiler

---

## Build

A common CMake build flow is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The generated library is usually placed somewhere under the `build/` directory, depending on the current `CMakeLists.txt`.

For example, the result may be a shared library such as:

```text
libmatchstick.so
```

on Linux

---

## Using LibMatchstick

To use LibMatchstick in another C++ project, include the public headers and link against the generated library.

Example include usage:

```cpp
#include "matrix.h"
#include "tensor_3d.h"
#include "activation.h"
#include "losses.h"
#include "network.h"
```

If you are using CMake manually, the basic idea is:

```cmake
target_include_directories(your_target PRIVATE /path/to/LibMatchstick/include)
target_link_libraries(your_target PRIVATE /path/to/libmatchstick.so)
```

Adjust the library path and file name according to your platform and build output.

---

## Minimal MLP Example

```cpp
#include "matrix.h"
#include "activation.h"
#include "losses.h"
#include "network.h"

#include <iostream>

using namespace LibMatchstick;
using namespace LibMatchstick::Activation;
using namespace LibMatchstick::Losses;

int main() {
    MLP net(2, 0.01f);

    net.setLayer(0, 2, 4);
    net.setLayer(1, 4, 2);

    net.setLayerActivation(0, sigmoid, sigmoid_d);
    net.setLayerActivation(1, softmax, softmax_d);

    net.setLoss(cross_entropy, cross_entropy_d);
    net.setSm();
    net.setCe();

    net.init(0.5f, -0.5f);

    Matrix input = {
        {1.0f},
        {0.0f}
    };

    Matrix expected = {
        {0.0f},
        {1.0f}
    };

    Matrix grad;
    float loss = net.train(input, expected, grad);

    Matrix output = net.use(input);

    std::cout << "loss = " << loss << std::endl;
    std::cout << output << std::endl;

    return 0;
}
```

---

## Minimal CNN Example

```cpp
#include "matrix.h"
#include "tensor_3d.h"
#include "activation.h"
#include "losses.h"
#include "network.h"

#include <iostream>

using namespace LibMatchstick;
using namespace LibMatchstick::Activation;
using namespace LibMatchstick::Losses;

int main() {
    CNN net(2, 0.001f, 2, 0.01f);

    // Input: 1 x 28 x 28
    // Output: 8 x 14 x 14
    net.setLayer(
        0,
        1, 28, 28,
        8, 14, 14,
        1, 3, 3,
        2, 1
    );

    // Output: 16 x 7 x 7
    net.setLayer(
        1,
        8, 14, 14,
        16, 7, 7,
        8, 3, 3,
        2, 1
    );

    net.setLayerActivation(0, relu_t, relu_t_d);
    net.setLayerActivation(1, relu_t, relu_t_d);

    // Final convolution output: 16 * 7 * 7 = 784
    net.mlp().setLayer(0, 784, 128);
    net.mlp().setLayer(1, 128, 10);

    net.mlp().setLayerActivation(0, relu, relu_d);
    net.mlp().setLayerActivation(1, softmax, softmax_d);

    net.mlp().setLoss(cross_entropy, cross_entropy_d);
    net.mlp().setSm();
    net.mlp().setCe();

    net.init(0.1f, -0.1f);

    Tensor3d input(1, 28, 28);

    Matrix expected(10, 1);
    expected.set(3, 0, 1.0f);

    float loss = net.train(input, expected);

    Matrix output = net.use(input);

    std::cout << "loss = " << loss << std::endl;
    std::cout << output << std::endl;

    return 0;
}
```

---

## API Overview

LibMatchstick exposes the following main modules.

### Core Containers

| Type | Description |
|---|---|
| `Matrix` | 2D floating-point matrix |
| `Tensor2d` | Alias of `Matrix` |
| `Tensor3d` | 3D tensor using `channel x height x width` layout |
| `Tensor4d` | 4D tensor using `batch x channel x height x width` layout |

### Activation Functions

Namespace:

```cpp
LibMatchstick::Activation
```

Matrix activation functions:

- `relu`
- `relu_d`
- `leaky_relu`
- `leaky_relu_d`
- `sigmoid`
- `sigmoid_d`
- `tanh`
- `tanh_d`
- `identity`
- `identity_d`
- `softmax`
- `softmax_d`

Tensor activation functions:

- `relu_t`
- `relu_t_d`
- `leaky_relu_t`
- `leaky_relu_t_d`
- `sigmoid_t`
- `sigmoid_t_d`
- `tanh_t`
- `tanh_t_d`
- `identity_t`
- `identity_t_d`

### Loss Functions

Namespace:

```cpp
LibMatchstick::Losses
```

Available loss functions:

- `MSE`
- `MSE_d`
- `MAE`
- `MAE_d`
- `cross_entropy`
- `cross_entropy_d`

### Layers

| Type | Description |
|---|---|
| `MLPLayer` | Fully connected layer |
| `CNNLayer` | Convolution layer |

### Networks

| Type | Description |
|---|---|
| `MLP` | Multi-layer perceptron wrapper |
| `CNN` | Convolutional neural network wrapper with internal MLP |

For full API details, see:

```text
docs/API.md
```

---

## Shape Conventions

### Matrix

```text
height x width
```

MLP inputs and outputs usually use column vectors:

```text
n x 1
```

### Tensor3d

```text
channel x height x width
```

Example:

```text
1 x 28 x 28
3 x 224 x 224
```

### Tensor4d

```text
batch x channel x height x width
```

For convolution kernels, `Tensor4d` is usually interpreted as:

```text
out_channel x in_channel x kernel_height x kernel_width
```

---

## Notes

### CUDA Memory

Some implementations may store data in CUDA device memory.

Therefore, pointers returned by `getData()` should be used carefully:

- Do not manually free them.
- Do not use them after the object is destroyed.
- Do not assume they are directly accessible from CPU code.
- Prefer the public APIs such as `set`, `get`, `saveWeight`, `loadWeight`, `saveKernel`, and `loadKernel`.

### CNN Output Shape

The current API requires users to explicitly provide output height and width for convolution layers.

For standard convolution:

```text
out_h = floor((in_h + 2 * padding - kernel_h) / stride) + 1
out_w = floor((in_w + 2 * padding - kernel_w) / stride) + 1
```

Users should make sure the provided output shape is consistent with the convolution configuration.

### Softmax and Cross Entropy

For classification tasks, the recommended output-layer setup is:

```cpp
net.setLayerActivation(last, softmax, softmax_d);
net.setLoss(cross_entropy, cross_entropy_d);
net.setSm();
net.setCe();
```

For `CNN`, configure the internal MLP in the same way:

```cpp
net.mlp().setLayerActivation(last, softmax, softmax_d);
net.mlp().setLoss(cross_entropy, cross_entropy_d);
net.mlp().setSm();
net.mlp().setCe();
```

---

## Documentation

Full API documentation is available in:

```text
docs/API.md
```

---

## License

LibMatchstick is licensed under the GNU Lesser General Public License v3.0 or later.

See:

- `LICENSE`
- `COPYING`
- `COPYING.LESSER`

---

## Disclaimer

LibMatchstick is an educational and experimental project.

It is intended for learning, demonstration, and small-scale experiments. It does not provide the same level of optimization, model coverage, numerical robustness, or ecosystem integration as mature machine learning frameworks.

