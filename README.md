# LibMatchstick

LibMatchstick is a lightweight C++ / CUDA neural-network library for learning, experimentation, and small course-project style models.

Version **2.0.0** keeps the original C++ interface, adds a new opaque-handle based **C API**, and slightly reorganizes the public C++ API layout.

---

## Features

- CUDA-backed implementation
- C++ API for direct use from C++ projects
- C API for C projects and FFI bindings
- Matrix container
- 3D tensor container using `channel × height × width`
- 4D tensor container using `batch × channel × height × width`
- Basic matrix and tensor operations
- MLP layers and MLP network wrapper
- CNN layers and CNN network wrapper
- Common activation functions:
  - ReLU
  - Leaky ReLU
  - Sigmoid
  - Tanh
  - Identity
  - Softmax for matrix output
- Common loss functions:
  - Mean Squared Error
  - Mean Absolute Error
  - Cross Entropy
- Parameter save/load interfaces for weights, biases, and kernels

---

## Version 2.0.0 Changes

Compared with v1.0.0, v2.0.0 mainly introduces the C API and makes small C++ API adjustments.

### New C API

v2.0.0 adds public C headers under:

```text
include/matchstick_c/
├── matchstick.h
├── matrix.h
├── tensor_3d.h
├── tensor_4d.h
└── network.h
```

The C API uses opaque handles such as:

```c
matchstick_matrix
matchstick_tensor_3d
matchstick_tensor_4d
matchstick_mlp
matchstick_cnn
```

These handles are suitable for C programs and for language bindings through FFI.

### C++ API Adjustments

Public C++ headers are now placed under:

```text
include/matchstick/
```

Use the new include style:

```cpp
#include <matchstick/matrix.h>
#include <matchstick/tensor_3d.h>
#include <matchstick/tensor_4d.h>
#include <matchstick/activation.h>
#include <matchstick/loss.h>
#include <matchstick/layer.h>
#include <matchstick/network.h>
```

The loss namespace and header have also been adjusted:

| v1.0.0 style | v2.0.0 style |
|---|---|
| `#include "losses.h"` | `#include <matchstick/loss.h>` |
| `LibMatchstick::Losses` | `LibMatchstick::Loss` |

---

## Repository Layout

```text
LibMatchstick/
├── include/
│   ├── matchstick/          # C++ public API
│   └── matchstick_c/        # C public API
├── src/
│   ├── cpp/                 # C++ / CUDA implementation
│   └── c_api/               # C API wrapper implementation
├── docs/
│   ├── API.md               # C++ API reference
│   └── API_C.md             # C API reference
├── CMakeLists.txt
├── LICENSE
├── COPYING
├── COPYING.LESSER
└── README.md
```

---

## Requirements

Common requirements:

- CMake 3.24 or newer
- CUDA Toolkit
- A CUDA-capable GPU
- A C++17 compiler supported by your CUDA Toolkit

The project is built as a shared library target named `matchstick`.

On Linux, the output is typically:

```text
libmatchstick.so
```

---

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The current CMake project enables CUDA and builds the shared library target `matchstick`.

If you need to choose a CUDA architecture manually, pass `CMAKE_CUDA_ARCHITECTURES`:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

If your CUDA Toolkit does not support your default host compiler, select a supported host compiler through your build environment or CMake toolchain settings.

---

## Using LibMatchstick from C++

### Include Headers

```cpp
#include <matchstick/matrix.h>
#include <matchstick/tensor_3d.h>
#include <matchstick/tensor_4d.h>
#include <matchstick/activation.h>
#include <matchstick/loss.h>
#include <matchstick/layer.h>
#include <matchstick/network.h>
```

### CMake Example

```cmake
target_include_directories(your_target PRIVATE /path/to/LibMatchstick/include)
target_link_libraries(your_target PRIVATE /path/to/libmatchstick.so)
```

Adjust the library path according to your build output.

---

## Minimal C++ MLP Example

```cpp
#include <matchstick/matrix.h>
#include <matchstick/activation.h>
#include <matchstick/loss.h>
#include <matchstick/network.h>

#include <iostream>

using namespace LibMatchstick;
using namespace LibMatchstick::Activation;
using namespace LibMatchstick::Loss;

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

    std::cout << "loss = " << loss << '\n';
    std::cout << output << '\n';

    return 0;
}
```

---

## Minimal C++ CNN Example

```cpp
#include <matchstick/matrix.h>
#include <matchstick/tensor_3d.h>
#include <matchstick/activation.h>
#include <matchstick/loss.h>
#include <matchstick/network.h>

#include <iostream>

using namespace LibMatchstick;
using namespace LibMatchstick::Activation;
using namespace LibMatchstick::Loss;

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

    std::cout << "loss = " << loss << '\n';
    std::cout << output << '\n';

    return 0;
}
```

---

## Using LibMatchstick from C

### Include Header

The recommended C include is:

```c
#include <matchstick_c/matchstick.h>
```

This umbrella header includes the matrix, tensor, and network C APIs.

### Minimal C MLP Example

```c
#include <matchstick_c/matchstick.h>

#include <stdio.h>

int main(void) {
    matchstick_mlp net = init_matchstick_mlp(2, 0.01f);

    set_layer_matchstick_mlp(net, 0, 2, 4);
    set_layer_matchstick_mlp(net, 1, 4, 2);

    set_layer_activation_matchstick_mlp(net, 0, matchstick_activation_sigmoid);
    set_layer_activation_matchstick_mlp(net, 1, matchstick_activation_softmax);

    set_loss_matchstick_mlp(net, matchstick_loss_ce);
    set_sm_matchstick_mlp(net);
    set_ce_matchstick_mlp(net);

    shuffle_matchstick_mlp(net, 0.5f, -0.5f);

    float input_data[] = {
        1.0f,
        0.0f
    };

    float expected_data[] = {
        0.0f,
        1.0f
    };

    float grad_data[] = {
        0.0f,
        0.0f
    };

    matchstick_matrix input = init_matchstick_matrix(2, 1, input_data);
    matchstick_matrix expected = init_matchstick_matrix(2, 1, expected_data);
    matchstick_matrix grad = init_matchstick_matrix(2, 1, grad_data);

    float loss = train_matchstick_mlp(net, input, expected, grad);
    matchstick_matrix output = use_matchstick_mlp(net, input);

    printf("loss = %f\n", loss);
    print_matchstick_matrix(output);

    free_matchstick_matrix(output);
    free_matchstick_matrix(grad);
    free_matchstick_matrix(expected);
    free_matchstick_matrix(input);
    free_matchstick_mlp(net);

    return 0;
}
```

---

## C API Memory Ownership Rules

The C API uses opaque handles. Users should not access the internal fields of these objects and should not release them with plain `free()`.

Every `matchstick_*` handle returned by LibMatchstick must be released with the matching `free_matchstick_*` function.

| Returned object | Correct release function |
|---|---|
| `matchstick_matrix` | `free_matchstick_matrix()` |
| `matchstick_tensor_3d` | `free_matchstick_tensor_3d()` |
| `matchstick_tensor_4d` | `free_matchstick_tensor_4d()` |
| `matchstick_mlp` | `free_matchstick_mlp()` |
| `matchstick_cnn` | `free_matchstick_cnn()` |

Special case:

```c
float *bias = save_bias_matchstick_cnn(cnn, layer_index, out_c);

/* use bias[0] ... bias[out_c - 1] */

free(bias);
```

`save_bias_matchstick_cnn()` returns a plain heap buffer created with `malloc`, so it must be released with standard C `free()`.

Do not pass this pointer to any `free_matchstick_*` function.

---

## API Overview

### C++ API Headers

```text
include/matchstick/matrix.h
include/matchstick/tensor_3d.h
include/matchstick/tensor_4d.h
include/matchstick/activation.h
include/matchstick/loss.h
include/matchstick/layer.h
include/matchstick/network.h
```

Main C++ namespace:

```cpp
namespace LibMatchstick;
namespace LibMatchstick::Activation;
namespace LibMatchstick::Loss;
```

### C API Headers

```text
include/matchstick_c/matchstick.h
include/matchstick_c/matrix.h
include/matchstick_c/tensor_3d.h
include/matchstick_c/tensor_4d.h
include/matchstick_c/network.h
```

Main C API object handles:

```c
matchstick_matrix
matchstick_tensor_3d
matchstick_tensor_4d
matchstick_mlp
matchstick_cnn
```

---

## Shape Conventions

### Matrix

```text
height × width
```

MLP inputs and outputs usually use column vectors:

```text
n × 1
```

### Tensor3d

```text
channel × height × width
```

Examples:

```text
1 × 28 × 28
3 × 224 × 224
```

### Tensor4d

```text
batch × channel × height × width
```

For convolution kernels, `Tensor4d` is usually interpreted as:

```text
out_channel × in_channel × kernel_height × kernel_width
```

---

## CNN Output Shape

The current CNN layer API requires users to explicitly provide the output height and output width.

For standard convolution:

```text
out_h = floor((in_h + 2 * padding - kernel_h) / stride) + 1
out_w = floor((in_w + 2 * padding - kernel_w) / stride) + 1
```

Make sure the output shape passed to `setLayer()` or `set_layer_matchstick_cnn()` is consistent with the convolution configuration.

---

## Softmax and Cross Entropy

For classification tasks using C++ `MLP`, the recommended output-layer setup is:

```cpp
net.setLayerActivation(last, softmax, softmax_d);
net.setLoss(cross_entropy, cross_entropy_d);
net.setSm();
net.setCe();
```

For C++ `CNN`, configure the internal MLP in the same way:

```cpp
net.mlp().setLayerActivation(last, softmax, softmax_d);
net.mlp().setLoss(cross_entropy, cross_entropy_d);
net.mlp().setSm();
net.mlp().setCe();
```

For the C API:

```c
set_layer_activation_matchstick_mlp(net, last, matchstick_activation_softmax);
set_loss_matchstick_mlp(net, matchstick_loss_ce);
set_sm_matchstick_mlp(net);
set_ce_matchstick_mlp(net);
```

---

## Documentation

Full API references are available in:

```text
docs/API.md
docs/API_C.md
```

For v2.0.0, the C API is part of the public interface and should be documented together with the C++ API.

---

## Project Status

LibMatchstick is an educational and experimental project.

It is suitable for:

- Simple MLP experiments
- Small CNN experiments
- Classification demos
- CUDA learning
- Neural-network implementation practice
- Course project demonstrations

It is not intended to replace mature machine-learning frameworks such as PyTorch, TensorFlow, ONNX Runtime, or TensorRT.

---

## License

LibMatchstick is licensed under the GNU Lesser General Public License v3.0 or later.

SPDX identifier:

```text
LGPL-3.0-or-later
```

See:

```text
LICENSE
COPYING
COPYING.LESSER
```

---

## Disclaimer

This library is provided for learning, demonstration, and small-scale experimentation.

It does not provide the same level of optimization, model coverage, numerical robustness, tooling, or ecosystem integration as mature machine-learning frameworks.

