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
- Max pooling layer
- MLP network wrapper
- CNN network wrapper
- C API wrapper based on opaque handles
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
│   ├── matchstick/
│   │   ├ activation.h
│   │   ├ loss.h
│   │   ├ matrix.h
│   │   ├ network.h
│   │   ├ tensor_3d.h
│   │   └ tensor_4d.h
│   └── matchstick_c/
│       ├ matchstick.h
│       ├ matrix.h
│       ├ network.h
│       ├ tensor_3d.h
│       └ tensor_4d.h
├── src/
├── docs/
│   ├── API.md
│   └── API_C.md
├── LICENSE
├── COPYING
├── COPYING.LESSER
├── makefile
└── README.md
```

---

## Requirements

The exact requirements depend on how the library is built on your system.

Common requirements:

- make
- CUDA Toolkit
- A CUDA-capable GPU
- A compiler supported by your CUDA Toolkit

On Linux, typical compilers include:

- GCC
- Clang
- NVCC with a supported host compiler

---

## Build

A common make build flow is:

```bash
make
make install
```

The generated library is usually placed somewhere under the `build/` directory, depending on the current `makefile`.

For example, the result may be a shared library such as:

```text
libmatchstick.so
```

on Linux

By default, the binary so file will be installed to:
```test
/usr/local/lib
```

and the headers installed to:
```text
/usr/local/include
```

---

## Using LibMatchstick

LibMatchstick provides both a native C++ API and an opaque-handle C API. Use the C++ API when writing C++ projects directly, and use the C API when binding LibMatchstick from C or other languages.

### C++ API

To use LibMatchstick in another C++ project, include the public headers and link against the generated library.

Example include usage:

```cpp
#include <matchstick/matrix.h>
#include <matchstick/tensor_3d.h>
#include <matchstick/activation.h>
#include <matchstick/loss.h>
#include <matchstick/network.h>
```


---

### C API

The C API headers are installed under `matchstick_c/`. The recommended umbrella header is:

```c
#include <matchstick_c/matchstick.h>
```

It includes the C wrappers for matrix, tensor, and network objects:

```c
#include <matchstick_c/matrix.h>
#include <matchstick_c/tensor_3d.h>
#include <matchstick_c/tensor_4d.h>
#include <matchstick_c/network.h>
```

The C API uses opaque handles instead of exposing C++ classes directly:

```c
matchstick_matrix
matchstick_tensor_3d
matchstick_tensor_4d
matchstick_mlp
matchstick_cnn
```

Objects returned by LibMatchstick must be released with the matching release function, for example:

```c
free_matchstick_matrix(m);
free_matchstick_tensor_3d(t);
free_matchstick_tensor_4d(t4);
free_matchstick_mlp(mlp);
free_matchstick_cnn(cnn);
```

Do not use plain `free()` on LibMatchstick opaque handles. The only common exception is `save_bias_matchstick_cnn()`, which returns a raw `float *` buffer and should be released with standard C `free()`.

A minimal C API matrix example:

```c
#include <matchstick_c/matchstick.h>

int main(void) {
    float data[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    matchstick_matrix m = init_matchstick_matrix(2, 2, data);

    print_matchstick_matrix(m);

    float x = get_matchstick_matrix(m, 0, 1);
    (void)x;

    free_matchstick_matrix(m);
    return 0;
}
```

A minimal C API MLP example:

```c
#include <matchstick_c/matchstick.h>

int main(void) {
    matchstick_mlp net = init_matchstick_mlp(2, 0.01f);

    set_layer_matchstick_mlp(net, 0, 2, 4);
    set_layer_matchstick_mlp(net, 1, 4, 2);

    set_layer_activation_matchstick_mlp(net, 0, matchstick_activation_sigmoid);
    set_layer_activation_matchstick_mlp(net, 1, matchstick_activation_softmax);

    set_loss_matchstick_mlp(net, matchstick_loss_ce);
    set_sm_matchstick_mlp(net);
    set_ce_matchstick_mlp(net);

    float input_data[] = {1.0f, 0.0f};
    float expected_data[] = {0.0f, 1.0f};

    matchstick_matrix input = init_matchstick_matrix(2, 1, input_data);
    matchstick_matrix expected = init_matchstick_matrix(2, 1, expected_data);
    matchstick_matrix grad = init_matchstick_matrix(2, 1, expected_data);

    float loss = train_matchstick_mlp(net, input, expected, grad);
    (void)loss;

    matchstick_matrix output = use_matchstick_mlp(net, input);
    print_matchstick_matrix(output);

    free_matchstick_matrix(output);
    free_matchstick_matrix(grad);
    free_matchstick_matrix(expected);
    free_matchstick_matrix(input);
    free_matchstick_mlp(net);

    return 0;
}
```

A minimal C API CNN layer setup example:

```c
#include <matchstick_c/matchstick.h>

int main(void) {
    matchstick_cnn net = init_matchstick_cnn(2, 0.001f, 2, 0.01f);

    set_convolution_layer_matchstick_cnn(
        net,
        0,
        1, 28, 28,
        8, 14, 14,
        1, 3, 3,
        2, 1,
        matchstick_activation_relu
    );

    set_pooling_layer_matchstick_cnn(
        net,
        1,
        8, 14, 14,
        8, 7, 7,
        2, 2,
        2, 0
    );

    set_layer_matchstick_cnn_mlp(net, 0, 8 * 7 * 7, 64);
    set_layer_matchstick_cnn_mlp(net, 1, 64, 10);

    free_matchstick_cnn(net);
    return 0;
}
```

For full C API details, see:

```text
docs/API_C.md
```

---

## Minimal MLP Example

```cpp
#include <matchstick/matrix.h>
#include <matchstick/activation.h>
#include <matchstick/loss.h>
#include <matchstick/network.h>

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
#include <matchstick/matrix.h>
#include <matchstick/tensor_3d.h>
#include <matchstick/activation.h>
#include <matchstick/loss.h>
#include <matchstick/network.h>

#include <iostream>

using namespace LibMatchstick;
using namespace LibMatchstick::Activation;
using namespace LibMatchstick::Losses;

int main() {
    CNN net(2, 0.001f, 2, 0.01f);

    // Input: 1 x 28 x 28
    // Output: 8 x 14 x 14
    net.setConvolutionLayer(
        0,
        1, 28, 28,
        8, 14, 14,
        1, 3, 3,
        2, 1,
        relu_t, relu_t_d
    );

    // Output: 16 x 7 x 7
    net.setConvolutionLayer(
        1,
        8, 14, 14,
        16, 7, 7,
        8, 3, 3,
        2, 1,
        relu_t, relu_t_d
    );


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
| `CNNLayer` | Abstract CNN layer base class |
| `CNNConvolutionLayer` | Convolution layer |
| `CNNPoolingLayer` | Max pooling layer |

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
docs/API_C.md
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

