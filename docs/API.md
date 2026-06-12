# LibMatchstick API Documentation

LibMatchstick is a lightweight C++ / CUDA neural network library. It provides matrix and tensor containers, activation functions, loss functions, MLP layers, CNN layers, and high-level MLP / CNN network wrappers.

This document describes the public API declared in the current header files:

- `matrix.h`
- `tensor_3d.h`
- `tensor_4d.h`
- `activation.h`
- `loss.h`
- `layer.h`
- `network.h`

> Note: This document describes the public declarations exposed by the headers. Runtime behavior such as invalid-shape handling, CUDA memory ownership, random initialization, and error reporting depends on the corresponding implementation files.

---

## Table of Contents

- [Namespace](#namespace)
- [Core Data Structures](#core-data-structures)
  - [`Matrix`](#matrix)
  - [`Tensor2d`](#tensor2d)
  - [`Tensor3d`](#tensor3d)
  - [`Tensor4d`](#tensor4d)
- [Activation Functions](#activation-functions)
  - [`LibMatchstick::Activation`](#libmatchstickactivation)
- [Loss Functions](#loss-functions)
  - [`LibMatchstick::Loss`](#libmatchstickloss)
- [Layers](#layers)
  - [`MLPLayer`](#mlplayer)
  - [`CNNLayer`](#cnnlayer)
  - [`CNNConvolutionLayer`](#cnnconvolutionlayer)
  - [`CNNPoolingLayer`](#cnnpoolinglayer)
- [Networks](#networks)
  - [`MLP`](#mlp)
  - [`CNN`](#cnn)
- [Typical Usage](#typical-usage)
  - [MLP Example](#mlp-example)
  - [CNN Example](#cnn-example)
- [Shape Conventions](#shape-conventions)
- [Memory and Object Lifetime](#memory-and-object-lifetime)
- [Common Notes](#common-notes)
- [Recommended Repository Layout](#recommended-repository-layout)
- [License](#license)

---

# Namespace

All core APIs are declared under:

```cpp
namespace LibMatchstick
```

Activation functions are declared under:

```cpp
namespace LibMatchstick::Activation
```

Loss functions are declared under:

```cpp
namespace LibMatchstick::Loss
```

Typical usage:

```cpp
using namespace LibMatchstick;
using namespace LibMatchstick::Activation;
using namespace LibMatchstick::Loss;
```

---

# Core Data Structures

## `Matrix`

Header:

```cpp
#include "matrix.h"
```

Full name:

```cpp
LibMatchstick::Matrix
```

`Matrix` represents a two-dimensional matrix. It stores floating-point data and exposes basic matrix operations.

Internally, the class stores:

```cpp
float* data;
size_t h;
size_t w;
```

The public API does not expose ownership details directly. Use `getData()` only when low-level access is required.

---

## Constructors and Destructor

```cpp
Matrix();
Matrix(size_t h, size_t w);
Matrix(std::initializer_list<std::initializer_list<float>> a);
~Matrix();
```

### `Matrix()`

Creates an empty matrix.

### `Matrix(size_t h, size_t w)`

Creates a matrix with shape:

```text
h x w
```

Parameters:

| Parameter | Description |
|---|---|
| `h` | Number of rows |
| `w` | Number of columns |

Example:

```cpp
Matrix a(3, 4);
```

### `Matrix(std::initializer_list<std::initializer_list<float>> a)`

Creates a matrix from a nested initializer list.

Example:

```cpp
Matrix a = {
    {1.0f, 2.0f, 3.0f},
    {4.0f, 5.0f, 6.0f}
};
```

The matrix above has shape:

```text
2 x 3
```

---

## Output

```cpp
friend std::ostream& operator<<(std::ostream& os, const Matrix& a);
```

Writes a matrix to a C++ output stream.

Example:

```cpp
Matrix a = {
    {1, 2},
    {3, 4}
};

std::cout << a << std::endl;
```

---

## Shape Query

```cpp
size_t getHeight() const;
size_t getWidth() const;
```

### `getHeight()`

Returns the number of rows.

### `getWidth()`

Returns the number of columns.

---

## Resize

```cpp
void resize(size_t h, size_t w);
```

Resizes the matrix to a new shape.

Parameters:

| Parameter | Description |
|---|---|
| `h` | New number of rows |
| `w` | New number of columns |

Note: Whether previous data is preserved depends on the implementation. Treat this operation as a possible reallocation.

---

## Copy and Assignment

```cpp
Matrix(const Matrix& a);
Matrix& operator=(const Matrix& a);
```

`Matrix` supports copy construction and copy assignment.

Example:

```cpp
Matrix a = {{1, 2}, {3, 4}};

Matrix b = a;

Matrix c;
c = a;
```

---

## Addition and Subtraction

```cpp
Matrix operator+(const Matrix& a) const;
Matrix operator-(const Matrix& a) const;

Matrix& operator+=(const Matrix& a);
Matrix& operator-=(const Matrix& a);
```

Performs element-wise matrix addition and subtraction.

The operands are expected to have compatible shapes, usually exactly the same shape.

Example:

```cpp
Matrix a = {{1, 2}, {3, 4}};
Matrix b = {{5, 6}, {7, 8}};

Matrix c = a + b;
Matrix d = b - a;

a += b;
b -= a;
```

---

## Hadamard Product

```cpp
Matrix hadamard(const Matrix& a) const;
```

Performs element-wise multiplication.

The operands are expected to have the same shape.

Example:

```cpp
Matrix a = {{1, 2}, {3, 4}};
Matrix b = {{5, 6}, {7, 8}};

Matrix c = a.hadamard(b);
```

---

## Matrix Multiplication

```cpp
Matrix operator*(const Matrix& a) const;
```

Performs matrix multiplication.

The expected shape rule is:

```text
this->getWidth() == a.getHeight()
```

The result usually has shape:

```text
this->getHeight() x a.getWidth()
```

Example:

```cpp
Matrix a = {
    {1, 2, 3},
    {4, 5, 6}
};

Matrix b = {
    {1, 2},
    {3, 4},
    {5, 6}
};

Matrix c = a * b; // 2 x 2
```

---

## Element Access

```cpp
void set(size_t i, size_t j, float v);
float get(size_t i, size_t j) const;
```

### `set`

Sets the element at row `i`, column `j`.

### `get`

Returns the element at row `i`, column `j`.

Example:

```cpp
Matrix a(2, 2);

a.set(0, 0, 1.0f);
a.set(0, 1, 2.0f);

float x = a.get(0, 1);
```

---

## Raw Data Access

```cpp
float* getData();
const float* getData() const;
```

Returns the internal data pointer.

Important notes:

- Do not manually free this pointer.
- Do not use it after the object has been destroyed.
- If the implementation stores data in CUDA device memory, the pointer may not be directly dereferenceable on the CPU.
- Directly modifying the raw buffer bypasses class-level checks.

---

## Transpose

```cpp
Matrix transpose() const;
```

Returns the transpose of the matrix.

Example:

```cpp
Matrix a = {
    {1, 2, 3},
    {4, 5, 6}
};

Matrix b = a.transpose(); // 3 x 2
```

---

## Scalar Multiplication

```cpp
Matrix operator*(float a) const;
Matrix operator*=(float a);

friend Matrix operator*(float a, const Matrix& b);
```

Supports multiplication by a scalar.

Example:

```cpp
Matrix a = {{1, 2}, {3, 4}};

Matrix b = a * 2.0f;
Matrix c = 2.0f * a;

a *= 0.5f;
```

---

## `Tensor2d`

```cpp
typedef Matrix Tensor2d;
```

`Tensor2d` is an alias for `Matrix`.

The following two declarations are equivalent:

```cpp
Matrix a(3, 4);
Tensor2d b(3, 4);
```

---

## `Tensor3d`

Header:

```cpp
#include "tensor_3d.h"
```

Full name:

```cpp
LibMatchstick::Tensor3d
```

`Tensor3d` represents a three-dimensional tensor with the shape convention:

```text
channel x height x width
```

That is:

```text
c x h x w
```

It is commonly used for CNN input images, feature maps, and intermediate convolution outputs.

Internally, the class stores:

```cpp
size_t c;
size_t h;
size_t w;
float* data;
```

---

## Constructors and Destructor

```cpp
Tensor3d();
Tensor3d(size_t c, size_t h, size_t w);
Tensor3d(std::initializer_list<std::initializer_list<std::initializer_list<float>>> a);
~Tensor3d();
```

### `Tensor3d()`

Creates an empty tensor.

### `Tensor3d(size_t c, size_t h, size_t w)`

Creates a tensor with shape:

```text
c x h x w
```

Parameters:

| Parameter | Description |
|---|---|
| `c` | Number of channels |
| `h` | Height |
| `w` | Width |

Example:

```cpp
Tensor3d x(3, 32, 32);
```

### Initializer-list Constructor

Example:

```cpp
Tensor3d x = {
    {
        {1, 2},
        {3, 4}
    },
    {
        {5, 6},
        {7, 8}
    }
};
```

This tensor has shape:

```text
2 x 2 x 2
```

It contains two channels, and each channel is a `2 x 2` matrix.

---

## Output

```cpp
friend std::ostream& operator<<(std::ostream& os, const Tensor3d& a);
```

Writes a `Tensor3d` to a C++ output stream.

Example:

```cpp
std::cout << x << std::endl;
```

---

## Shape Query

```cpp
size_t getChannel() const;
size_t getHeight() const;
size_t getWidth() const;
```

Returns the number of channels, height, and width.

---

## Resize

```cpp
void resize(size_t c, size_t h, size_t w);
```

Resizes the tensor.

Note: Whether previous data is preserved depends on the implementation. Treat this operation as a possible reallocation.

---

## Copy and Assignment

```cpp
Tensor3d(const Tensor3d& a);
Tensor3d& operator=(const Tensor3d& a);
```

`Tensor3d` supports copy construction and copy assignment.

---

## Addition and Subtraction

```cpp
Tensor3d operator+(const Tensor3d& a) const;
Tensor3d operator-(const Tensor3d& a) const;

Tensor3d& operator+=(const Tensor3d& a);
Tensor3d& operator-=(const Tensor3d& a);
```

Performs element-wise addition and subtraction.

The operands are expected to have the same shape.

---

## Hadamard Product

```cpp
Tensor3d hadamard(const Tensor3d& a) const;
```

Performs element-wise multiplication.

The operands are expected to have the same shape.

---

## Element Access

```cpp
void set(size_t i, size_t j, size_t k, float v);
float get(size_t i, size_t j, size_t k) const;
```

Index meaning:

| Parameter | Description |
|---|---|
| `i` | Channel index |
| `j` | Height index |
| `k` | Width index |

Example:

```cpp
Tensor3d x(3, 32, 32);

x.set(0, 10, 20, 1.0f);

float v = x.get(0, 10, 20);
```

---

## Raw Data Access

```cpp
float* getData();
const float* getData() const;
```

Returns the internal data pointer.

Important notes:

- Do not manually release this pointer.
- Do not use it after the tensor has been destroyed.
- If the implementation stores data in CUDA device memory, the pointer may not be directly accessible from CPU code.
- Prefer `set`, `get`, and tensor operations unless low-level access is required.

---

## Convolution

```cpp
Tensor3d convolution(const Tensor4d& k, size_t stride, size_t padding) const;
```

Applies convolution to the current tensor.

Parameters:

| Parameter | Description |
|---|---|
| `k` | Convolution kernel |
| `stride` | Stride |
| `padding` | Padding size |

Returns:

```cpp
Tensor3d
```

This is usually the output feature map.

In CNN usage, `Tensor4d` kernels are usually interpreted as:

```text
out_channel x in_channel x kernel_height x kernel_width
```

That is:

```text
b x c x h x w
```

where the first dimension represents the output channel / kernel index.

---

## Flatten

```cpp
Matrix flatten();
```

Flattens a `Tensor3d` into a `Matrix`.

This is commonly used when connecting CNN layers to an MLP.

Common convention:

```text
c x h x w
```

is flattened into:

```text
(c * h * w) x 1
```

Example:

```cpp
Tensor3d x(16, 7, 7);
Matrix v = x.flatten(); // usually 784 x 1
```

---

## Deflatten

```cpp
static Tensor3d deflatten(const Matrix& a, size_t c, size_t h, size_t w);
```

Converts a matrix back into a `Tensor3d`.

Parameters:

| Parameter | Description |
|---|---|
| `a` | Input matrix |
| `c` | Target channel count |
| `h` | Target height |
| `w` | Target width |

Example:

```cpp
Matrix v(784, 1);

Tensor3d x = Tensor3d::deflatten(v, 16, 7, 7);
```

The number of elements in `a` is expected to match:

```text
c * h * w
```

---

## `Tensor4d`

Header:

```cpp
#include "tensor_4d.h"
```

Full name:

```cpp
LibMatchstick::Tensor4d
```

`Tensor4d` represents a four-dimensional tensor with the shape convention:

```text
batch x channel x height x width
```

That is:

```text
b x c x h x w
```

In the current library, it is mainly used for CNN convolution kernels.

Internally, the class stores:

```cpp
float* data;
size_t b;
size_t c;
size_t h;
size_t w;
```

---

## Constructors and Destructor

```cpp
Tensor4d();
Tensor4d(size_t b, size_t c, size_t h, size_t w);
Tensor4d(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<float>>>> a);
~Tensor4d();
```

### `Tensor4d()`

Creates an empty tensor.

### `Tensor4d(size_t b, size_t c, size_t h, size_t w)`

Creates a tensor with shape:

```text
b x c x h x w
```

Parameters:

| Parameter | Description |
|---|---|
| `b` | Batch size or number of kernels |
| `c` | Number of channels |
| `h` | Height |
| `w` | Width |

### Initializer-list Constructor

Example:

```cpp
Tensor4d k = {
    {
        {
            {1, 0, -1},
            {1, 0, -1},
            {1, 0, -1}
        }
    }
};
```

This can be used to construct a simple convolution kernel.

---

## Shape Query

```cpp
size_t getBatch() const;
size_t getChannel() const;
size_t getHeight() const;
size_t getWidth() const;
```

Returns the four shape dimensions.

---

## Resize

```cpp
void resize(size_t b, size_t c, size_t h, size_t w);
```

Resizes the tensor.

---

## Copy and Assignment

```cpp
Tensor4d(const Tensor4d& a);
Tensor4d& operator=(const Tensor4d& a);
```

`Tensor4d` supports copy construction and copy assignment.

---

## Element Access

```cpp
void set(size_t i, size_t j, size_t k, size_t l, float v);
float get(size_t i, size_t j, size_t k, size_t l) const;
```

Index meaning:

| Parameter | Description |
|---|---|
| `i` | Batch / kernel index |
| `j` | Channel index |
| `k` | Height index |
| `l` | Width index |

Example:

```cpp
Tensor4d k(8, 3, 3, 3);

k.set(0, 0, 1, 1, 0.5f);

float v = k.get(0, 0, 1, 1);
```

---

## Raw Data Access

```cpp
float* getData();
const float* getData() const;
```

Returns the internal data pointer.

The same ownership and CUDA-memory notes from `Matrix::getData()` and `Tensor3d::getData()` apply here.

---

# Activation Functions

## `LibMatchstick::Activation`

Header:

```cpp
#include "activation.h"
```

Namespace:

```cpp
namespace LibMatchstick::Activation
```

This module provides activation functions for both `Matrix` and `Tensor3d`.

---

## Matrix Activation Functions

```cpp
Matrix relu(const Matrix& a);
Matrix relu_d(const Matrix& a);

Matrix leaky_relu(const Matrix& a);
Matrix leaky_relu_d(const Matrix& a);

Matrix sigmoid(const Matrix& a);
Matrix sigmoid_d(const Matrix& a);

Matrix tanh(const Matrix& a);
Matrix tanh_d(const Matrix& a);

Matrix identity(const Matrix& a);
Matrix identity_d(const Matrix& a);

Matrix softmax(const Matrix& a);
Matrix softmax_d(const Matrix& a);
```

Function summary:

| Function | Description |
|---|---|
| `relu` | ReLU activation |
| `relu_d` | Derivative of ReLU |
| `leaky_relu` | Leaky ReLU activation |
| `leaky_relu_d` | Derivative of Leaky ReLU |
| `sigmoid` | Sigmoid activation |
| `sigmoid_d` | Derivative of sigmoid |
| `tanh` | Hyperbolic tangent activation |
| `tanh_d` | Derivative of hyperbolic tangent |
| `identity` | Identity activation |
| `identity_d` | Derivative of identity activation |
| `softmax` | Softmax activation |
| `softmax_d` | Derivative of softmax |

Example:

```cpp
using namespace LibMatchstick;
using namespace LibMatchstick::Activation;

Matrix x = {
    {1.0f},
    {-2.0f},
    {3.0f}
};

Matrix y = relu(x);
Matrix dy = relu_d(x);
```

---

## Tensor3d Activation Functions

```cpp
Tensor3d relu_t(const Tensor3d& a);
Tensor3d relu_t_d(const Tensor3d& a);

Tensor3d leaky_relu_t(const Tensor3d& a);
Tensor3d leaky_relu_t_d(const Tensor3d& a);

Tensor3d sigmoid_t(const Tensor3d& a);
Tensor3d sigmoid_t_d(const Tensor3d& a);

Tensor3d tanh_t(const Tensor3d& a);
Tensor3d tanh_t_d(const Tensor3d& a);

Tensor3d identity_t(const Tensor3d& a);
Tensor3d identity_t_d(const Tensor3d& a);
```

Naming convention:

- `_t` means tensor version.
- `_d` means derivative version.

For example:

```cpp
relu_t      // Tensor3d ReLU
relu_t_d    // Tensor3d ReLU derivative
```

Example:

```cpp
using namespace LibMatchstick;
using namespace LibMatchstick::Activation;

Tensor3d x(3, 32, 32);

Tensor3d y = relu_t(x);
Tensor3d dy = relu_t_d(x);
```

---

# Loss Functions

## `LibMatchstick::Loss`

Header:

```cpp
#include "loss.h"
```

Namespace:

```cpp
namespace LibMatchstick::Loss
```

This module provides common loss functions and their derivatives.

All currently declared loss functions operate on `Matrix`.

```cpp
float MSE(const Matrix& x, const Matrix& e);
Matrix MSE_d(const Matrix& x, const Matrix& e);

float MAE(const Matrix& x, const Matrix& e);
Matrix MAE_d(const Matrix& x, const Matrix& e);

float cross_entropy(const Matrix& x, const Matrix& e);
Matrix cross_entropy_d(const Matrix& x, const Matrix& e);
```

Parameter convention:

| Parameter | Description |
|---|---|
| `x` | Network output / prediction |
| `e` | Expected output / target |

---

## `MSE`

```cpp
float MSE(const Matrix& x, const Matrix& e);
Matrix MSE_d(const Matrix& x, const Matrix& e);
```

Mean squared error and its derivative.

This is suitable for regression tasks and simple experiments.

---

## `MAE`

```cpp
float MAE(const Matrix& x, const Matrix& e);
Matrix MAE_d(const Matrix& x, const Matrix& e);
```

Mean absolute error and its derivative.

---

## `cross_entropy`

```cpp
float cross_entropy(const Matrix& x, const Matrix& e);
Matrix cross_entropy_d(const Matrix& x, const Matrix& e);
```

Cross entropy loss and its derivative.

This is commonly used with `Activation::softmax` for classification tasks.

Example:

```cpp
using namespace LibMatchstick;
using namespace LibMatchstick::Loss;

Matrix output = {
    {0.1f},
    {0.7f},
    {0.2f}
};

Matrix expected = {
    {0.0f},
    {1.0f},
    {0.0f}
};

float loss = cross_entropy(output, expected);
Matrix grad = cross_entropy_d(output, expected);
```

---

# Layers

## `MLPLayer`

Header:

```cpp
#include "layer.h"
```

Full name:

```cpp
LibMatchstick::MLPLayer
```

`MLPLayer` represents a fully connected layer.

Internally, the layer stores:

- Weight matrix
- Bias matrix
- Activation function
- Activation derivative
- Last input
- Linear output `z`
- Input size
- Output size
- Softmax-layer marker

---

## Constructors

```cpp
MLPLayer();
MLPLayer(size_t in_size, size_t out_size);
```

### `MLPLayer()`

Creates an empty fully connected layer.

### `MLPLayer(size_t in_size, size_t out_size)`

Creates a fully connected layer.

Parameters:

| Parameter | Description |
|---|---|
| `in_size` | Input vector size |
| `out_size` | Output vector size |

The input is usually expected to have shape:

```text
in_size x 1
```

The output is usually:

```text
out_size x 1
```

---

## Forward Propagation

```cpp
Matrix forward(const Matrix& input);
```

Runs forward propagation for this layer.

Parameters:

| Parameter | Description |
|---|---|
| `input` | Input matrix, usually a column vector |

Returns the layer output.

---

## Backward Propagation

```cpp
Matrix backward(const Matrix& dl_da, float step);
Matrix backward_dz(const Matrix& dl_dz, float step);
```

### `backward`

Runs backpropagation using the gradient of the loss with respect to the activation output.

Parameters:

| Parameter | Description |
|---|---|
| `dl_da` | Gradient of loss with respect to activation output `a` |
| `step` | Learning rate |

Returns the gradient passed to the previous layer.

### `backward_dz`

Runs backpropagation using the gradient of the loss with respect to the linear output `z`.

This is useful when the caller has already combined part of the derivative, for example in the common softmax + cross entropy case.

Parameters:

| Parameter | Description |
|---|---|
| `dl_dz` | Gradient of loss with respect to linear output `z` |
| `step` | Learning rate |

Returns the gradient passed to the previous layer.

---

## Initialization

```cpp
void init(float high = 1, float low = -1);
```

Initializes weights and biases.

Parameters:

| Parameter | Description |
|---|---|
| `high` | Upper bound of initialization range |
| `low` | Lower bound of initialization range |

Example:

```cpp
MLPLayer layer(784, 128);
layer.init(0.1f, -0.1f);
```

---

## Save and Load Parameters

```cpp
Matrix saveWeight() const;
Matrix saveBias() const;

bool loadWeight(const Matrix& W);
bool loadBias(const Matrix& b);
```

### `saveWeight`

Returns a copy of the weight matrix.

### `saveBias`

Returns a copy of the bias matrix.

### `loadWeight`

Loads the weight matrix.

Returns `true` on success and `false` on failure, depending on the implementation.

### `loadBias`

Loads the bias matrix.

Returns `true` on success and `false` on failure, depending on the implementation.

Example:

```cpp
Matrix w = layer.saveWeight();
Matrix b = layer.saveBias();

layer.loadWeight(w);
layer.loadBias(b);
```

---

## Set Activation Function

```cpp
void setActivation(
    const std::function<Matrix(const Matrix&)>& a,
    const std::function<Matrix(const Matrix&)>& a_d
);
```

Sets the activation function and its derivative.

Example:

```cpp
using namespace LibMatchstick::Activation;

MLPLayer layer(784, 128);

layer.setActivation(relu, relu_d);
```

---

## Softmax Marker

```cpp
bool isSm() const;
void setSm();
```

### `setSm`

Marks the layer as a softmax layer.

### `isSm`

Returns whether this layer is marked as a softmax layer.

This is typically used for the output layer when combined with cross entropy loss.

---

## `CNNLayer`

Header:

```cpp
#include "layer.h"
```

Full name:

```cpp
LibMatchstick::CNNLayer
```

`CNNLayer` is the abstract base class for CNN layers. It is not meant to be instantiated directly. The current concrete layer classes are:

- `CNNConvolutionLayer`
- `CNNPoolingLayer`

The base class stores common metadata such as layer type, input shape, output shape, stride, and padding.

Public base API:

```cpp
enum class CNNLayerType {
    Convolution,
    Pooling
};

virtual Tensor3d forward(const Tensor3d& input) = 0;
virtual Tensor3d backward(const Tensor3d& dl_da, float step) = 0;

CNNLayerType getType();
size_t getOutChannel() const;
size_t getOutHeight() const;
size_t getOutWidth() const;
~CNNLayer() = default;
```

### `getType`

Returns whether the layer is a convolution layer or a pooling layer.

### Output Shape Query

```cpp
size_t getOutChannel() const;
size_t getOutHeight() const;
size_t getOutWidth() const;
```

Returns the output channel count, output height, and output width.

---

## `CNNConvolutionLayer`

Full name:

```cpp
LibMatchstick::CNNConvolutionLayer
```

`CNNConvolutionLayer` is the concrete convolution layer. It owns the kernel, bias, cached input, cached pre-activation output, and activation functions.

Constructors and destructor:

```cpp
CNNConvolutionLayer();

CNNConvolutionLayer(
    size_t in_c,
    size_t in_h,
    size_t in_w,
    size_t out_c,
    size_t out_h,
    size_t out_w,
    size_t k_c,
    size_t k_h,
    size_t k_w,
    size_t s,
    size_t p,
    const std::function<Tensor3d(const Tensor3d&)>& a,
    const std::function<Tensor3d(const Tensor3d&)>& a_d
);

~CNNConvolutionLayer();
```

Public API:

```cpp
void init(float high = 1, float low = -1);
Tensor4d saveKernel() const;
std::vector<float> saveBias() const;
bool loadKernel(const Tensor4d& k);
bool loadBias(const std::vector<float>& b);

void setActivation(
    const std::function<Tensor3d(const Tensor3d&)>& a,
    const std::function<Tensor3d(const Tensor3d&)>& a_d
);

CNNConvolutionLayer(const CNNConvolutionLayer& a);
CNNConvolutionLayer& operator=(const CNNConvolutionLayer& a);

Tensor3d forward(const Tensor3d& input);
Tensor3d backward(const Tensor3d& dl_da, float step);
```

For standard convolution, output spatial size is usually:

```text
out_h = floor((in_h + 2 * padding - k_h) / stride) + 1
out_w = floor((in_w + 2 * padding - k_w) / stride) + 1
```

The current API asks the caller to provide `out_h` and `out_w` explicitly, so the caller should ensure that these values are consistent with the convolution formula.

---

## `CNNPoolingLayer`

Full name:

```cpp
LibMatchstick::CNNPoolingLayer
```

`CNNPoolingLayer` is the concrete pooling layer. It stores the pooling kernel size and the saved max-position information used by backpropagation.

Constructors:

```cpp
CNNPoolingLayer();

CNNPoolingLayer(
    size_t in_c,
    size_t in_h,
    size_t in_w,
    size_t out_c,
    size_t out_h,
    size_t out_w,
    size_t ker_h,
    size_t ker_w,
    size_t s,
    size_t p
);
```

Public API:

```cpp
Tensor3d forward(const Tensor3d& input);
Tensor3d backward(const Tensor3d& dl_da, float step);
```

For pooling layers, `ker_h` and `ker_w` are the pooling window size. The current API also asks the caller to provide `out_c`, `out_h`, and `out_w` explicitly, so the caller should keep them consistent with the pooling geometry.

---

# Networks

## `MLP`

Header:

```cpp
#include "network.h"
```

Full name:

```cpp
LibMatchstick::MLP
```

`MLP` is a multi-layer perceptron wrapper.

Internally, it stores:

- A vector of `MLPLayer`
- Learning rate
- Cross-entropy marker
- Loss function
- Loss derivative function

---

## Constructors

```cpp
MLP();
MLP(size_t layer_size, float step);
```

### `MLP()`

Creates an empty MLP.

### `MLP(size_t layer_size, float step)`

Creates an MLP with `layer_size` fully connected layers.

Parameters:

| Parameter | Description |
|---|---|
| `layer_size` | Number of fully connected layers |
| `step` | Learning rate |

---

## Set Layer

```cpp
void setLayer(size_t index, size_t in_size, size_t out_size);
```

Sets the shape of a layer.

Parameters:

| Parameter | Description |
|---|---|
| `index` | Layer index |
| `in_size` | Input vector size |
| `out_size` | Output vector size |

Example:

```cpp
MLP m(3, 0.01f);

m.setLayer(0, 784, 128);
m.setLayer(1, 128, 64);
m.setLayer(2, 64, 10);
```

---

## Set Layer Activation

```cpp
void setLayerActivation(
    size_t index,
    std::function<Matrix(const Matrix&)> a,
    std::function<Matrix(const Matrix&)> a_d
);
```

Sets the activation function and derivative for a specific layer.

Example:

```cpp
using namespace LibMatchstick::Activation;

m.setLayerActivation(0, relu, relu_d);
m.setLayerActivation(1, relu, relu_d);
m.setLayerActivation(2, softmax, softmax_d);
```

---

## Set Loss Function

```cpp
void setLoss(
    std::function<float(const Matrix&, const Matrix&)> loss,
    std::function<Matrix(const Matrix&, const Matrix&)> loss_d
);
```

Sets the loss function and its derivative.

Example:

```cpp
using namespace LibMatchstick::Loss;

m.setLoss(MSE, MSE_d);
```

or:

```cpp
m.setLoss(cross_entropy, cross_entropy_d);
```

---

## Training

```cpp
float train(const Matrix& input, const Matrix& expected, Matrix& l_dl_da);
```

Runs one training step.

Parameters:

| Parameter | Description |
|---|---|
| `input` | Input matrix, usually a column vector |
| `expected` | Expected output, usually a one-hot column vector |
| `l_dl_da` | Output parameter used to store the gradient with respect to the MLP input |

Returns the loss value for this training step.

`l_dl_da` is useful when connecting an MLP after CNN layers, because the gradient can be passed back into the convolution part.

---

## Inference

```cpp
Matrix use(const Matrix& input);
```

Runs forward inference.

Parameters:

| Parameter | Description |
|---|---|
| `input` | Input matrix |

Returns the network output.

---

## Softmax and Cross-Entropy Markers

```cpp
void setSm();
void setCe();
```

### `setSm`

Marks the last layer as a softmax layer.

### `setCe`

Marks the MLP as using cross entropy related logic.

When the last layer is softmax and the loss is cross entropy, the common simplified derivative is:

```text
dl/dz = y_hat - y
```

where `y_hat` is the network output and `y` is the expected one-hot target.

---

## Save and Load Parameters

```cpp
void loadWeight(size_t index, const Matrix& W);
void loadBias(size_t index, const Matrix& b);

Matrix saveWeight(size_t index) const;
Matrix saveBias(size_t index) const;
```

Saves or loads parameters of a specific layer.

Parameters:

| Parameter | Description |
|---|---|
| `index` | Layer index |
| `W` | Weight matrix |
| `b` | Bias matrix |

Example:

```cpp
Matrix w0 = m.saveWeight(0);
Matrix b0 = m.saveBias(0);

m.loadWeight(0, w0);
m.loadBias(0, b0);
```

---

## Initialization

```cpp
void init(float high = 1, float low = -1);
```

Initializes all layers.

Example:

```cpp
m.init(0.1f, -0.1f);
```

---

## `CNN`

Header:

```cpp
#include "network.h"
```

Full name:

```cpp
LibMatchstick::CNN
```

`CNN` is a convolutional neural network wrapper.

Internally, it stores:

- A vector of `CNNLayer`
- CNN learning rate
- An internal `MLP`

A typical structure is:

```text
Tensor3d input
    -> CNNLayer
    -> CNNLayer
    -> ...
    -> flatten
    -> MLP
    -> Matrix output
```

---

## Constructors

```cpp
CNN();
CNN(size_t layer_size, float step, size_t mlp_layer_size, float mlp_step);
```

### `CNN()`

Creates an empty CNN.

### `CNN(size_t layer_size, float step, size_t mlp_layer_size, float mlp_step)`

Creates a CNN with convolution layers and an internal MLP.

Parameters:

| Parameter | Description |
|---|---|
| `layer_size` | Number of convolution layers |
| `step` | Learning rate for convolution layers |
| `mlp_layer_size` | Number of layers in the internal MLP |
| `mlp_step` | Learning rate for the internal MLP |

---

## Access Internal MLP

```cpp
MLP& mlp();
const MLP& mlp() const;
```

Returns the internal MLP.

Example:

```cpp
CNN net(2, 0.001f, 2, 0.01f);

net.mlp().setLayer(0, 16 * 7 * 7, 128);
net.mlp().setLayer(1, 128, 10);
```

---

## Set Convolution Layer

```cpp
void setConvolutionLayer(
    size_t index,
    size_t in_c,
    size_t in_h,
    size_t in_w,
    size_t out_c,
    size_t out_h,
    size_t out_w,
    size_t k_c,
    size_t k_h,
    size_t k_w,
    size_t s,
    size_t p,
    std::function<Tensor3d(const Tensor3d&)> activation,
    std::function<Tensor3d(const Tensor3d&)> activation_d
);
```

Sets a convolution layer.

Parameters:

| Parameter | Description |
|---|---|
| `index` | Convolution layer index |
| `in_c` | Input channel count |
| `in_h` | Input height |
| `in_w` | Input width |
| `out_c` | Output channel count |
| `out_h` | Output height |
| `out_w` | Output width |
| `k_c` | Kernel channel count |
| `k_h` | Kernel height |
| `k_w` | Kernel width |
| `s` | Stride |
| `p` | Padding |

Example:

```cpp
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
```

---

## Set Pooling Layer

```cpp
void setPoolingLayer(
    size_t index,
    size_t in_c,
    size_t in_h,
    size_t in_w,
    size_t out_c,
    size_t out_h,
    size_t out_w,
    size_t ker_h,
    size_t ker_w,
    size_t s,
    size_t p
);
```

Sets a pooling layer.

---

## Training

```cpp
float train(const Tensor3d& input, const Matrix& expected);
```

Runs one CNN training step.

Parameters:

| Parameter | Description |
|---|---|
| `input` | Input tensor |
| `expected` | Expected output, usually a one-hot column vector |

Returns the loss value for this training step.

---

## Inference

```cpp
Matrix use(const Tensor3d& input);
```

Runs forward inference.

Parameters:

| Parameter | Description |
|---|---|
| `input` | Input tensor |

Returns the network output.

---

## Save and Load Convolution Parameters

```cpp
Tensor4d saveKernel(size_t index) const;
std::vector<float> saveBias(size_t index) const;

bool loadKernel(size_t index, const Tensor4d& k);
bool loadBias(size_t index, const std::vector<float>& b);
```

Saves or loads parameters of a specific convolution layer.

Parameters:

| Parameter | Description |
|---|---|
| `index` | Convolution layer index |
| `k` | Kernel tensor |
| `b` | Bias vector |

Example:

```cpp
Tensor4d k0 = net.saveKernel(0);
std::vector<float> b0 = net.saveBias(0);

net.loadKernel(0, k0);
net.loadBias(0, b0);
```

---

## Initialization

```cpp
void init(float high = 1, float low = -1);
```

Initializes convolution layers and the internal MLP.

Example:

```cpp
net.init(0.1f, -0.1f);
```

---

# Typical Usage

## MLP Example

```cpp
#include "matrix.h"
#include "activation.h"
#include "loss.h"
#include "network.h"

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

    std::cout << "loss = " << loss << std::endl;
    std::cout << output << std::endl;

    return 0;
}
```

---

## CNN Example

```cpp
#include "matrix.h"
#include "tensor_3d.h"
#include "activation.h"
#include "loss.h"
#include "network.h"

#include <iostream>

using namespace LibMatchstick;
using namespace LibMatchstick::Activation;
using namespace LibMatchstick::Loss;

int main() {
    CNN net(2, 0.001f, 2, 0.01f);

    // Input shape: 1 x 28 x 28
    // First convolution layer: 1 x 28 x 28 -> 8 x 14 x 14
    net.setConvolutionLayer(
        0,
        1, 28, 28,
        8, 14, 14,
        1, 3, 3,
        2, 1
    );

    // Second convolution layer: 8 x 14 x 14 -> 16 x 7 x 7
    net.setConvolutionLayer(
        1,
        8, 14, 14,
        16, 7, 7,
        8, 3, 3,
        2, 1
    );

    net.setLayerActivation(0, relu_t, relu_t_d);
    net.setLayerActivation(1, relu_t, relu_t_d);

    // Final CNN output after flatten: 16 * 7 * 7 = 784
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

# Shape Conventions

## Matrix

```text
height x width
```

MLP input and output are commonly column vectors:

```text
n x 1
```

---

## Tensor3d

```text
channel x height x width
```

Common CNN input examples:

```text
1 x 28 x 28    // grayscale MNIST-like image
3 x H x W      // RGB image
```

---

## Tensor4d

```text
batch x channel x height x width
```

For convolution kernels, it is usually interpreted as:

```text
out_channel x in_channel x kernel_height x kernel_width
```

---

## CNNLayer Output Shape

For standard convolution:

```text
out_h = floor((in_h + 2 * padding - k_h) / stride) + 1
out_w = floor((in_w + 2 * padding - k_w) / stride) + 1
```

The current API requires the caller to pass `out_h` and `out_w` manually.

---

# Memory and Object Lifetime

## RAII

`Matrix`, `Tensor3d`, and `Tensor4d` provide destructors and copy operations.

They can normally be returned by value:

```cpp
Matrix make_matrix() {
    Matrix a(10, 1);
    return a;
}
```

---

## `getData()` Notes

The following types expose `getData()`:

```cpp
Matrix
Tensor3d
Tensor4d
```

This function returns an internal `float*`.

Important notes:

1. Do not manually release the returned pointer.
2. Do not use the pointer after the object has been destroyed.
3. Do not assume the pointer is always CPU-accessible memory.
4. If the implementation uses CUDA device memory, use CUDA APIs to copy or modify the data.
5. Directly modifying the internal buffer bypasses class-level validation.

---

# Common Notes

## 1. Initialize Before Training

A typical setup order is:

```cpp
setLayer(...)
setLayerActivation(...)
setLoss(...)
init(...)
train(...)
```

Do not train before configuring the network structure, activations, and loss function.

---

## 2. CNN Shapes Must Be Consistent

Example:

```cpp
net.setConvolutionLayer(
    0,
    1, 28, 28,
    8, 14, 14,
    1, 3, 3,
    2, 1
);
```

The caller should ensure:

```text
14 = floor((28 + 2 * 1 - 3) / 2) + 1
```

Incorrect `out_h` or `out_w` values may break flattening, MLP input sizing, and backpropagation.

---

## 3. Compute Flatten Size Before Connecting MLP

If the final convolution output is:

```text
out_c x out_h x out_w
```

then the first MLP layer should usually receive:

```text
out_c * out_h * out_w
```

Example:

```text
16 x 7 x 7 = 784
```

So:

```cpp
net.mlp().setLayer(0, 784, 128);
```

---

## 4. Use Softmax and Cross Entropy Together for Classification

For classification tasks, a common setup is:

```cpp
net.setLayerActivation(last, softmax, softmax_d);
net.setLoss(cross_entropy, cross_entropy_d);
net.setSm();
net.setCe();
```

For `CNN`, configure the internal MLP similarly:

```cpp
net.mlp().setLayerActivation(last, softmax, softmax_d);
net.mlp().setLoss(cross_entropy, cross_entropy_d);
net.mlp().setSm();
net.mlp().setCe();
```

---

## 5. Check Shape Compatibility When Loading Parameters

Functions such as:

```cpp
loadWeight(...)
loadBias(...)
loadKernel(...)
```

expect compatible shapes.

For functions returning `bool`, check the return value:

```cpp
if (!layer.loadWeight(w)) {
    std::cerr << "failed to load weight" << std::endl;
}
```

`MLP::loadWeight` and `MLP::loadBias` return `void`, so callers should ensure that the target layer structure matches the saved parameters.

---

## 6. Prefer Public APIs Over Raw Pointers

Unless you are writing low-level CUDA kernels or serialization code, prefer:

```cpp
set(...)
get(...)
saveWeight()
loadWeight(...)
saveKernel()
loadKernel(...)
```

over direct access through `getData()`.

---

# Recommended Repository Layout

```text
LibMatchstick/
├── include/
│   ├── matrix.h
│   ├── tensor_3d.h
│   ├── tensor_4d.h
│   ├── activation.h
│   ├── loss.h
│   ├── layer.h
│   └── network.h
├── src/
├── docs/
│   └── API.md
├── README.md
├── LICENSE
├── COPYING
├── COPYING.LESSER
└── CMakeLists.txt
```

---

# License

LibMatchstick is licensed under the GNU Lesser General Public License v3.0 or later.

See:

- `LICENSE`
- `COPYING`
- `COPYING.LESSER`

