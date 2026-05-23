# LibMatchstick C API Reference

This document describes the public C API exposed by LibMatchstick.

The recommended umbrella include is:

```c
#include "matchstick.h"
```

`matchstick.h` includes the matrix, 3D tensor, 4D tensor, and network APIs.

```c
#include "matrix.h"
#include "tensor_3d.h"
#include "tensor_4d.h"
#include "network.h"
```

---

## 1. Design Overview

LibMatchstick exposes an opaque-handle C API.

Most public object types are pointers to incomplete implementation structs:

```c
matchstick_matrix
matchstick_tensor_3d
matchstick_tensor_4d
matchstick_mlp
matchstick_cnn
```

Users should treat these values as handles. Do not access their internal fields, do not allocate them manually, and do not release them with plain `free()` unless this document explicitly says so.

---

## 2. Memory Ownership Rules

### 2.1 General Rule

Every LibMatchstick object returned as a `matchstick_*` handle must be released with its matching `free_matchstick_*` function.

Do **not** use standard `free()` on these handles.

### 2.2 Ownership Table

| Returned pointer type | Typical source functions | Correct release function |
|---|---|---|
| `matchstick_matrix` | `init_matchstick_matrix`, `use_matchstick_mlp`, `save_weight_matchstick_mlp`, `save_bias_matchstick_mlp`, `use_matchstick_cnn`, `save_weight_matchstick_cnn_mlp`, `save_bias_matchstick_cnn_mlp` | `free_matchstick_matrix()` |
| `matchstick_tensor_3d` | `init_matchstick_tensor_3d` | `free_matchstick_tensor_3d()` |
| `matchstick_tensor_4d` | `init_matchstick_tensor_4d`, `save_kernel_matchstick_cnn` | `free_matchstick_tensor_4d()` |
| `matchstick_mlp` | `init_matchstick_mlp` | `free_matchstick_mlp()` |
| `matchstick_cnn` | `init_matchstick_cnn` | `free_matchstick_cnn()` |
| `float *` | `save_bias_matchstick_cnn` | Standard C `free()` |

### 2.3 Special Case: `save_bias_matchstick_cnn`

`save_bias_matchstick_cnn()` returns a plain `float *`, not a LibMatchstick handle.

The returned bias buffer is a continuous heap allocation created with `malloc`. The user must release it with standard C `free()`.

```c
#include <stdlib.h>

float *bias = save_bias_matchstick_cnn(cnn, layer_index, out_c);

/* use bias[0] ... bias[out_c - 1] */

free(bias);
```

Do not pass this pointer to any `free_matchstick_*` function.

### 2.4 Input Pointers

Input arrays passed into initialization or loading functions, such as `float *v` or `float *b`, remain owned by the caller unless the implementation explicitly documents otherwise.

In other words, this API should be used as if `load_*` and `init_*` functions do **not** transfer ownership of caller-provided raw arrays.

---

# 3. Matrix API

Header:

```c
#include "matrix.h"
```

Type:

```c
struct matchstick_matrix_impl;
typedef struct matchstick_matrix_impl matchstick_matrix_impl;
typedef matchstick_matrix_impl *matchstick_matrix;
```

A `matchstick_matrix` is an opaque matrix handle.

---

## 3.1 `init_matchstick_matrix`

```c
matchstick_matrix init_matchstick_matrix(size_t h, size_t w, float *v);
```

Creates a matrix.

Parameters:

| Parameter | Description |
|---|---|
| `h` | Matrix height, usually the number of rows |
| `w` | Matrix width, usually the number of columns |
| `v` | Initial data buffer |

Returns:

A new `matchstick_matrix`.

Release with:

```c
free_matchstick_matrix(matrix);
```

Example:

```c
float data[] = {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f
};

matchstick_matrix m = init_matchstick_matrix(2, 3, data);

/* use m */

free_matchstick_matrix(m);
```

---

## 3.2 `free_matchstick_matrix`

```c
void free_matchstick_matrix(matchstick_matrix a);
```

Releases a matrix object created or returned by LibMatchstick.

---

## 3.3 `print_matchstick_matrix`

```c
void print_matchstick_matrix(matchstick_matrix a);
```

Prints a matrix to standard output.

---

## 3.4 `get_height_matchstick_matrix`

```c
size_t get_height_matchstick_matrix(matchstick_matrix a);
```

Returns the matrix height.

---

## 3.5 `get_width_matchstick_matrix`

```c
size_t get_width_matchstick_matrix(matchstick_matrix a);
```

Returns the matrix width.

---

## 3.6 `assignment_matchstick_matrix`

```c
void assignment_matchstick_matrix(matchstick_matrix to, matchstick_matrix from);
```

Assigns the contents of `from` to `to`.

The two matrices should have compatible dimensions.

---

# 4. Tensor 3D API

Header:

```c
#include "tensor_3d.h"
```

Type:

```c
struct matchstick_tensor_3d_impl;
typedef struct matchstick_tensor_3d_impl matchstick_tensor_3d_impl;
typedef matchstick_tensor_3d_impl *matchstick_tensor_3d;
```

A `matchstick_tensor_3d` is an opaque 3D tensor handle.

A typical layout is:

```text
channel × height × width
```

---

## 4.1 `init_matchstick_tensor_3d`

```c
matchstick_tensor_3d init_matchstick_tensor_3d(
    size_t c,
    size_t h,
    size_t w,
    float *v
);
```

Creates a 3D tensor.

Parameters:

| Parameter | Description |
|---|---|
| `c` | Number of channels |
| `h` | Tensor height |
| `w` | Tensor width |
| `v` | Initial data buffer |

Returns:

A new `matchstick_tensor_3d`.

Release with:

```c
free_matchstick_tensor_3d(tensor);
```

Example:

```c
float data[] = {
    1, 2,
    3, 4,

    5, 6,
    7, 8
};

matchstick_tensor_3d t = init_matchstick_tensor_3d(2, 2, 2, data);

/* use t */

free_matchstick_tensor_3d(t);
```

---

## 4.2 `free_matchstick_tensor_3d`

```c
void free_matchstick_tensor_3d(matchstick_tensor_3d a);
```

Releases a 3D tensor object.

---

## 4.3 `print_matchstick_tensor_3d`

```c
void print_matchstick_tensor_3d(matchstick_tensor_3d a);
```

Prints a 3D tensor to standard output.

---

## 4.4 `get_channel_matchstick_tensor_3d`

```c
size_t get_channel_matchstick_tensor_3d(matchstick_tensor_3d a);
```

Returns the channel count.

---

## 4.5 `get_height_matchstick_tensor_3d`

```c
size_t get_height_matchstick_tensor_3d(matchstick_tensor_3d a);
```

Returns the tensor height.

---

## 4.6 `get_width_matchstick_tensor_3d`

```c
size_t get_width_matchstick_tensor_3d(matchstick_tensor_3d a);
```

Returns the tensor width.

---

## 4.7 `assignment_matchstick_tensor_3d`

```c
void assignment_matchstick_tensor_3d(
    matchstick_tensor_3d to,
    matchstick_tensor_3d from
);
```

Assigns the contents of `from` to `to`.

The two tensors should have compatible dimensions.

---

# 5. Tensor 4D API

Header:

```c
#include "tensor_4d.h"
```

Type:

```c
struct matchstick_tensor_4d_impl;
typedef struct matchstick_tensor_4d_impl matchstick_tensor_4d_impl;
typedef struct matchstick_tensor_4d_impl *matchstick_tensor_4d;
```

A `matchstick_tensor_4d` is an opaque 4D tensor handle.

A generic layout is:

```text
batch × channel × height × width
```

For CNN kernels, it can be interpreted as:

```text
out_channel × in_channel × kernel_height × kernel_width
```

---

## 5.1 `init_matchstick_tensor_4d`

```c
matchstick_tensor_4d init_matchstick_tensor_4d(
    size_t b,
    size_t c,
    size_t h,
    size_t w,
    float *v
);
```

Creates a 4D tensor.

Parameters:

| Parameter | Description |
|---|---|
| `b` | Batch dimension |
| `c` | Channel dimension |
| `h` | Height dimension |
| `w` | Width dimension |
| `v` | Initial data buffer |

Returns:

A new `matchstick_tensor_4d`.

Release with:

```c
free_matchstick_tensor_4d(tensor);
```

---

## 5.2 `free_matchstick_tensor_4d`

```c
void free_matchstick_tensor_4d(matchstick_tensor_4d a);
```

Releases a 4D tensor object.

---

## 5.3 `get_batch_matchstick_tensor_4d`

```c
size_t get_batch_matchstick_tensor_4d(matchstick_tensor_4d a);
```

Returns the batch dimension.

---

## 5.4 `get_channel_matchstick_tensor_4d`

```c
size_t get_channel_matchstick_tensor_4d(matchstick_tensor_4d a);
```

Returns the channel dimension.

---

## 5.5 `get_height_matchstick_tensor_4d`

```c
size_t get_height_matchstick_tensor_4d(matchstick_tensor_4d a);
```

Returns the height dimension.

---

## 5.6 `get_width_matchstick_tensor_4d`

```c
size_t get_width_matchstick_tensor_4d(matchstick_tensor_4d a);
```

Returns the width dimension.

---

## 5.7 `assignment_matchstick_tensor_4d`

```c
void assignment_matchstick_tensor_4d(
    matchstick_tensor_4d to,
    matchstick_tensor_4d from
);
```

Assigns the contents of `from` to `to`.

The two tensors should have compatible dimensions.

---

# 6. Network Common Types

Header:

```c
#include "network.h"
```

---

## 6.1 Activation Functions

```c
typedef enum {
    matchstick_activation_relu,
    matchstick_activation_leaky_relu,
    matchstick_activation_sigmoid,
    matchstick_activation_tanh,
    matchstick_activation_identity,
    matchstick_activation_softmax
} activation;
```

Available activation values:

| Value | Meaning |
|---|---|
| `matchstick_activation_relu` | ReLU |
| `matchstick_activation_leaky_relu` | Leaky ReLU |
| `matchstick_activation_sigmoid` | Sigmoid |
| `matchstick_activation_tanh` | Tanh |
| `matchstick_activation_identity` | Identity |
| `matchstick_activation_softmax` | Softmax |

---

## 6.2 Loss Functions

```c
typedef enum {
    matchstick_loss_mse,
    matchstick_loss_mae,
    matchstick_loss_ce
} loss;
```

Available loss values:

| Value | Meaning |
|---|---|
| `matchstick_loss_mse` | Mean Squared Error |
| `matchstick_loss_mae` | Mean Absolute Error |
| `matchstick_loss_ce` | Cross Entropy |

---

# 7. MLP API

Type:

```c
struct matchstick_mlp_impl;
typedef struct matchstick_mlp_impl matchstick_mlp_impl;
typedef matchstick_mlp_impl *matchstick_mlp;
```

A `matchstick_mlp` is an opaque multilayer perceptron handle.

---

## 7.1 `init_matchstick_mlp`

```c
matchstick_mlp init_matchstick_mlp(size_t layer_size, float step);
```

Creates an MLP.

Parameters:

| Parameter | Description |
|---|---|
| `layer_size` | Number of MLP layers |
| `step` | Learning rate |

Returns:

A new `matchstick_mlp`.

Release with:

```c
free_matchstick_mlp(mlp);
```

---

## 7.2 `free_matchstick_mlp`

```c
void free_matchstick_mlp(matchstick_mlp a);
```

Releases an MLP object.

---

## 7.3 `set_layer_matchstick_mlp`

```c
void set_layer_matchstick_mlp(
    matchstick_mlp a,
    size_t index,
    size_t in_size,
    size_t out_size
);
```

Configures one MLP layer.

Parameters:

| Parameter | Description |
|---|---|
| `a` | MLP object |
| `index` | Layer index |
| `in_size` | Input size |
| `out_size` | Output size |

Example:

```c
matchstick_mlp net = init_matchstick_mlp(2, 0.01f);

set_layer_matchstick_mlp(net, 0, 784, 128);
set_layer_matchstick_mlp(net, 1, 128, 10);
```

---

## 7.4 `set_layer_activation_matchstick_mlp`

```c
void set_layer_activation_matchstick_mlp(
    matchstick_mlp a,
    size_t index,
    activation ac
);
```

Sets the activation function of one MLP layer.

Example:

```c
set_layer_activation_matchstick_mlp(
    net,
    0,
    matchstick_activation_relu
);

set_layer_activation_matchstick_mlp(
    net,
    1,
    matchstick_activation_softmax
);
```

---

## 7.5 `set_loss_matchstick_mlp`

```c
void set_loss_matchstick_mlp(matchstick_mlp a, loss l);
```

Sets the MLP loss function.

Example:

```c
set_loss_matchstick_mlp(net, matchstick_loss_ce);
```

---

## 7.6 `train_matchstick_mlp`

```c
float train_matchstick_mlp(
    matchstick_mlp a,
    matchstick_matrix input,
    matchstick_matrix expected,
    matchstick_matrix l_dl_da
);
```

Trains the MLP on one input sample or batch, depending on how the matrix dimensions are used by the implementation.

Parameters:

| Parameter | Description |
|---|---|
| `a` | MLP object |
| `input` | Input matrix |
| `expected` | Expected output matrix |
| `l_dl_da` | Loss gradient with respect to activation output |

Returns:

The loss value for this training call.

---

## 7.7 `use_matchstick_mlp`

```c
matchstick_matrix use_matchstick_mlp(
    matchstick_mlp a,
    matchstick_matrix input
);
```

Runs MLP inference.

Returns:

A newly allocated `matchstick_matrix`.

Release with:

```c
matchstick_matrix output = use_matchstick_mlp(net, input);

/* use output */

free_matchstick_matrix(output);
```

---

## 7.8 `set_sm_matchstick_mlp`

```c
void set_sm_matchstick_mlp(matchstick_mlp a);
```

Marks the MLP as using softmax-related behavior.

This is typically used when the final layer uses softmax.

---

## 7.9 `set_ce_matchstick_mlp`

```c
void set_ce_matchstick_mlp(matchstick_mlp a);
```

Marks the MLP as using cross-entropy-related behavior.

This is typically used with `matchstick_loss_ce`.

---

## 7.10 `load_weight_matchstick_mlp`

```c
void load_weight_matchstick_mlp(
    matchstick_mlp a,
    size_t index,
    matchstick_matrix w
);
```

Loads the weight matrix for one MLP layer.

The caller still owns `w` and must release it when appropriate.

---

## 7.11 `load_bias_matchstick_mlp`

```c
void load_bias_matchstick_mlp(
    matchstick_mlp a,
    size_t index,
    matchstick_matrix b
);
```

Loads the bias matrix for one MLP layer.

The caller still owns `b` and must release it when appropriate.

---

## 7.12 `save_weight_matchstick_mlp`

```c
matchstick_matrix save_weight_matchstick_mlp(
    matchstick_mlp a,
    size_t index
);
```

Saves and returns the weight matrix of one MLP layer.

Returns:

A newly allocated `matchstick_matrix`.

Release with:

```c
free_matchstick_matrix(w);
```

---

## 7.13 `save_bias_matchstick_mlp`

```c
matchstick_matrix save_bias_matchstick_mlp(
    matchstick_mlp a,
    size_t index
);
```

Saves and returns the bias matrix of one MLP layer.

Returns:

A newly allocated `matchstick_matrix`.

Release with:

```c
free_matchstick_matrix(b);
```

---

## 7.14 `shuffle_matchstick_mlp`

```c
void shuffle_matchstick_mlp(
    matchstick_mlp a,
    float high,
    float low
);
```

Randomizes MLP parameters within a range controlled by `high` and `low`.

---

# 8. CNN API

Type:

```c
struct matchstick_cnn_impl;
typedef struct matchstick_cnn_impl matchstick_cnn_impl;
typedef matchstick_cnn_impl *matchstick_cnn;
```

A `matchstick_cnn` is an opaque convolutional neural network handle.

The CNN API also exposes functions for configuring the internal MLP part of the CNN.

---

## 8.1 `init_matchstick_cnn`

```c
matchstick_cnn init_matchstick_cnn(
    size_t layer_size,
    float step,
    size_t mlp_layer_size,
    float mlp_step
);
```

Creates a CNN.

Parameters:

| Parameter | Description |
|---|---|
| `layer_size` | Number of convolution layers |
| `step` | Learning rate for the convolution part |
| `mlp_layer_size` | Number of layers in the internal MLP part |
| `mlp_step` | Learning rate for the internal MLP part |

Returns:

A new `matchstick_cnn`.

Release with:

```c
free_matchstick_cnn(cnn);
```

---

## 8.2 `free_matchstick_cnn`

```c
void free_matchstick_cnn(matchstick_cnn a);
```

Releases a CNN object.

---

## 8.3 `set_layer_matchstick_cnn`

```c
void set_layer_matchstick_cnn(
    matchstick_cnn a,
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
    size_t p
);
```

Configures one convolution layer.

Parameters:

| Parameter | Description |
|---|---|
| `a` | CNN object |
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

```c
set_layer_matchstick_cnn(
    cnn,
    0,
    1, 28, 28,
    8, 24, 24,
    1, 5, 5,
    1,
    0
);
```

---

## 8.4 `set_layer_activation_matchstick_cnn`

```c
void set_layer_activation_matchstick_cnn(
    matchstick_cnn a,
    size_t index,
    activation ac
);
```

Sets the activation function of one convolution layer.

---

## 8.5 `set_loss_matchstick_cnn`

```c
void set_loss_matchstick_cnn(matchstick_cnn a, loss l);
```

Sets the CNN loss function.

---

## 8.6 `train_matchstick_cnn`

```c
float train_matchstick_cnn(
    matchstick_cnn a,
    matchstick_tensor_3d input,
    matchstick_matrix expected
);
```

Trains the CNN on one input sample.

Parameters:

| Parameter | Description |
|---|---|
| `a` | CNN object |
| `input` | Input 3D tensor |
| `expected` | Expected output matrix |

Returns:

The loss value for this training call.

---

## 8.7 `use_matchstick_cnn`

```c
matchstick_matrix use_matchstick_cnn(
    matchstick_cnn a,
    matchstick_tensor_3d input
);
```

Runs CNN inference.

Returns:

A newly allocated `matchstick_matrix`.

Release with:

```c
matchstick_matrix output = use_matchstick_cnn(cnn, input);

/* use output */

free_matchstick_matrix(output);
```

---

## 8.8 `load_kernel_matchstick_cnn`

```c
void load_kernel_matchstick_cnn(
    matchstick_cnn a,
    size_t index,
    matchstick_tensor_4d w
);
```

Loads the convolution kernel tensor for one CNN layer.

The caller still owns `w` and must release it when appropriate.

---

## 8.9 `load_bias_matchstick_cnn`

```c
void load_bias_matchstick_cnn(
    matchstick_cnn a,
    size_t index,
    float *b,
    size_t out_c
);
```

Loads the convolution bias for one CNN layer from a continuous `float` buffer.

Parameters:

| Parameter | Description |
|---|---|
| `a` | CNN object |
| `index` | Convolution layer index |
| `b` | Bias buffer |
| `out_c` | Number of bias values, usually equal to output channels |

The caller still owns `b`.

---

## 8.10 `save_kernel_matchstick_cnn`

```c
matchstick_tensor_4d save_kernel_matchstick_cnn(
    matchstick_cnn a,
    size_t index
);
```

Saves and returns the convolution kernel tensor of one CNN layer.

Returns:

A newly allocated `matchstick_tensor_4d`.

Release with:

```c
matchstick_tensor_4d kernel = save_kernel_matchstick_cnn(cnn, index);

/* use kernel */

free_matchstick_tensor_4d(kernel);
```

---

## 8.11 `save_bias_matchstick_cnn`

```c
float *save_bias_matchstick_cnn(
    matchstick_cnn a,
    size_t index,
    size_t out_c
);
```

Saves and returns the convolution bias of one CNN layer.

Returns:

A continuous heap-allocated `float *` buffer.

Release with standard C `free()`:

```c
#include <stdlib.h>

float *bias = save_bias_matchstick_cnn(cnn, index, out_c);

/* use bias */

free(bias);
```

Important:

This is not a LibMatchstick handle. Do not release it with `free_matchstick_matrix`, `free_matchstick_tensor_3d`, `free_matchstick_tensor_4d`, or `free_matchstick_cnn`.

---

## 8.12 `shuffle_matchstick_cnn`

```c
void shuffle_matchstick_cnn(
    matchstick_cnn a,
    float high,
    float low
);
```

Randomizes CNN parameters within a range controlled by `high` and `low`.

---

# 9. CNN Internal MLP API

These functions configure and manipulate the MLP part inside a `matchstick_cnn`.

---

## 9.1 `set_layer_matchstick_cnn_mlp`

```c
void set_layer_matchstick_cnn_mlp(
    matchstick_cnn a,
    size_t index,
    size_t in_size,
    size_t out_size
);
```

Configures one internal MLP layer.

Parameters:

| Parameter | Description |
|---|---|
| `a` | CNN object |
| `index` | Internal MLP layer index |
| `in_size` | Input size |
| `out_size` | Output size |

---

## 9.2 `set_layer_activation_matchstick_cnn_mlp`

```c
void set_layer_activation_matchstick_cnn_mlp(
    matchstick_cnn a,
    size_t index,
    activation ac
);
```

Sets the activation function of one internal MLP layer.

---

## 9.3 `set_sm_matchstick_cnn_mlp`

```c
void set_sm_matchstick_cnn_mlp(matchstick_cnn a);
```

Marks the internal MLP as using softmax-related behavior.

---

## 9.4 `set_ce_matchstick_cnn_mlp`

```c
void set_ce_matchstick_cnn_mlp(matchstick_cnn a);
```

Marks the internal MLP as using cross-entropy-related behavior.

---

## 9.5 `load_weight_matchstick_cnn_mlp`

```c
void load_weight_matchstick_cnn_mlp(
    matchstick_cnn a,
    size_t index,
    matchstick_matrix w
);
```

Loads the weight matrix for one internal MLP layer.

The caller still owns `w`.

---

## 9.6 `load_bias_matchstick_cnn_mlp`

```c
void load_bias_matchstick_cnn_mlp(
    matchstick_cnn a,
    size_t index,
    matchstick_matrix b
);
```

Loads the bias matrix for one internal MLP layer.

The caller still owns `b`.

---

## 9.7 `save_weight_matchstick_cnn_mlp`

```c
matchstick_matrix save_weight_matchstick_cnn_mlp(
    matchstick_cnn a,
    size_t index
);
```

Saves and returns the weight matrix of one internal MLP layer.

Returns:

A newly allocated `matchstick_matrix`.

Release with:

```c
free_matchstick_matrix(w);
```

---

## 9.8 `save_bias_matchstick_cnn_mlp`

```c
matchstick_matrix save_bias_matchstick_cnn_mlp(
    matchstick_cnn a,
    size_t index
);
```

Saves and returns the bias matrix of one internal MLP layer.

Returns:

A newly allocated `matchstick_matrix`.

Release with:

```c
free_matchstick_matrix(b);
```

---

# 10. Example: MLP

```c
#include "matchstick.h"
#include <stdio.h>

int main(void) {
    matchstick_mlp net = init_matchstick_mlp(2, 0.01f);

    set_layer_matchstick_mlp(net, 0, 2, 4);
    set_layer_matchstick_mlp(net, 1, 4, 2);

    set_layer_activation_matchstick_mlp(net, 0, matchstick_activation_relu);
    set_layer_activation_matchstick_mlp(net, 1, matchstick_activation_softmax);

    set_loss_matchstick_mlp(net, matchstick_loss_ce);
    set_sm_matchstick_mlp(net);
    set_ce_matchstick_mlp(net);

    shuffle_matchstick_mlp(net, 1.0f, -1.0f);

    float x_data[] = {1.0f, 0.0f};
    float y_data[] = {0.0f, 1.0f};
    float grad_data[] = {0.0f, 0.0f};

    matchstick_matrix x = init_matchstick_matrix(2, 1, x_data);
    matchstick_matrix y = init_matchstick_matrix(2, 1, y_data);
    matchstick_matrix grad = init_matchstick_matrix(2, 1, grad_data);

    float loss = train_matchstick_mlp(net, x, y, grad);
    printf("loss = %f\n", loss);

    matchstick_matrix out = use_matchstick_mlp(net, x);
    print_matchstick_matrix(out);

    free_matchstick_matrix(out);
    free_matchstick_matrix(grad);
    free_matchstick_matrix(y);
    free_matchstick_matrix(x);

    free_matchstick_mlp(net);

    return 0;
}
```

---

# 11. Example: CNN Bias Save and Release

```c
#include "matchstick.h"
#include <stdlib.h>

int main(void) {
    matchstick_cnn cnn = init_matchstick_cnn(
        1,
        0.01f,
        1,
        0.01f
    );

    set_layer_matchstick_cnn(
        cnn,
        0,
        1, 28, 28,
        8, 24, 24,
        1, 5, 5,
        1,
        0
    );

    shuffle_matchstick_cnn(cnn, 1.0f, -1.0f);

    size_t out_c = 8;

    float *bias = save_bias_matchstick_cnn(cnn, 0, out_c);

    /*
        bias is a malloc-allocated continuous float buffer.
        The user must release it with standard free().
    */

    free(bias);

    free_matchstick_cnn(cnn);

    return 0;
}
```

---

# 12. Safety Checklist

Before releasing user code that calls LibMatchstick, check the following:

- Every `matchstick_matrix` is released with `free_matchstick_matrix`.
- Every `matchstick_tensor_3d` is released with `free_matchstick_tensor_3d`.
- Every `matchstick_tensor_4d` is released with `free_matchstick_tensor_4d`.
- Every `matchstick_mlp` is released with `free_matchstick_mlp`.
- Every `matchstick_cnn` is released with `free_matchstick_cnn`.
- The result of `save_bias_matchstick_cnn` is released with standard `free()`.
- No opaque `matchstick_*` handle is released with plain `free()`.
- Dimensions passed to layer setup, load, save, and assignment functions are consistent.
- Objects returned by `use_*` and `save_*` functions are manually released after use.
- Caller-owned input arrays are not assumed to be released by the library.

---

# 13. Notes on ABI Usage

This API is intended to be consumed from C and other languages that can call C ABI functions.

The public headers expose opaque pointer handles instead of C++ classes. This keeps the public ABI simpler and avoids exposing C++ implementation details to C users.

For C++ users, the headers use `extern "C"` when included from C++, so the exported function names remain C ABI compatible.

