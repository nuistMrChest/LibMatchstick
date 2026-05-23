#ifndef TENSOR_4D_CAPI_H
#define TENSOR_4D_CAPI_H

#include"../../include/matchstick/tensor_4d.h"
#include"../../include/matchstick_c/tensor_4d.h"

struct matchstick_tensor_4d_impl{
	LibMatchstick::Tensor4d t;
};

#endif
