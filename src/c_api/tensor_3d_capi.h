#ifndef TENSOR_3D_CAPI_H
#define TENSOR_3D_CAPI_H

#include"../../include/matchstick_c/tensor_3d.h"
#include"../../include/matchstick/tensor_3d.h"

struct matchstick_tensor_3d_impl{
	LibMatchstick::Tensor3d t;
};

#endif
