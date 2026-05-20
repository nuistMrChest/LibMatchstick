#ifndef ACTIVATION_H
#define ACTIVATION_H

#include"matrix.h"
#include"tensor_3d.h"

namespace LibMatchstick{
	namespace Activation{
		Matrix relu(const Matrix&a);
		Matrix relu_d(const Matrix&a);
		Matrix leaky_relu(const Matrix&a);
		Matrix leaky_relu_d(const Matrix&a);
		Matrix sigmoid(const Matrix&a);
		Matrix sigmoid_d(const Matrix&a);
		Matrix tanh(const Matrix&a);
		Matrix tanh_d(const Matrix&a);
		Matrix identity(const Matrix&a);
		Matrix identity_d(const Matrix&a);
		Matrix softmax(const Matrix&a);
		Matrix softmax_d(const Matrix&a);

		Tensor3d relu_t(const Tensor3d&a);
		Tensor3d relu_t_d(const Tensor3d&a);
		Tensor3d leaky_relu_t(const Tensor3d&a);
		Tensor3d leaky_relu_t_d(const Tensor3d&a);
		Tensor3d sigmoid_t(const Tensor3d&a);
		Tensor3d sigmoid_t_d(const Tensor3d&a);
		Tensor3d tanh_t(const Tensor3d&a);
		Tensor3d tanh_t_d(const Tensor3d&a);
		Tensor3d identity_t(const Tensor3d&a);
		Tensor3d identity_t_d(const Tensor3d&a);
	}
}

#endif
