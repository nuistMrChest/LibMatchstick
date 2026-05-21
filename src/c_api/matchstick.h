#ifndef MATCHSTICK_H
#define MATCHSTICK_H

#ifdef __cplusplus
extern "C"{
#endif

#include<stddef.h>

	typedef struct{
		float*data;
		size_t height;
		size_t width;
	}matrix_tmp;

	void free_matrix_tmp(matrix_tmp*a);

	typedef void*matrix;

	matrix*materialize_matrix(matrix_tmp*a);
	matrix_tmp virtualize_matrix(matrix*a);

	typedef struct{
		float*data;
		size_t channel;
		size_t height;
		size_t width;
	}tensor3d_tmp;

	void free_tensor3d_tmp(tensor3d_tmp*a);

	typedef void*tensor3d;

	tensor3d*materialize_tensor3d(tensor3d_tmp*a);
	tensor3d_tmp virtualize_tensor3d(tensor3d*a);

	typedef struct{
		float*data;
		size_t batch;
		size_t channel;
		size_t height;
		size_t width;
	}tensor4d_tmp;

	void free_tensot4d_tmp(tensor4d_tmp);

	typedef void*tensor4d;

	tensor4d*materialize_tensor4d(tensor4d_tmp*a);
	tensor4d_tmp virtualize_tensor4d(tensor4d*a);

	typedef void*mlp;

	mlp init_mlp(size_t layer_size,float step);
	void free_mlp(mlp a);

	void set_mlp_layer(mlp a,size_t index,size_t in_size,size_t out_size);

	enum activation{
		relu,
		leaky_relu,
		sigmoid,
		tanh,
		softmax,
		identity
	};

	void set_mlp_layer_activation(mlp a,size_t index,activation act);

	enum losses{
		MSE,
		MAE,
		CE
	};

	void set_mlp_loss(mlp a,losses l);

	void set_mlp_smce(mlp a);

	

#ifdef __cplusplus
}
#endif

#endif
