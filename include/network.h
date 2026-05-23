#ifndef NETWORK_C_H
#define NETWORK_C_H

#ifdef __cplusplus
extern "C"{
#endif

#include<stddef.h>
#include"matrix.h"
#include"tensor_3d.h"
#include"tensor_4d.h"

	struct matchstick_mlp_impl;

	typedef struct matchstick_mlp_impl matchstick_mlp_impl;

	typedef matchstick_mlp_impl*matchstick_mlp;

	enum activation{
		a_relu,
		a_leaky_relu,
		a_sigmoid,
		a_tanh,
		a_identity,
		a_softmax
	};

	enum loss{
		l_mse,
		l_mae,
		l_ce
	};

	matchstick_mlp init_matchstick_mlp(size_t layer_size,float step);

	void free_matchstick_mlp(matchstick_mlp a);

	void set_layer_matchstick_mlp(matchstick_mlp a,size_t index,size_t in_size,size_t out_size);

	void set_layer_activation_matchstick_mlp(matchstick_mlp a,size_t index,activation ac);

	void set_loss_matchstick_mlp(matchstick_mlp a,loss l);

	float train_matchsticl_mlp(
		matchstick_mlp a,
		matchstick_matrix input,
		matchstick_matrix expected,
		matchstick_matrix l_dl_da
	);

	matchstick_matrix use_matchstick_mlp(matchstick_mlp a,matchstick_matrix input);

	void set_sm_matchstick_mlp(matchstick_mlp a);

	void set_ce_matchstick_mlp(matchstick_mlp a);

	void load_weight_matchstick_mlp(matchstick_mlp a,size_t index,matchstick_matrix w);

	void load_bias_matchstick_mlp(matchstick_mlp a,size_t index,matchstick_matrix b);

	matchstick_matrix save_weight_matchstick_mlp(matchstick_mlp a,size_t index);

	matchstick_matrix save_bias_matchstick_mlp(matchstick_mlp a,size_t index);

	void shuffle_matchstic_mlp(matchstick_mlp a,float high,float low);


	struct matchstick_cnn_impl;

	typedef struct matchstick_cnn_impl matchstick_cnn_impl;

	typedef matchstick_cnn_impl*matchstick_cnn;

	matchstick_cnn init_matchstick_cnn(size_t layer_size,float step,size_t mlp_layer_size,float mlp_step);

	void free_matchstick_cnn(matchstick_cnn a);

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

	void set_layer_activation_matchstick_cnn(matchstick_cnn a,size_t index,activation ac);

	void set_loss_matchstick_cnn(matchstick_cnn a,loss l);

	float train_matchstick_cnn(matchstick_cnn a,matchstick_tensor_3d input,matchstick_matrix expected);

	matchstick_matrix use_matchstick_cnn(matchstick_cnn a,matchstick_tensor_3d input);

	void load_kernel_matchstick_cnn(matchstick_cnn a,size_t index,matchstick_tensor_4d w);

	void load_bias_matchstick_cnn(matchstick_cnn a,size_t index,float*b,size_t out_c);

	matchstick_tensor_4d save_kernel_matchstick_cnn(matchstick_cnn a,size_t index);

	float*save_bias_matchstick_cnn(matchstick_cnn a,size_t index,size_t out_c);

	void shuffle_matchstic_cnn(matchstick_cnn a,float high,float low);

	void set_layer_matchstick_cnn_mlp(matchstick_cnn a,size_t index,size_t in_size,size_t out_size);

	void set_layer_activation_matchstick_cnn_mlp(matchstick_cnn a,size_t index,activation ac);

	void set_sm_matchstick_cnn_mlp(matchstick_cnn a);

	void set_ce_matchstick_cnn_mlp(matchstick_cnn a);

	void load_weight_matchstick_cnn_mlp(matchstick_cnn a,size_t index,matchstick_matrix w);

	void load_bias_matchstick_cnn_mlp(matchstick_cnn a,size_t index,matchstick_matrix b);

	matchstick_matrix save_weight_matchstick_cnn_mlp(matchstick_cnn a,size_t index);

	matchstick_matrix save_bias_matchstick_cnn_mlp(matchstick_cnn a,size_t index);

#ifdef __cplusplus
}
#endif

#endif
