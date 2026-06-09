#ifndef NETWORK_CAPI_H
#define NETWORK_CAPI_H

#include"../../include/matchstick_c/network.h"
#include"../../include/matchstick/network.h"

struct matchstick_mlp_impl{
	LibMatchstick::MLP m;
	matchstick_mlp_impl(size_t layer_size,float step)
		:m(layer_size,step)
	{}
};

struct matchstick_cnn_impl{
	LibMatchstick::CNN c;
	matchstick_cnn_impl(
		size_t layer_size,
		float step,
		size_t mlp_layer_size,
		float mlp_step
	)
		:c(layer_size,step,mlp_layer_size,mlp_step)
	{}
};

#endif
