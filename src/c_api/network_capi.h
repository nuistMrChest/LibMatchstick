#ifndef NETWORK_CAPI_H
#define NETWORK_CAPI_H

#include"../../include/matchstick_c/network.h"
#include"../../include/matchstick/network.h"

struct matchstick_mlp_impl{
	LibMatchstick::MLP m;
};

struct matchstick_cnn_impl{
	LibMatchstick::CNN c;
};

#endif
