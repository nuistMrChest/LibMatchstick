#ifndef NETWORK_CAPI_H
#define NETWORK_CAPI_H

#include"../../include/network.h"
#include"../../internal/network.h"

struct matchstick_mlp_impl{
	LibMatchstick::MLP m;
};

struct matchstick_cnn_impl{
	LibMatchstick::CNN c;
};

#endif
