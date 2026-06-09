#ifndef TENSOR_4D_C_H
#define TENSOR_4D_C_H

#include "tensor_3d.h"
#ifdef __cplusplus
extern "C"{
#endif

	#include<stddef.h>

	struct matchstick_tensor_4d_impl;

	typedef struct matchstick_tensor_4d_impl matchstick_tensor_4d_impl;

	typedef struct matchstick_tensor_4d_impl*matchstick_tensor_4d;

	matchstick_tensor_4d init_matchstick_tensor_4d(size_t b,size_t c,size_t h,size_t w,float*v);

	void free_matchstick_tensor_4d(matchstick_tensor_4d a);

	size_t get_batch_matchstick_tensor_4d(matchstick_tensor_4d a);

	size_t get_channel_matchstick_tensor_4d(matchstick_tensor_4d a);

	size_t get_height_matchstick_tensor_4d(matchstick_tensor_4d a);

	size_t get_width_matchstick_tensor_4d(matchstick_tensor_4d a);

	void assignment_matchstick_tensor_4d(matchstick_tensor_4d to,matchstick_tensor_4d from);

	float get_matchstick_tensor_4d(matchstick_tensor_4d a,size_t i,size_t j,size_t k,size_t l);

#ifdef __cplusplus
}
#endif

#endif
