#ifndef TENSOR_3D_C_H
#define TENSOR_3D_C_H

#ifdef __cplusplus
extern "C"{
#endif

	#include<stddef.h>

	struct matchstick_tensor_3d_impl;

	typedef struct matchstick_tensor_3d_impl matchstick_tensor_3d_impl;

	typedef matchstick_tensor_3d_impl*matchstick_tensor_3d;

	matchstick_tensor_3d init_matchstick_tensor_3d(size_t c,size_t h,size_t w,float*v);

	void free_matchstick_tensor_3d(matchstick_tensor_3d a);

	void print_matchstick_tensor_3d(matchstick_tensor_3d a);

	size_t get_channel_matchstick_tensor_3d(matchstick_tensor_3d a);

	size_t get_height_matchstick_tensor_3d(matchstick_tensor_3d a);

	size_t get_width_matchstick_tensor_3d(matchstick_tensor_3d a);

	void assignment_matchstick_tensor_3d(matchstick_tensor_3d to,matchstick_tensor_3d from);

#ifdef __cplusplus
}
#endif

#endif

