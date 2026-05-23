#ifndef MATRIX_C_H
#define MATRIX_C_H

#ifdef __cplusplus
extern "C"{
#endif

	#include<stddef.h>

	struct matchstick_matrix_impl;

	typedef struct matchstick_matrix_impl matchstick_matrix_impl;

	typedef matchstick_matrix_impl*matchstick_matrix;

	matchstick_matrix init_matchstick_matrix(size_t h,size_t w,float*v);

	void free_matchstick_matrix(matchstick_matrix a);

	void print_matchstick_matrix(matchstick_matrix a);

	size_t get_height_matchstick_matrix(matchstick_matrix a);

	size_t get_width_matchstick_matrix(matchstick_matrix a);

	void assignment_matchstich_matrix(matchstick_matrix to,matchstick_matrix from);

#ifdef __cplusplus
}
#endif

#endif
