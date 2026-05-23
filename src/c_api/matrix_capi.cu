#include"matrix_capi.h"
#include"iostream"

matchstick_matrix init_matcistick_matrix(size_t h,size_t w,float*v){
	matchstick_matrix a=new matchstick_matrix_impl();
	a->m=LibMatchstick::Matrix(h,w);
	cudaMemcpy(a->m.getData(),v,h*w*sizeof(float),cudaMemcpyHostToDevice);
	return a;
}

void free_matchstick_matrix(matchstick_matrix a){
	delete a;
}

void print_matchstick_matrix(matchstick_matrix a){
	std::cout<<a->m;
}


size_t get_height_matchstick_matrix(matchstick_matrix a){
	return a->m.getHeight();
}

size_t get_width_matchstick_matrix(matchstick_matrix a){
	return a->m.getWidth();
}

void assignment_matcistich_matrix(matchstick_matrix to,matchstick_matrix from){
	to->m=from->m;
}
