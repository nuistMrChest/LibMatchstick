#include<iostream>
#include"tensor_3d_capi.h"

matchstick_tensor_3d init_matchstick_tensor_3d(size_t c,size_t h,size_t w,float*v){
	matchstick_tensor_3d a=new matchstick_tensor_3d_impl();
	a->t=LibMatchstick::Tensor3d(c,h,w);
	cudaMemcpy(a->t.getData(),v,c*h*w*sizeof(float),cudaMemcpyHostToDevice);
	return a;
}

void free_matchstick_tensor_3d(matchstick_tensor_3d a){
	delete a;
}

void print_matchstick_tensor_3d(matchstick_tensor_3d a){
	std::cout<<a->t;
}

size_t get_channel_matchstick_tensor_3d(matchstick_tensor_3d a){
	return a->t.getChannel();
}

size_t get_height_matchstick_tensor_3d(matchstick_tensor_3d a){
	return a->t.getHeight();
}

size_t get_width_matchstick_tensor_3d(matchstick_tensor_3d a){
	return a->t.getWidth();
}

void assignment_matchstick_tensor_3d(matchstick_tensor_3d to,matchstick_tensor_3d from){
	to->t=from->t;
}

float get_matchstick_tensor_3d(matchstick_tensor_3d a,size_t i,size_t j,size_t k){
	return a->t.get(i,j,k);
}
