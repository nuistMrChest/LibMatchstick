#include"tensor_4d_capi.h"

matchstick_tensor_4d init_matchstick_tensor_4d(size_t b,size_t c,size_t h,size_t w,float*v){
	matchstick_tensor_4d a=new matchstick_tensor_4d_impl();
	a->t=LibMatchstick::Tensor4d(b,c,h,w);
	cudaMemcpy(a->t.getData(),v,b*c*h*w*sizeof(float),cudaMemcpyHostToDevice);
	return a;
}

void free_matchstick_tensor_4d(matchstick_tensor_4d a){
	delete a;
}

size_t get_batch_matchstick_tensor_4d(matchstick_tensor_4d a){
	return a->t.getBatch();
}

size_t get_channel_matchstick_tensor_4d(matchstick_tensor_4d a){
	return a->t.getChannel();
}

size_t get_height_matchstick_tensor_4d(matchstick_tensor_4d a){
	return a->t.getHeight();
}

size_t get_width_matchstick_tensor_4d(matchstick_tensor_4d a){
	return a->t.getWidth();
}

void assignment_matchstick_tensor_4d(matchstick_tensor_4d to,matchstick_tensor_4d from){
	to->t=from->t;
}
