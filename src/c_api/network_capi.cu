#include"network_capi.h"
#include"matrix_capi.h"
#include"tensor_3d_capi.h"
#include"tensor_4d_capi.h"
#include"../../internal/activation.h"
#include"../../internal/loss.h"

matchstick_mlp init_matchstick_mlp(size_t layer_size,float step){
	matchstick_mlp a=new matchstick_mlp_impl();
	a->m=LibMatchstick::MLP(layer_size,step);
	return a;
}

void free_matchstick_mlp(matchstick_mlp a){
	delete a;
}

void set_layer_matchstick_mlp(matchstick_mlp a,size_t index,size_t in_size,size_t out_size){
	a->m.setLayer(index,in_size, out_size);
}

void set_layer_activation_matchstick_mlp(matchstick_mlp a,size_t index,activation ac){
	switch(ac){
		case matchstick_activation_relu:
			a->m.setLayerActivation(
				index,
				LibMatchstick::Activation::relu,
				LibMatchstick::Activation::relu_d
			);
			break;
		case matchstick_activation_leaky_relu:
			a->m.setLayerActivation(
				index,
				LibMatchstick::Activation::leaky_relu,
				LibMatchstick::Activation::leaky_relu_d
			);
			break;
		case matchstick_activation_sigmoid:
			a->m.setLayerActivation(
				index,
				LibMatchstick::Activation::sigmoid,
				LibMatchstick::Activation::sigmoid_d
			);
			break;
		case matchstick_activation_tanh:
			a->m.setLayerActivation(
				index,
				LibMatchstick::Activation::tanh,
				LibMatchstick::Activation::tanh_d
			);
			break;
		case matchstick_activation_identity:
			a->m.setLayerActivation(
				index,
				LibMatchstick::Activation::identity,
				LibMatchstick::Activation::identity_d
			);
			break;
		case matchstick_activation_softmax:
			a->m.setLayerActivation(
				index,
				LibMatchstick::Activation::softmax,
				LibMatchstick::Activation::softmax_d
			);
			break;
	}
}

void set_loss_matchstick_mlp(matchstick_mlp a,loss l){
	switch(l){
		case matchstick_loss_mse:
			a->m.setLoss(
				LibMatchstick::Loss::MSE,
				LibMatchstick::Loss::MSE_d
			);
			break;
		case matchstick_loss_mae:
			a->m.setLoss(
				LibMatchstick::Loss::MAE,
				LibMatchstick::Loss::MAE_d
			);
			break;
		case matchstick_loss_ce:
			a->m.setLoss(
				LibMatchstick::Loss::cross_entropy,
				LibMatchstick::Loss::cross_entropy_d
			);
			break;
	}
}

float train_matchstick_mlp(
	matchstick_mlp a,
	matchstick_matrix input,
	matchstick_matrix expected,
	matchstick_matrix l_dl_da
){
	return a->m.train(input->m,expected->m,l_dl_da->m);
}

matchstick_matrix use_matchstick_mlp(matchstick_mlp a,matchstick_matrix input){
	matchstick_matrix res=new matchstick_matrix_impl();
	res->m=a->m.use(input->m);
	return res;
}

void set_sm_matchstick_mlp(matchstick_mlp a){
	a->m.setSm();
}

void set_ce_matchstick_mlp(matchstick_mlp a){
	a->m.setCe();
}

void load_weight_matchstick_mlp(matchstick_mlp a,size_t index,matchstick_matrix w){
	a->m.loadWeight(index,w->m);
}

void load_bias_matchstick_mlp(matchstick_mlp a,size_t index,matchstick_matrix b){
	a->m.loadBias(index,b->m);
}

matchstick_matrix save_weight_matchstick_mlp(matchstick_mlp a,size_t index){
	matchstick_matrix res=new matchstick_matrix_impl();
	res->m=a->m.saveWeight(index);
	return res;
}

matchstick_matrix save_bias_matchstick_mlp(matchstick_mlp a,size_t index){
	matchstick_matrix res=new matchstick_matrix_impl();
	res->m=a->m.saveBias(index);
	return res;
}

void shuffle_matchstick_mlp(matchstick_mlp a,float high,float low){
	a->m.init(high,low);
}

matchstick_cnn init_matchstick_cnn(size_t layer_size,float step,size_t mlp_layer_size,float mlp_step){
	matchstick_cnn a=new matchstick_cnn_impl();
	a->c=LibMatchstick::CNN(layer_size,step,mlp_layer_size,mlp_step);
	return a;
}

void free_matchstick_cnn(matchstick_cnn a){
	delete a;
}

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
){
	a->c.setLayer(index,in_c,in_h,in_w,out_c,out_h,out_w,k_c,k_h,k_w,s,p);
}

void set_layer_activation_matchstick_cnn(matchstick_cnn a,size_t index,activation ac){
	switch(ac){
		case matchstick_activation_relu:
			a->c.setLayerActivation(
				index,
				LibMatchstick::Activation::relu_t,
				LibMatchstick::Activation::relu_t_d
			);
			break;
		case matchstick_activation_leaky_relu:
			a->c.setLayerActivation(
				index,
				LibMatchstick::Activation::leaky_relu_t,
				LibMatchstick::Activation::leaky_relu_t_d
			);
			break;
		case matchstick_activation_sigmoid:
			a->c.setLayerActivation(
				index,
				LibMatchstick::Activation::sigmoid_t,
				LibMatchstick::Activation::sigmoid_t_d
			);
			break;
		case matchstick_activation_tanh:
			a->c.setLayerActivation(
				index,
				LibMatchstick::Activation::tanh_t,
				LibMatchstick::Activation::tanh_t_d
			);
			break;
		case matchstick_activation_identity:
			a->c.setLayerActivation(
				index,
				LibMatchstick::Activation::identity_t,
				LibMatchstick::Activation::identity_t_d
			);
			break;
		case matchstick_activation_softmax:
			break;
	}
}

void set_loss_matchstick_cnn(matchstick_cnn a,loss l){
	switch(l){
		case matchstick_loss_mse:
			a->c.mlp().setLoss(
				LibMatchstick::Loss::MSE,
				LibMatchstick::Loss::MSE_d
			);
			break;
		case matchstick_loss_mae:
			a->c.mlp().setLoss(
				LibMatchstick::Loss::MAE,
				LibMatchstick::Loss::MAE_d
			);
			break;
		case matchstick_loss_ce:
			a->c.mlp().setLoss(
				LibMatchstick::Loss::cross_entropy,
				LibMatchstick::Loss::cross_entropy_d
			);
			break;
	}
}

float train_matchstick_cnn(matchstick_cnn a,matchstick_tensor_3d input,matchstick_matrix expected){
	return a->c.train(input->t,expected->m);
}

matchstick_matrix use_matchstick_cnn(matchstick_cnn a,matchstick_tensor_3d input){
	matchstick_matrix tmp=new matchstick_matrix_impl();
	tmp->m=a->c.use(input->t);
	return tmp;
}

void load_kernel_matchstick_cnn(matchstick_cnn a,size_t index,matchstick_tensor_4d w){
	a->c.loadKernel(index,w->t);
}

void load_bias_matchstick_cnn(matchstick_cnn a,size_t index,float*b,size_t out_c){
	a->c.loadBias(index, std::vector<float>(b,b+out_c));
}

matchstick_tensor_4d save_kernel_matchstick_cnn(matchstick_cnn a,size_t index){
	matchstick_tensor_4d tmp=new matchstick_tensor_4d_impl();
	tmp->t=a->c.saveKernel(index);
	return tmp;
}

float*save_bias_matchstick_cnn(matchstick_cnn a,size_t index,size_t out_c){
	float*tmp=(float*)malloc(out_c*sizeof(float));
	memcpy(tmp,a->c.saveBias(index).data(),out_c*sizeof(float));
	return tmp;
}

void shuffle_matchstick_cnn(matchstick_cnn a,float high,float low){
	a->c.init(high,low);
}

void set_layer_matchstick_cnn_mlp(matchstick_cnn a,size_t index,size_t in_size,size_t out_size){
	a->c.mlp().setLayer(index,in_size,out_size);
}

void set_layer_activation_matchstick_cnn_mlp(matchstick_cnn a,size_t index,activation ac){
	switch(ac){
		case matchstick_activation_relu:
			a->c.mlp().setLayerActivation(
				index,
				LibMatchstick::Activation::relu,
				LibMatchstick::Activation::relu_d
			);
			break;
		case matchstick_activation_leaky_relu:
			a->c.mlp().setLayerActivation(
				index,
				LibMatchstick::Activation::leaky_relu,
				LibMatchstick::Activation::leaky_relu_d
			);
			break;
		case matchstick_activation_sigmoid:
			a->c.mlp().setLayerActivation(
				index,
				LibMatchstick::Activation::sigmoid,
				LibMatchstick::Activation::sigmoid_d
			);
			break;
		case matchstick_activation_tanh:
			a->c.mlp().setLayerActivation(
				index,
				LibMatchstick::Activation::tanh,
				LibMatchstick::Activation::tanh_d
			);
			break;
		case matchstick_activation_identity:
			a->c.mlp().setLayerActivation(
				index,
				LibMatchstick::Activation::identity,
				LibMatchstick::Activation::identity_d
			);
			break;
		case matchstick_activation_softmax:
			a->c.mlp().setLayerActivation(
				index,
				LibMatchstick::Activation::softmax,
				LibMatchstick::Activation::softmax_d
			);
			break;
	}
}

void set_sm_matchstick_cnn_mlp(matchstick_cnn a){
	a->c.mlp().setSm();
}

void set_ce_matchstick_cnn_mlp(matchstick_cnn a){
	a->c.mlp().setCe();
}

void load_weight_matchstick_cnn_mlp(matchstick_cnn a,size_t index,matchstick_matrix w){
	a->c.mlp().loadWeight(index,w->m);
}

void load_bias_matchstick_cnn_mlp(matchstick_cnn a,size_t index,matchstick_matrix b){
	a->c.mlp().loadBias(index,b->m);
}

matchstick_matrix save_weight_matchstick_cnn_mlp(matchstick_cnn a,size_t index){
	matchstick_matrix tmp=new matchstick_matrix_impl();
	tmp->m=a->c.mlp().saveWeight(index);
	return tmp;
}

matchstick_matrix save_bias_matchstick_cnn_mlp(matchstick_cnn a,size_t index){
	matchstick_matrix tmp=new matchstick_matrix_impl();
	tmp->m=a->c.mlp().saveBias(index);
	return tmp;
}
