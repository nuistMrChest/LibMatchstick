#include"../network.h"
#include"../matrix.h"
#include"../layer.h"
#include"../losses.h"
#include"../tensor_3d.h"
#include"../tensor_4d.h"

namespace LibMatchstick{
	MLP::MLP():
		ce(false),
		step(0)
	{
		loss=Losses::MSE;
		loss_d=Losses::MSE_d;
	}

	MLP::MLP(size_t layer_size,float step):
		ce(false),
		step(step)
	{
		layers.resize(layer_size);
		loss=Losses::MSE;
		loss_d=Losses::MSE_d;
	}

	void MLP::setLayer(size_t index,size_t in_size,size_t out_size){
		layers[index]=MLPLayer(in_size,out_size);
	}

	void MLP::setLayerActivation(
		size_t index,
		std::function<Matrix(const Matrix&)>a,
		std::function<Matrix(const Matrix&)>a_d
	){
		layers[index].setActivation(a,a_d);
	}

	void MLP::setLoss(
		std::function<float(const Matrix&,const Matrix&)>loss,
		std::function<Matrix(const Matrix&,const Matrix&)>loss_d
	){
		this->loss=loss;
		this->loss_d=loss_d;
	}

	float MLP::train(const Matrix&input,const Matrix&expected,Matrix&l_dl_da){
		float res;
		Matrix last_output=input;
		Matrix output;
		for(size_t i=0;i<layers.size();i++){
			output=layers[i].forward(last_output);
			last_output=output;
		}
		res=loss(output,expected);
		Matrix last_grad;
		if(layers.back().isSm()&&ce){
			Matrix dl_dz=output-expected;
			last_grad=layers.back().backward_dz(dl_dz,step);
			l_dl_da=last_grad;
			for(size_t i=0;i<layers.size()-1;i++){
				size_t j=layers.size()-2-i;
				last_grad=layers[j].backward(last_grad,step);
				if(i==layers.size()-2)l_dl_da=last_grad;
			}
		}
		else{
			Matrix last_dl_da=loss_d(output,expected);
			for(size_t i=0;i<layers.size();i++){
				size_t j=layers.size()-1-i;
				last_dl_da=layers[j].backward(last_dl_da,step);
				if(i==layers.size()-1)l_dl_da=last_dl_da;
			}
		}
		return res;
	}

	Matrix MLP::use(const Matrix&input){
		Matrix res;
		Matrix last_output=input;
		Matrix output;
		for(size_t i=0;i<layers.size();i++){
			output=layers[i].forward(last_output);
			last_output=output;
		}
		res=output;
		return res;
	}

	void MLP::setSm(){
		layers[layers.size()-1].setSm();
	}

	void MLP::setCe(){
		ce=true;
	}

	void MLP::loadWeight(size_t index,const Matrix&W){
		layers[index].loadWeight(W);
	}

	void MLP::loadBias(size_t index,const Matrix&b){
		layers[index].loadBias(b);
	}

	Matrix MLP::saveWeight(size_t index)const{
		return layers[index].saveWeight();
	}

	Matrix MLP::saveBias(size_t index)const{
		return layers[index].saveBias();
	}

	void MLP::init(float high,float low){
		for(size_t i=0;i<layers.size();i++)
			layers[i].init(high,low);
	}

	CNN::CNN():
		step(0)
	{}

	CNN::CNN(size_t layer_size,float step,size_t mlp_layer_size,float mlp_step):
		step(step),
		m(MLP(mlp_layer_size,mlp_step))
	{
		layers.resize(layer_size);
	}

	MLP&CNN::mlp(){
		return m;
	}

	const MLP&CNN::mlp()const{
		return m;
	}

	void CNN::setLayer(
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
		layers[index]=CNNLayer(
			in_c,
			in_h,
			in_w,
			out_c,
			out_h,
			out_w,
			k_c,
			k_h,
			k_w,
			s,
			p
		);
	}

	void CNN::setLayerActivation(
		size_t index,
		std::function<Tensor3d(const Tensor3d&)>activation,
		std::function<Tensor3d(const Tensor3d&)>activation_d
	){
		layers[index].setActivation(activation,activation_d);
	}

	float CNN::train(const Tensor3d&input,const Matrix&expected){
		float res;
		Tensor3d last_input=input;
		Tensor3d output;
		for(size_t i=0;i<layers.size();i++){
			output=layers[i].forward(last_input);
			last_input=output;
		}
		Matrix m_l_dl_da;
		res=m.train(output.flatten(),expected,m_l_dl_da);
		Tensor3d last_dl_da=Tensor3d::deflatten(
			m_l_dl_da,
			layers.back().getOutChannel(),
			layers.back().getOutHeight(),
			layers.back().getOutWidth()
		);
		for(size_t i=0;i<layers.size();i++){
			size_t j=layers.size()-1-i;
			last_dl_da=layers[j].backward(last_dl_da,step);
		}
		return res;
	}

	Matrix CNN::use(const Tensor3d&input){
		Tensor3d last_input=input;
		Tensor3d output;
		for(size_t i=0;i<layers.size();i++){
			output=layers[i].forward(last_input);
			last_input=output;
		}
		return m.use(output.flatten());
	}

	Tensor4d CNN::saveKernel(size_t index)const{
		return layers[index].saveKernel();
	}

	std::vector<float>CNN::saveBias(size_t index)const{
		return layers[index].saveBias();
	}

	bool CNN::loadKernel(size_t index,const Tensor4d&k){
		return layers[index].loadKernel(k);
	}

	bool CNN::loadBias(size_t index,const std::vector<float>&b){
		return layers[index].loadBias(b);
	}

	void CNN::init(float high,float low){
		for(size_t i=0;i<layers.size();i++)
			layers[i].init(high,low);
	}
}

