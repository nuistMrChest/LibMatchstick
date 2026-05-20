#include"../layer.h"
#include"../matrix.h"
#include"../activation.h"
#include"../tensor_3d.h"
#include"../tensor_4d.h"
#include <cstdio>
#include<memory>
#include<random>

namespace LibMatchstick{
	MLPLayer::MLPLayer():
		sm(false),
		in_size(0),
		out_size(0),
		activation(Activation::identity),
		activation_d(Activation::identity_d)
	{}

	MLPLayer::MLPLayer(size_t in_size,size_t out_size):
		sm(false),
		in_size(in_size),
		out_size(out_size),
		activation(Activation::identity),
		activation_d(Activation::identity_d)
	{
		W=std::make_unique<Matrix>(out_size,in_size);
		b=std::make_unique<Matrix>(out_size,1);
		last_input=std::make_unique<Matrix>(in_size,1);
		z=std::make_unique<Matrix>(out_size,1);
	}

	Matrix MLPLayer::forward(const Matrix&input){
		Matrix res(out_size,1);
		*last_input=input;
		*z=((*W*input)+*b);
		res=activation(*z);
		return res;
	}

	Matrix MLPLayer::backward(const Matrix&dl_da,float step){
		Matrix res;
		Matrix dl_dz=dl_da.hadamard(activation_d(*z));
		res=W->transpose()*dl_dz;
		*W-=step*(dl_dz*last_input->transpose());
		*b-=step*dl_dz;
		return res;
	}

	Matrix MLPLayer::backward_dz(const Matrix&dl_dz,float step){
		Matrix res=W->transpose()*dl_dz;
		*W-=step*(dl_dz*last_input->transpose());
		*b-=step*dl_dz;
		return res;
	}

	void MLPLayer::init(float low,float high){
		float*tmp_W=(float*)malloc(W->getWidth()*W->getHeight()*sizeof(float));
		float*tmp_b=(float*)malloc(b->getWidth()*b->getHeight()*sizeof(float));
		static std::mt19937 rng(std::random_device{}());
		std::uniform_real_distribution<float>dist(low,high);
		for(size_t i=0;i<out_size;i++){
			for(size_t j=0;j<in_size;j++)
				tmp_W[i*in_size+j]=dist(rng);
			tmp_b[i]=dist(rng);
		}
		cudaMemcpy(W->getData(),tmp_W,W->getHeight()*W->getWidth()*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(b->getData(),tmp_b,b->getHeight()*b->getWidth()*sizeof(float),cudaMemcpyHostToDevice);
		free(tmp_W);
		free(tmp_b);
	}

	Matrix MLPLayer::saveWeight()const{
		return*W;
	}

	Matrix MLPLayer::saveBias()const{
		return*b;
	}

	bool MLPLayer::loadWeight(const Matrix&W){
		if(W.getHeight()==this->W->getHeight()&&W.getWidth()==this->W->getWidth()){
			*(this->W)=W;
			return true;
		}
		return false;
	}

	bool MLPLayer::loadBias(const Matrix&b){
		if(b.getHeight()==this->b->getHeight()&&b.getWidth()==this->b->getWidth()){
			*(this->b)=b;
			return true;
		}
		return false;
	}

	void MLPLayer::setActivation(
		const std::function<Matrix(const Matrix&)>&a,
		const std::function<Matrix(const Matrix&)>&a_d
	){
		activation=a;
		activation_d=a_d;
	}

	CNNLayer::CNNLayer():
		b(nullptr),
		in_c(0),
		in_h(0),
		in_w(0),
		out_c(0),
		out_h(0),
		out_w(0),
		stride(0),
		padding(0)
	{
		activation=Activation::identity_t;
		activation_d=Activation::identity_t_d;
	}

	CNNLayer::CNNLayer(
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
	):
		in_c(in_c),
		in_h(in_h),
		in_w(in_w),
		out_c(out_c),
		out_h(out_h),
		out_w(out_w),
		stride(s),
		padding(p)
	{
		cudaMalloc(&b,out_c*sizeof(float));
		kernel=std::make_unique<Tensor4d>(out_c,k_c,k_h,k_w);
		last_input=std::make_unique<Tensor3d>(in_c,in_h,in_w);
		z=std::make_unique<Tensor3d>(out_c,out_h,out_w);
		activation=Activation::identity_t;
		activation_d=Activation::identity_t_d;
	}

	void CNNLayer::init(float low,float high){
		float*tmp_k=(float*)malloc(
			kernel->getBatch()*
			kernel->getChannel()*
			kernel->getWidth()*
			kernel->getHeight()*
			sizeof(float)
		);
		float*tmp_b=(float*)malloc(out_c*sizeof(float));
		static std::mt19937 rng(std::random_device{}());
		std::uniform_real_distribution<float>dist(low,high);
		for(size_t i=0;i<out_c;i++){
			for(size_t j=0;j<kernel->getChannel();j++)
				for(size_t k;k<kernel->getHeight();k++)
					for(size_t l=0;l<kernel->getWidth();l++)
						tmp_k[
							i*kernel->getChannel()*kernel->getHeight()*kernel->getWidth()+
							j*kernel->getHeight()*kernel->getWidth()+
							k*kernel->getWidth()+
							l
						]=dist(rng);
			tmp_b[i]=dist(rng);
		}
		cudaMemcpy(
			kernel->getData(),
			tmp_k,
			kernel->getBatch()*kernel->getChannel()*kernel->getHeight()*kernel->getWidth()*sizeof(float),
			cudaMemcpyHostToDevice
		);
		cudaMemcpy(b,tmp_b,out_c*sizeof(float),cudaMemcpyHostToDevice);
		free(tmp_k);
		free(tmp_b);
	}

	Tensor3d CNNLayer::forward(const Tensor3d&input){
		Tensor3d res;
		if(input.getChannel()==in_c&&input.getHeight()==in_h&&in_w==input.getWidth()){
			*last_input=input;
			*z=input.convolution(*kernel,stride,padding);
			//here!!!!
		}
		return res;
	}
}
