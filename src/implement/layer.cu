#include"../layer.h"
#include"../matrix.h"
#include <memory>
#include <random>

namespace LibMatchstick{
	MLPLayer::MLPLayer():
		sm(false),
		in_size(0),
		out_size(0)
	{}

	MLPLayer::MLPLayer(size_t in_size,size_t out_size):
		sm(false),
		in_size(in_size),
		out_size(out_size)
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
		*(this->W)=W;
	}

	bool MLPLayer::loadBias(const Matrix&b){
		*(this->b)=b;
	}
}
