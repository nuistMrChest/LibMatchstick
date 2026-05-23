#include"../../internal/tensor_4d.h"
#include <cstdlib>

namespace LibMatchstick{
	Tensor4d::Tensor4d():
		b(0),
		c(0),
		h(0),
		w(0),
		data(nullptr)
	{}

	Tensor4d::Tensor4d(size_t b,size_t c,size_t h,size_t w):
		b(b),
		c(c),
		h(h),
		w(w)
	{
		cudaMalloc(&data,b*c*h*w*sizeof(float));
	}

	Tensor4d::Tensor4d(
		std::initializer_list<
			std::initializer_list<
				std::initializer_list<
					std::initializer_list<
						float
					>
				>
			>
		>a
	){
		b=a.size();
		c=a.begin()->size();
		h=a.begin()->begin()->size();
		w=a.begin()->begin()->begin()->size();
		cudaMalloc(&data,b*c*h*w*sizeof(float));
		size_t cnt=0;
		float*tmp=(float*)std::malloc(b*c*h*w*sizeof(float));
		for(auto i:a)for(auto j:i)for(auto k:j)for(auto l:k){
			tmp[cnt]=l;
			cnt++;
		}
		cudaMemcpy(data,tmp,b*c*h*w*sizeof(float),cudaMemcpyHostToDevice);
		free(tmp);
	}

	Tensor4d::~Tensor4d(){
		cudaFree(data);
	}

	size_t Tensor4d::getBatch()const{
		return b;
	}

	size_t Tensor4d::getChannel()const{
		return c;
	}

	size_t Tensor4d::getHeight()const{
		return h;
	}

	size_t Tensor4d::getWidth()const{
		return w;
	}

	void Tensor4d::resize(size_t b,size_t c,size_t h,size_t w){
		size_t size=b*c*h*w<this->b*this->c*this->h*this->w?b*c*h*w:this->b*this->c*this->h*this->w;
		this->b=b;
		this->c=c;
		this->h=h;
		this->w=w;
		float*tmp;
		cudaMalloc(&tmp,b*c*h*w*sizeof(float));
		if(data!=nullptr)cudaMemcpy(tmp,data,size*sizeof(float),cudaMemcpyDeviceToDevice);
		cudaFree(data);
		data=tmp;
	}

	Tensor4d::Tensor4d(const Tensor4d&a){
		this->b=a.b;
		this->c=a.c;
		this->h=a.h;
		this->w=a.w;
		cudaMalloc(&this->data,b*c*h*w*sizeof(float));
		cudaMemcpy(this->data,a.data,b*c*h*w*sizeof(float),cudaMemcpyDeviceToDevice);
	}

	Tensor4d&Tensor4d::operator=(const Tensor4d&a){
		if(this!=&a){
			this->b=a.b;
			this->c=a.c;
			this->h=a.h;
			this->w=a.w;
			cudaFree(this->data);
			cudaMalloc(&this->data,b*c*h*w*sizeof(float));
			cudaMemcpy(this->data,a.data,b*c*h*w*sizeof(float),cudaMemcpyDeviceToDevice);
		}
		return*this;
	}

	void Tensor4d::set(size_t i,size_t j,size_t k,size_t l,float v){
		if(i<b&&j<c&&k<h&&l<w)
			cudaMemcpy(data+i*c*h*w+j*h*w+k*w+l,&v,sizeof(float),cudaMemcpyHostToDevice);
	}

	float Tensor4d::get(size_t i,size_t j,size_t k,size_t l)const{
		float tmp=0;
		if(i<b&&j<c&&k<h&&l<w)
			cudaMemcpy(&tmp,data+i*c*h*w+j*h*w+k*w+l,sizeof(float),cudaMemcpyDeviceToHost);
		return tmp;
	}

	float*Tensor4d::getData(){
		return data;
	}

	const float*Tensor4d::getData()const{
		return data;
	}

}
