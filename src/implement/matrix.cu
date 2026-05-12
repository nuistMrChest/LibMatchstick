#include"../matrix.h"
#include <cstdlib>
#include<cuda_runtime.h>
#include<stdlib.h>
#include<string.h>

namespace LibMatchstick{
	Matrix::Matrix(){
		h=0;
		w=0;
		data=nullptr;
	}

	Matrix::Matrix(size_t h,size_t w){
		this->h=h;
		this->w=w;
		cudaMalloc(&data,h*w*sizeof(float));
	}

	Matrix::Matrix(std::initializer_list<std::initializer_list<float>>a){
		this->h=a.size();
		this->w=a.begin()->size();
		cudaMalloc(&data,h*w*sizeof(float));
		size_t cnt=0;
		float*tmp=(float*)malloc(h*w*sizeof(float));
		for(auto i:a)for(auto j:i){
			tmp[cnt]=j;
			cnt++;
		}
		cudaMemcpy(data,tmp,h*w*sizeof(float),cudaMemcpyHostToDevice);
		free(tmp);
	}

	Matrix::~Matrix(){
		cudaFree(data);
	}

	std::ostream&operator<<(std::ostream&os,const Matrix&a){
		float*tmp=(float*)malloc(a.w*a.h*sizeof(float));
		cudaMemcpy(tmp,a.data,a.w*a.h*sizeof(float),cudaMemcpyDeviceToHost);
		for(size_t i=0;i<a.h;i++){
			if(i==0)os<<"{ ";
			else os<<"  ";
			for(size_t j=0;j<a.w;j++){
				os<<tmp[i*a.w+j]<<" ";
			}
			if(i==a.h-1)os<<"}";
			else os<<"\n";
		}
		free(tmp);
		return os;
	}

	size_t Matrix::getHeight()const{
		return h;
	}

	size_t Matrix::getWidth()const{
		return w;
	}

	void Matrix::resize(size_t h,size_t w){
		this->w=w;
		this->h=h;
		float*tmp;
		cudaMalloc(&tmp,h*w*sizeof(float));
		if(data!=nullptr)cudaMemcpy(tmp,data,h*w*sizeof(float),cudaMemcpyDeviceToDevice);
		cudaFree(data);
		data=tmp;
	}

	Matrix::Matrix(const Matrix&a){
		this->w=a.w;
		this->h=a.h;
		cudaMalloc(&this->data,w*h*sizeof(float));
		cudaMemcpy(this->data,a.data,w*h*sizeof(float),cudaMemcpyDeviceToDevice);
	}

	Matrix&Matrix::operator=(const Matrix&a){
		if(this!=&a){
			this->w=a.w;
			this->h=a.h;
			cudaFree(this->data);
			cudaMalloc(&this->data,w*h*sizeof(float));
			cudaMemcpy(this->data,a.data,w*h*sizeof(float),cudaMemcpyDeviceToDevice);
		}
		return*this;
	}

	size_t __device__ flatten_index(size_t x,size_t y,size_t w){
		return x*w+y;
	}

	void __global__ scalor_add(
		float*res,
		float*const left,
		float*const right,
		size_t h,
		size_t w
	){
		size_t i=blockIdx.x*blockDim.x+threadIdx.x;
		if(i<h*w)
			res[i]=
				left[i]+
				right[i];
	}

	Matrix Matrix::operator+(const Matrix&a)const{
		Matrix res;
		if(this->h==a.h&&this->w==a.w){
			res.resize(a.h,a.w);
			size_t bs=256;
			size_t gs=(a.h*a.w+bs-1)/bs;
			scalor_add<<<gs,bs>>>(res.data,data,a.data,h,w);
		}
		return res;
	}

	void __global__ scalor_sub(
		float*res,
		float*const left,
		float*const right,
		size_t h,
		size_t w
	){
		size_t i=blockIdx.x*blockDim.x+threadIdx.x;
		if(i<h*w)
			res[i]=
				left[i]-
				right[i];
	}

	Matrix Matrix::operator-(const Matrix&a)const{
		Matrix res;
		if(this->h==a.h&&this->w==a.w){
			res.resize(a.h,a.w);
			size_t bs=256;
			size_t gs=(a.h*a.w+bs-1)/bs;
			scalor_sub<<<gs,bs>>>(res.data,data,a.data,h,w);
		}
		return res;
	}

	Matrix&Matrix::operator+=(const Matrix&a){
		if(this->h==a.h&&this->w==a.w){
			size_t bs=256;
			size_t gs=(a.h*a.w+bs-1)/bs;
			scalor_add<<<gs,bs>>>(data,data,a.data,h,w);
		}
		return*this;
	}

	Matrix&Matrix::operator-=(const Matrix&a){
		if(this->h==a.h&&this->w==a.w){
			size_t bs=256;
			size_t gs=(a.h*a.w+bs-1)/bs;
			scalor_sub<<<gs,bs>>>(data,data,a.data,h,w);
		}
		return*this;
	}

	void __global__ scalor_mul(
		float*res,
		float*const left,
		float*const right,
		size_t h,
		size_t w
	){
		size_t i=blockIdx.x*blockDim.x+threadIdx.x;
		if(i<h*w)
			res[i]=
				left[i]*
				right[i];
	}

	Matrix Matrix::hadamard(const Matrix&a)const{
		Matrix res;
		if(this->h==a.h&&this->w==a.w){
			res.resize(a.h,a.w);
			size_t bs=256;
			size_t gs=(a.h*a.w+bs-1)/bs;
			scalor_mul<<<gs,bs>>>(res.data,data,a.data,h,w);
		}
		return res;
	}

	void __global__ vector_dot(
		float*res,
		float*const left,
		float*const right,
		size_t h,
		size_t w,
		size_t k
	){
		size_t i=blockIdx.x*blockDim.x+threadIdx.x;
		size_t j=blockIdx.y*blockDim.y+threadIdx.y;
		if(i<h&&j<w){
			res[flatten_index(i,j,w)]=0;
			for(size_t x=0;x<k;x++)
				res[flatten_index(i,j,w)]+=
					left[flatten_index(i,x,k)]*
					right[flatten_index(x,j,w)];
		}
	}

	Matrix Matrix::operator*(const Matrix&a)const{
		Matrix res;
		if(this->w==a.h){
			res.resize(this->h,a.w);
			dim3 block(16,16);
			dim3 grid(
				(res.h+block.x-1)/block.x,
				(res.w+block.y-1)/block.y
			);
			vector_dot<<<grid,block>>>(res.data,data,a.data,res.h,res.w,this->w);
		}
		return res;
	}

	void Matrix::set(size_t i,size_t j,float v){
		if(i<h&&j<w)
			cudaMemcpy(data+i*w+j,&v,sizeof(float),cudaMemcpyHostToDevice);
	}

	float Matrix::get(size_t i,size_t j)const{
		float tmp=0;
		if(i<h&&j<w)
			cudaMemcpy(&tmp,data+i*w+j,sizeof(float),cudaMemcpyDeviceToHost);
		return tmp;
	}

	float*Matrix::getData(){
		return data;
	}
}
