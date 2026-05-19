#include"../tensor_3d.h"
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include"../tensor_4d.h"

namespace LibMatchstick{
	Tensor3d::Tensor3d():
		c(0),
		h(0),
		w(0),
		data(nullptr)
	{}

	Tensor3d::Tensor3d(size_t c,size_t h,size_t w):
		c(c),
		h(h),
		w(w)
	{
		cudaMalloc(&data,c*h*w*sizeof(float));
	}

	Tensor3d::Tensor3d(std::initializer_list<std::initializer_list<std::initializer_list<float>>>a){
		c=a.size();
		h=a.begin()->size();
		w=a.begin()->begin()->size();
		cudaMalloc(&data,c*h*w*sizeof(float));
		float*tmp=(float*)std::malloc(c*h*w*sizeof(float));
		size_t cnt=0;
		for(auto i:a)for(auto j:i)for(auto k:j){
			tmp[cnt]=k;
			cnt++;
		}
		cudaMemcpy(data,tmp,c*h*w*sizeof(float),cudaMemcpyHostToDevice);
		free(tmp);
	}

	Tensor3d::~Tensor3d(){
		cudaFree(data);
	}

	std::ostream&operator<<(std::ostream&os,const Tensor3d&a){
		if(a.c==0){
				os<<"{ NULL }";
			}
			for(size_t i=0;i<a.c;i++){
				if(i==0)os<<"{\n";
				else os<<" ";
				for(size_t j=0;j<a.h;j++){
					if(j==0)os<<"{\n";
					else os<<"  ";
					for(size_t k=0;k<a.w;k++){
						os<<a.data[i*a.h+j*a.w+k]<<" ";
					}
					if(j==a.h-1)os<<"\n }";
					else os<<"\n";
				}
				if(i==a.c-1)os<<"\n}";
				else os<<"\n";
			}
		return os;
	}

	size_t Tensor3d::getChannel()const{
		return c;
	}

	size_t Tensor3d::getHeight()const{
		return h;
	}

	size_t Tensor3d::getWidth()const{
		return w;
	}

	void Tensor3d::resize(size_t c,size_t h,size_t w){
		size_t size=c*h*w<this->c*this->h*this->w?c*h*w:this->c*this->h*this->w;
		this->c=c;
		this->h=h;
		this->w=w;
		float*tmp;
		cudaMalloc(&tmp,c*h*w*sizeof(float));
		if(data!=nullptr)cudaMemcpy(data,tmp,size*sizeof(float),cudaMemcpyDeviceToDevice);
		cudaFree(data);
		data=tmp;
	}

	Tensor3d::Tensor3d(const Tensor3d&a){
		this->c=a.c;
		this->h=a.h;
		this->w=a.w;
		cudaMalloc(&this->data,c*h*w*sizeof(float));
		cudaMemcpy(this->data,a.data,c*h*w*sizeof(float),cudaMemcpyDeviceToDevice);
	}

	Tensor3d&Tensor3d::operator=(const Tensor3d&a){
		if(this!=&a){
			this->c=a.c;
			this->h=a.h;
			this->w=a.w;
			cudaFree(this->data);
			cudaMalloc(&this->data,c*h*w*sizeof(float));
			cudaMemcpy(this->data,a.data,c*h*w*sizeof(float),cudaMemcpyDeviceToDevice);
		}
		return*this;
	}

	void __global__ scalor_add(
		float*res,
		float*const left,
		float*const right,
		size_t c,
		size_t h,
		size_t w
	){
		size_t i=blockIdx.x*blockDim.x+threadIdx.x;
		size_t j=blockIdx.y*blockDim.y+threadIdx.y;
		size_t k=blockIdx.z*blockDim.z+threadIdx.z;
		if(c>i&&h>j&&w>k)
			res[i*h*w+j*w+k]=left[i*h*w+j*w+k]+right[i*h*w+j*w+k];
	}

	Tensor3d Tensor3d::operator+(const Tensor3d&a)const{
		Tensor3d res;
		if(c==a.c&&h==a.h&&w==a.w){
			res.resize(c,h,w);
			dim3 block(8,8,8);
			dim3 grid(
				(c+block.x-1)/block.x,
				(h+block.y-1)/block.y,
				(w+block.z-1)/block.z
			);
			scalor_add<<<grid,block>>>(res.data,data,a.data,c,h,w);
		}
		return res;
	}

	void __global__ scalor_sub(
		float*res,
		float*const left,
		float*const right,
		size_t c,
		size_t h,
		size_t w
	){
		size_t i=blockIdx.x*blockDim.x+threadIdx.x;
		size_t j=blockIdx.y*blockDim.y+threadIdx.y;
		size_t k=blockIdx.z*blockDim.z+threadIdx.z;
		if(c>i&&h>j&&w>k)
			res[i*h*w+j*w+k]=left[i*h*w+j*w+k]-right[i*h*w+j*w+k];
	}

	Tensor3d Tensor3d::operator-(const Tensor3d&a)const{
		Tensor3d res;
		if(c==a.c&&h==a.h&&w==a.w){
			res.resize(c,h,w);
			dim3 block(8,8,8);
			dim3 grid(
				(c+block.x-1)/block.x,
				(h+block.y-1)/block.y,
				(w+block.z-1)/block.z
			);
			scalor_sub<<<grid,block>>>(res.data,data,a.data,c,h,w);
		}
		return res;
	}

	Tensor3d&Tensor3d::operator+=(const Tensor3d&a){
		if(c==a.c&&h==a.h&&w==a.w){
			dim3 block(8,8,8);
			dim3 grid(
				(c+block.x-1)/block.x,
				(h+block.y-1)/block.y,
				(w+block.z-1)/block.z
			);
			scalor_add<<<grid,block>>>(data,data,a.data,c,h,w);
		}
		return*this;
	}

	Tensor3d&Tensor3d::operator-=(const Tensor3d&a){
		if(c==a.c&&h==a.h&&w==a.w){
			dim3 block(8,8,8);
			dim3 grid(
				(c+block.x-1)/block.x,
				(h+block.y-1)/block.y,
				(w+block.z-1)/block.z
			);
			scalor_sub<<<grid,block>>>(data,data,a.data,c,h,w);
		}
		return*this;
	}

	void __global__ scalor_mul(
		float*res,
		float*const left,
		float*const right,
		size_t c,
		size_t h,
		size_t w
	){
		size_t i=blockIdx.x*blockDim.x+threadIdx.x;
		size_t j=blockIdx.y*blockDim.y+threadIdx.y;
		size_t k=blockIdx.z*blockDim.z+threadIdx.z;
		if(c>i&&h>j&&w>k)
			res[i*h*w+j*w+k]=left[i*h*w+j*w+k]*right[i*h*w+j*w+k];
	}

	Tensor3d Tensor3d::hadamard(const Tensor3d&a)const{
		Tensor3d res;
		if(c==a.c&&h==a.h&&w==a.w){
			res.resize(c,h,w);
			dim3 block(8,8,8);
			dim3 grid(
				(c+block.x-1)/block.x,
				(h+block.y-1)/block.y,
				(w+block.z-1)/block.z
			);
			scalor_mul<<<grid,block>>>(res.data,data,a.data,c,h,w);
		}
		return res;
	}

	void Tensor3d::set(size_t i,size_t j,size_t k,float v){
		if(i<c&&j<h&&k<w)
			cudaMemcpy(data+i*h*w+j*w+k,&v,sizeof(float),cudaMemcpyHostToDevice);
	}

	float Tensor3d::get(size_t i,size_t j,size_t k)const{
		float tmp=0;
		if(i<c&&j<h&&k<w)
			cudaMemcpy(&tmp,data+i*h*w+j*w+k,sizeof(float),cudaMemcpyDeviceToHost);
		return tmp;
	}

	float*Tensor3d::getData(){
		return data;
	}

	const float*Tensor3d::getData()const{
		return data;
	}

	void __global__ dot(
		float*res,
		const float*ten,
		const float*ker,
		size_t r_c,
		size_t r_h,
		size_t r_w,
		size_t stride,
		size_t padding,
		size_t k_c,
		size_t k_h,
		size_t k_w,
		size_t t_c,
		size_t t_h,
		size_t t_w
	){
		size_t i=blockIdx.x*blockDim.x+threadIdx.x;
		size_t j=blockIdx.y*blockDim.y+threadIdx.y;
		size_t k=blockIdx.z*blockDim.z+threadIdx.z;
		if(i<r_c&&j<r_h&&k<r_w){
			res[r_h*r_w*i+r_w*j+k]=0;
			size_t x=j*stride;
			size_t y=k*stride;
			for(size_t ii=0;ii<k_c;ii++)
				for(size_t jj=0;jj<k_h;jj++)
					for(size_t kk=0;kk<k_w;kk++)
						if(
							!(
								(long long)(x+jj)-
								(long long)padding<
								0||
								(long long)(y+kk)-
								(long long)(padding)<
								0||
								(long long)(x+jj)>=
								(long long)(t_h)||
								(long long)(y+kk)-
								(long long)(padding)>=
								(long long)(k_w)
							)
						)
							res[r_h*r_w*i+r_w*j+k]+=
								ten[ii*t_h*t_w+(x+jj-padding)*t_w+(y+kk-padding)]*
								ker[i*k_c*k_h*k_w+ii*k_h*k_w+jj*k_w+kk];
		}
	}

	Tensor3d Tensor3d::convolution(const Tensor4d&k,size_t stride,size_t padding)const{
		Tensor3d res(
			k.getBatch(),
			(h+2*padding-k.getHeight())/stride+1,
			(w+2*padding-k.getWidth())/stride+1
		);
		dim3 block(8,8,8);
		dim3 grid(
			(c+block.x-1)/block.x,
			(h+block.y-1)/block.y,
			(w+block.z-1)/block.z
		);
		dot<<<grid,block>>>(
				res.getData(),
				data,
				k.getData(),
				res.getChannel(),
				res.getHeight(),
				res.getWidth(),
				stride,
				padding,
				k.getChannel(),
				k.getHeight(),
				k.getWidth(),
				c,
				h,
				w
		);
		return res;
	}
}
