#include"../activation.h"
#include <__clang_cuda_builtin_vars.h>

namespace LibMatchstick{
	namespace Actication{
		float __device__ scalor_relu(float*from,float*to,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			if(i<h&&j<w)
				to[i*w+j]=from[i*w+j]>0?from[i*w+j]:0;
		}

		Matrix relu(const Matrix&a){
			Matrix res(a.getHeight(),a.getWidth());
			dim3 block(16,16);
			dim3 grid(
				(res.getHeight()+block.x-1)/block.x,
				(res.getWidth()+block.y-1)/block.y
			);
			scalor_relu<<<grid,block>>>(a.data,res.data,res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_relu_d(float*from,float*to,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			if(i<h&&j<w)
				to[i*w+j]=from[i*w+j]>0?1:0;
		}

		Matrix relu_d(const Matrix&a){
			Matrix res(a.getHeight(),a.getWidth());
			dim3 block(16,16);
			dim3 grid(
				(res.getHeight()+block.x-1)/block.x,
				(res.getWidth()+block.y-1)/block.y
			);
			scalor_relu_d<<<grid,block>>>(a.data,res.data,res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_leaky_relu(float*from,float*to,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			if(i<h&&j<w)
				to[i*w+j]=from[i*w+j]>0?from[i*w+j]:0.01;
		}

		Matrix leaky_relu(const Matrix&a){
			Matrix res(a.getHeight(),a.getWidth());
			dim3 block(16,16);
			dim3 grid(
				(res.getHeight()+block.x-1)/block.x,
				(res.getWidth()+block.y-1)/block.y
			);
			scalor_leaky_relu<<<grid,block>>>(a.data,res.data,res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_leaky_relu_d(float*from,float*to,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			if(i<h&&j<w)
				to[i*w+j]=from[i*w+j]>0?1:0.01;
		}

		Matrix leaky_relu_d(const Matrix&a){
			Matrix res(a.getHeight(),a.getWidth());
			dim3 block(16,16);
			dim3 grid(
				(res.getHeight()+block.x-1)/block.x,
				(res.getWidth()+block.y-1)/block.y
			);
			scalor_leaky_relu_d<<<grid,block>>>(a.data,res.data,res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_sigmoid(float*from,float*to,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			if(i<h&&j<w)
				to[i*w+j]=1/(1+__expf(-1*from[i*w+j]));
		}

		Matrix sigmoid(const Matrix&a){
			Matrix res(a.getHeight(),a.getWidth());
			dim3 block(16,16);
			dim3 grid(
				(res.getHeight()+block.x-1)/block.x,
				(res.getWidth()+block.y-1)/block.y
			);
			scalor_sigmoid<<<grid,block>>>(a.data,res.data,res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_sigmoid_d(float*from,float*to,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			if(i<h&&j<w)
				to[i*w+j]=from[i*w+j]*(1-from[i*w+j]);
		}

		Matrix sigmoid_d(const Matrix&a){
			Matrix res(a.getHeight(),a.getWidth());
			dim3 block(16,16);
			dim3 grid(
				(res.getHeight()+block.x-1)/block.x,
				(res.getWidth()+block.y-1)/block.y
			);
			scalor_sigmoid_d<<<grid,block>>>(a.data,res.data,res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_tanh(float*from,float*to,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			if(i<h&&j<w)
				to[i*w+j]=tanhf(from[i*w+j]);
		}

		Matrix tanh(const Matrix&a){
			Matrix res(a.getHeight(),a.getWidth());
			dim3 block(16,16);
			dim3 grid(
				(res.getHeight()+block.x-1)/block.x,
				(res.getWidth()+block.y-1)/block.y
			);
			scalor_tanh<<<grid,block>>>(a.data,res.data,res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_tanh_d(float*from,float*to,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			if(i<h&&j<w){
				float tmp=tanhf(from[i*w+j]);
				to[i*w+j]=1-tmp*tmp;
			}
		}

		Matrix tanh_d(const Matrix&a){
			Matrix res(a.getHeight(),a.getWidth());
			dim3 block(16,16);
			dim3 grid(
				(res.getHeight()+block.x-1)/block.x,
				(res.getWidth()+block.y-1)/block.y
			);
			scalor_tanh_d<<<grid,block>>>(a.data,res.data,res.getHeight(),res.getWidth());
			return res;
		}

		Matrix identity(const Matrix&a){
			return a;
		}

		Matrix identity_d(const Matrix&a){
			Matrix res(a.getHeight(),a.getWidth());
			cudaMemset(res.getData(),0,a.getHeight()*a.getWidth()*sizeof(float));
			return res;
		}

		Matrix softmax(const Matrix&a){
			Matrix res(a.getHeight(),a.getWidth());
			float mx=0;
			
			return res;
		}

		Matrix softmax_d(const Matrix &a){
		}

	}
}
