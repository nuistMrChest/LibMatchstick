#include"../activation.h"

namespace LibMatchstick{
	namespace Activation{
		void __global__ scalor_relu(const float*from,float*to,size_t h,size_t w){
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
			scalor_relu<<<grid,block>>>(a.getData(),res.getData(),res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_relu_d(const float*from,float*to,size_t h,size_t w){
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
			scalor_relu_d<<<grid,block>>>(a.getData(),res.getData(),res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_leaky_relu(const float*from,float*to,size_t h,size_t w){
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
			scalor_leaky_relu<<<grid,block>>>(a.getData(),res.getData(),res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_leaky_relu_d(const float*from,float*to,size_t h,size_t w){
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
			scalor_leaky_relu_d<<<grid,block>>>(a.getData(),res.getData(),res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_sigmoid(const float*from,float*to,size_t h,size_t w){
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
			scalor_sigmoid<<<grid,block>>>(a.getData(),res.getData(),res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_sigmoid_d(const float*from,float*to,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			if(i<h&&j<w){
				float x=from[i*w+j];
				float s=1.0f/(1.0f+expf(-x));
				to[i*w+j]=s*(1.0f-s);
			}
		}

		Matrix sigmoid_d(const Matrix&a){
			Matrix res(a.getHeight(),a.getWidth());
			dim3 block(16,16);
			dim3 grid(
				(res.getHeight()+block.x-1)/block.x,
				(res.getWidth()+block.y-1)/block.y
			);
			scalor_sigmoid_d<<<grid,block>>>(a.getData(),res.getData(),res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_tanh(const float*from,float*to,size_t h,size_t w){
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
			scalor_tanh<<<grid,block>>>(a.getData(),res.getData(),res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_tanh_d(const float*from,float*to,size_t h,size_t w){
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
			scalor_tanh_d<<<grid,block>>>(a.getData(),res.getData(),res.getHeight(),res.getWidth());
			return res;
		}

		Matrix identity(const Matrix&a){
			return a;
		}

		Matrix identity_d(const Matrix&a){
			Matrix res(a.getHeight(),a.getWidth());
			cudaMemset(res.getData(),1,a.getHeight()*a.getWidth()*sizeof(float));
			return res;
		}

		void __global__ matrix_softmax(const float*from,float*to,size_t h,size_t w){
			float mx=0;
			for(size_t i=0;i<h;i++)
				for(size_t j=0;j<w;j++)
					if(from[i*w+j]>mx)mx=from[i*w+j];
			float sum=0;
			for(size_t i=0;i<h;i++)
				for(size_t j=0;j<w;j++){
					to[i*w+j]=expf(from[i*w+j]-mx);
					sum+=to[i*w+j];
				}
			for(size_t i=0;i<h;i++)
				for(size_t j=0;j<w;j++)
					to[i*w+j]/=sum;
		}

		Matrix softmax(const Matrix&a){
			Matrix res(a.getHeight(),a.getWidth());
			matrix_softmax<<<1,1>>>(a.getData(),res.getData(),a.getHeight(),a.getWidth());
			return res;
		}

		void __global__ matrix_softmax_d(const float*from,float*to,size_t h,size_t w){
			for(size_t i=0;i<h;i++)
				for(size_t j=0;j<w;j++)
					to[i*w+j]=from[i*w+j]*(1-from[i*w+j]);
		}

		Matrix softmax_d(const Matrix &a){
			Matrix s=softmax(a);
			Matrix res(a.getHeight(),a.getWidth());
			matrix_softmax_d<<<1,1>>>(s.getData(),res.getData(),a.getHeight(),a.getWidth());
			return res;
		}

		void __global__ scalor_relu_t(const float*from,float*to,size_t c,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			size_t k=blockIdx.z*blockDim.z+threadIdx.z;
			if(i<c&&j<h&&k<w)
				to[i*h*w+j*w+k]=from[i*h*w+j*w+k]>0?from[i*h*w+j*w+k]:0;
		}

		Tensor3d relu_t(const Tensor3d&a){
			Tensor3d res(a.getChannel(),a.getHeight(),a.getWidth());
			dim3 block(8,8,8);
			dim3 grid(
				(res.getChannel()+block.x-1)/block.x,
				(res.getHeight()+block.y-1)/block.y,
				(res.getWidth()+block.z-1)/block.z
			);
			scalor_relu_t<<<grid,block>>>(a.getData(),res.getData(),res.getChannel(),res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_relu_t_d(const float*from,float*to,size_t c,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			size_t k=blockIdx.z*blockDim.z+threadIdx.z;
			if(i<c&&j<h&&k<w)
				to[i*h*w+j*w+k]=from[i*h*w+j*w+k]>0?1:0;
		}

		Tensor3d relu_t_d(const Tensor3d&a){
			Tensor3d res(a.getChannel(),a.getHeight(),a.getWidth());
			dim3 block(8,8,8);
			dim3 grid(
				(res.getChannel()+block.x-1)/block.x,
				(res.getHeight()+block.y-1)/block.y,
				(res.getWidth()+block.z-1)/block.z
			);
			scalor_relu_t_d<<<grid,block>>>(a.getData(),res.getData(),res.getChannel(),res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_leaky_relu_t(const float*from,float*to,size_t c,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			size_t k=blockIdx.z*blockDim.z+threadIdx.z;
			if(i<c&&j<h&&k<w)
				to[i*h*w+j*w+k]=from[i*h*w+j*w+k]>0?from[i*h*w+j*w+k]:0.01;
		}

		Tensor3d leaky_relu_t(const Tensor3d&a){
			Tensor3d res(a.getChannel(),a.getHeight(),a.getWidth());
			dim3 block(8,8,8);
			dim3 grid(
				(res.getChannel()+block.x-1)/block.x,
				(res.getHeight()+block.y-1)/block.y,
				(res.getWidth()+block.z-1)/block.z
			);
			scalor_leaky_relu_t<<<grid,block>>>(
				a.getData(),
				res.getData(),
				res.getChannel(),
				res.getHeight(),
				res.getWidth()
			);
			return res;
		}

		void __global__ scalor_leaky_relu_t_d(const float*from,float*to,size_t c,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			size_t k=blockIdx.z*blockDim.z+threadIdx.z;
			if(i<c&&j<h&&k<w)
				to[i*h*w+j*w+k]=from[i*h*w+j*w+k]>0?1:0.01;
		}

		Tensor3d leaky_relu_t_d(const Tensor3d&a){
			Tensor3d res(a.getChannel(),a.getHeight(),a.getWidth());
			dim3 block(8,8,8);
			dim3 grid(
				(res.getChannel()+block.x-1)/block.x,
				(res.getHeight()+block.y-1)/block.y,
				(res.getWidth()+block.z-1)/block.z
			);
			scalor_leaky_relu_t_d<<<grid,block>>>(
				a.getData(),
				res.getData(),
				res.getChannel(),
				res.getHeight(),
				res.getWidth()
			);
			return res;
		}

		void __global__ scalor_sigmoid_t(const float*from,float*to,size_t c,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			size_t k=blockIdx.z*blockDim.z+threadIdx.z;
			if(i<c&&j<h&&k<w)
				to[i*h*w+j*w+k]=1/(1+__expf(-1*from[i*h*w+j*w+k]));
		}

		Tensor3d sigmoid_t(const Tensor3d&a){
			Tensor3d res(a.getChannel(),a.getHeight(),a.getWidth());
			dim3 block(8,8,8);
			dim3 grid(
				(res.getChannel()+block.x-1)/block.x,
				(res.getHeight()+block.y-1)/block.y,
				(res.getWidth()+block.z-1)/block.z
			);
			scalor_sigmoid_t<<<grid,block>>>(a.getData(),res.getData(),res.getChannel(),res.getHeight(),res.getWidth());
			return res;
		}

		void __global__ scalor_sigmoid_t_d(const float*from,float*to,size_t c,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			size_t k=blockIdx.z*blockDim.z+threadIdx.z;
			if(i<c&&j<h&&k<w){
				float x=from[i*h*w+j*w+k];
				float s=1.0f/(1.0f+expf(-x));
				to[i*h*w+j*w+k]=s*(1.0f-s);
			}
		}

		Tensor3d sigmoid_t_d(const Tensor3d&a){
			Tensor3d res(a.getChannel(),a.getHeight(),a.getWidth());
			dim3 block(8,8,8);
			dim3 grid(
				(res.getChannel()+block.x-1)/block.x,
				(res.getHeight()+block.y-1)/block.y,
				(res.getWidth()+block.z-1)/block.z
			);
			scalor_sigmoid_t_d<<<grid,block>>>(
				a.getData(),
				res.getData(),
				res.getChannel(),
				res.getHeight(),
				res.getWidth()
			);
			return res;
		}

		void __global__ scalor_tanh_t(const float*from,float*to,size_t c,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			size_t k=blockIdx.z*blockDim.z+threadIdx.z;
			if(i<c&&j<h&&k>w)
				to[i*h*w+j*w+k]=tanhf(from[i*h*w+j*w+k]);
		}

		Tensor3d tanh_t(const Tensor3d&a){
			Tensor3d res(a.getChannel(),a.getHeight(),a.getWidth());
			dim3 block(8,8,8);
			dim3 grid(
				(res.getChannel()+block.x-1)/block.x,
				(res.getHeight()+block.y-1)/block.y,
				(res.getWidth()+block.z-1)/block.z
			);
			scalor_tanh_t<<<grid,block>>>(
				a.getData(),
				res.getData(),
				res.getChannel(),
				res.getHeight(),
				res.getWidth()
			);
			return res;
		}

		void __global__ scalor_tanh_t_d(const float*from,float*to,size_t c,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			size_t k=blockIdx.z*blockDim.z+threadIdx.z;
			if(i<c&&j<h&&k<w){
				float tmp=tanhf(from[i*h*w+j*w+k]);
				to[i*h*w+j*w+k]=1-tmp*tmp;
			}
		}

		Tensor3d tanh_t_d(const Tensor3d&a){
			Tensor3d res(a.getChannel(),a.getHeight(),a.getWidth());
			dim3 block(8,8,8);
			dim3 grid(
				(res.getChannel()+block.x-1)/block.x,
				(res.getHeight()+block.y-1)/block.y,
				(res.getWidth()+block.z-1)/block.z
			);
			scalor_tanh_t_d<<<grid,block>>>(
				a.getData(),
				res.getData(),
				res.getChannel(),
				res.getHeight(),
				res.getWidth()
			);
			return res;
		}

		Tensor3d identity_t(const Tensor3d&a){
			return a;
		}

		Tensor3d identity_t_d(const Tensor3d&a){
			Tensor3d res(a.getChannel(),a.getHeight(),a.getWidth());
			cudaMemset(res.getData(),1,a.getChannel()*a.getHeight()*a.getWidth()*sizeof(float));
			return res;
		}

	}
}
