#include"../losses.h"
#include"../matrix.h"

namespace LibMatchstick{
	namespace Losses{
		float MSE(const Matrix&x,const Matrix&e){
			float*tmp_x=(float*)malloc(x.getHeight()*x.getWidth()*sizeof(float));
			cudaMemcpy(tmp_x,x.getData(),x.getHeight()*x.getWidth()*sizeof(float),cudaMemcpyDeviceToHost);
			float*tmp_e=(float*)malloc(x.getHeight()*x.getWidth()*sizeof(float));
			cudaMemcpy(tmp_e,e.getData(),x.getHeight()*x.getWidth()*sizeof(float),cudaMemcpyDeviceToHost);
			float sum=0;
			size_t w=x.getWidth();
			size_t h=x.getHeight();
			for(size_t i=0;i<h;i++)
				for(size_t j=0;j<w;j++)
					sum+=(tmp_x[i*w+j]-tmp_e[i*w+j])*(tmp_x[i*w+j]-tmp_e[i*w+j]);
			float res=sum/2;
			return res;
		}

		Matrix MSE_d(const Matrix&x,const Matrix&e){
			return x-e;
		}

		float MAE(const Matrix&x,const Matrix&e){
			float*tmp_x=(float*)malloc(x.getHeight()*x.getWidth()*sizeof(float));
			cudaMemcpy(tmp_x,x.getData(),x.getHeight()*x.getWidth()*sizeof(float),cudaMemcpyDeviceToHost);
			float*tmp_e=(float*)malloc(x.getHeight()*x.getWidth()*sizeof(float));
			cudaMemcpy(tmp_e,e.getData(),x.getHeight()*x.getWidth()*sizeof(float),cudaMemcpyDeviceToHost);
			size_t w=x.getWidth();
			size_t h=x.getHeight();
			float sum=0;
			for(size_t i=0;i<h;i++)
				for(size_t j=0;j<w;j++){
					float tmp=(tmp_x[i*w+j]-tmp_e[i*w+j]);
					if(tmp>0)sum+=tmp;
					else sum-=tmp;
				}
			return sum/(w*h);
		}

		void __global__ scalor_MAE_d(float scale,const float*x,const float*e,float*res,size_t h,size_t w){
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			if(i<h&&j<w){
				if(x[i*w+j]>e[i*w+j])res[i*w+j]=scale;
				else if(x[i*w+j]<e[i*w+j])res[i*w+j]=-1*scale;
				else res[i*w+j]=0;
			}
		}

		Matrix MAE_d(const Matrix&x,const Matrix&e){
			Matrix res(x.getHeight(),x.getWidth());
			dim3 block(
				16,
				16
			);
			dim3 grid(
				(res.getHeight()+block.x-1)/block.x,
				(res.getWidth()+block.y-1)/block.y
			);
			float scale=1/(x.getHeight()*x.getWidth());
			scalor_MAE_d<<<grid,block>>>(scale,x.getData(),e.getData(),res.getData(),x.getHeight(),x.getWidth());
			return res;
		}

		float cross_entropy(const Matrix&x,const Matrix&e){
			float*tmp_x=(float*)malloc(x.getHeight()*x.getWidth()*sizeof(float));
			cudaMemcpy(tmp_x,x.getData(),x.getHeight()*x.getWidth()*sizeof(float),cudaMemcpyDeviceToHost);
			float*tmp_e=(float*)malloc(x.getHeight()*x.getWidth()*sizeof(float));
			cudaMemcpy(tmp_e,e.getData(),x.getHeight()*x.getWidth()*sizeof(float),cudaMemcpyDeviceToHost);
			size_t w=x.getWidth();
			size_t h=x.getHeight();
			const float eps=1e-21;
			float res=0;
			for(size_t i=0;i<h;i++)
				for(size_t j=0;j<w;j++){
					float v=tmp_x[i*w+j];
					if(v<eps)v=eps;
					if(v>1-eps)v=1-eps;
					res-=tmp_e[i*w+j]*logf(v);
				}
			return res;
		}

		void __global__ scalor_cross_entropy_d(const float*x,const float*e,float*res,size_t h,size_t w){
			const float eps=1e-12;
			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t j=blockIdx.y*blockDim.y+threadIdx.y;
			if(i<h&&j<w){
				float v=x[i*w+j];
				if(v<eps)v=eps;
				if(v>1-eps)v=1-eps;
				res[i*w+j]=-1*e[i*w+j]/v;
			}
		}

		Matrix cross_entropy_d(const Matrix&x,const Matrix&e){
			Matrix res(x.getHeight(),x.getWidth());
			dim3 block(
				16,
				16
			);
			dim3 grid(
				(res.getHeight()+block.x-1)/block.x,
				(res.getWidth()+block.y-1)/block.y
			);
			scalor_cross_entropy_d<<<grid,block>>>(x.getData(),e.getData(),res.getData(),x.getHeight(),x.getHeight());
			return res;
		}
	}
}
