#include"../../include/matchstick/layer.h"
#include"../../include/matchstick/matrix.h"
#include"../../include/matchstick/activation.h"
#include"../../include/matchstick/tensor_3d.h"
#include"../../include/matchstick/tensor_4d.h"
#include<algorithm>
#include<cstdio>
#include<memory>
#include<random>
#include<cfloat>

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

	void MLPLayer::init(float high,float low){
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

	bool MLPLayer::isSm()const{
		return sm;
	}

	void MLPLayer::setSm(){
		sm=true;
	}

	CNNLayerType CNNLayer::getType(){
		return type;
	}

	CNNPoolingLayer::CNNPoolingLayer():
		k_h(0),
		k_w(0)
	{
		in_c=0; 
		in_h=0; 
		in_w=0; 
		out_c=0; 
		out_h=0; 
		out_w=0; 
		stride=0; 
		padding=0; 
		type=CNNLayerType::Pooling;
	}

	CNNConvolutionLayer::CNNConvolutionLayer():
		b(nullptr)
	{
		in_c=0; 
		in_h=0; 
		in_w=0; 
		out_c=0; 
		out_h=0; 
		out_w=0; 
		stride=0; 
		padding=0; 
		type=CNNLayerType::Convolution;
		activation=Activation::identity_t;
		activation_d=Activation::identity_t_d;
	}


	CNNPoolingLayer::CNNPoolingLayer(
		size_t in_c,
		size_t in_h,
		size_t in_w,
		size_t out_c,
		size_t out_h,
		size_t out_w,
		size_t ker_h,
		size_t ker_w,
		size_t s,
		size_t p
	):
		k_h(ker_h),
		k_w(ker_w)
	{
		this->in_c=in_c;
		this->in_h=in_h;
		this->in_w=in_w;
		this->out_c=out_c;
		this->out_h=out_h;
		this->out_w=out_w;
		stride=s;
		padding=p;
		type=CNNLayerType::Pooling;
		saved_max=std::make_unique<Tensor4d>(out_c,2,out_h,out_w);
	}

	static void __global__ max(
		float*res,
		const float*ten,
		float*savk,
		size_t r_c,
		size_t r_h,
		size_t r_w,
		size_t stride,
		size_t padding,
		size_t k_h,
		size_t k_w,
		size_t t_c,
		size_t t_h,
		size_t t_w
	){
		size_t i=blockIdx.x*blockDim.x+threadIdx.x;
		size_t j=blockIdx.y*blockDim.y+threadIdx.y;
		size_t k=blockIdx.z*blockDim.z+threadIdx.z;
		float max=-FLT_MAX;
		if(i<r_c&&j<r_h&&k<r_w){
			res[r_h*r_w*i+r_w*j+k]=0;
			size_t x=j*stride;
			size_t y=k*stride;
			size_t s_i;
			size_t s_j;
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
							(long long)(x+jj)-
							(long long)(padding)>=
							(long long)(t_h)||
							(long long)(y+kk)-
							(long long)(padding)>=
							(long long)(t_w)
						)
					)
						if(ten[i*t_h*t_w+(x+jj-padding)*t_w+(y+kk-padding)]>max){
							max=ten[i*t_h*t_w+(x+jj-padding)*t_w+(y+kk-padding)];
							s_i=x+jj-padding;
							s_j=y+kk-padding;
						}
			res[r_h*r_w*i+r_w*j+k]=max;
			const size_t s_c=2;
			const size_t s_h=r_h;
			const size_t s_w=r_w;
			savk[i*s_c*s_h*s_w+0*s_h*s_w+j*s_w+k]=s_i;
			savk[i*s_c*s_h*s_w+1*s_h*s_w+j*s_w+k]=s_j;
		}
	}

	Tensor3d CNNPoolingLayer::forward(const Tensor3d&input){
		Tensor3d res;
		if(input.getChannel()==in_c&&input.getHeight()==in_h&&input.getWidth()==in_w){
			res.resize(
				input.getChannel(),
				(input.getHeight()+2*padding-k_h)/stride+1,
				(input.getWidth()+2*padding-k_w)/stride+1
			);
			dim3 block(8,8,8);
			dim3 grid(
				(res.getChannel()+block.x-1)/block.x,
				(res.getHeight()+block.y-1)/block.y,
				(res.getWidth()+block.z-1)/block.z
			);
			max<<<grid,block>>>(
				res.getData(),
				input.getData(),
				saved_max->getData(),
				res.getChannel(),
				res.getHeight(),
				res.getWidth(),
				stride,
				padding,
				k_h,
				k_w,
				input.getChannel(),
				input.getHeight(),
				input.getWidth()
			);
		}
		return res;
	}

	static void __global__ pass_grad(
		float*res,
		const float*grad,
		const float*savk,
		size_t r_h,
		size_t r_w,
		size_t g_c,
		size_t g_h,
		size_t g_w
	){
		size_t i=blockIdx.x*blockDim.x+threadIdx.x;
		size_t j=blockIdx.y*blockDim.y+threadIdx.y;
		size_t k=blockIdx.z*blockDim.z+threadIdx.z;
		if(i<g_c&&j<g_h&&k<g_w){
			const size_t s_c=2;
			const size_t s_h=g_h;
			const size_t s_w=g_w;
			const size_t r_i=i;
			const size_t r_j=savk[i*s_c*s_h*s_w+0*s_h*s_w+j*s_w+k];
			const size_t r_k=savk[i*s_c*s_h*s_w+1*s_h*s_w+j*s_w+k];
			res[r_i*r_h*r_w+r_j*r_w+r_k]=grad[i*g_h*g_w+j*g_w+k];
		}
	}

	Tensor3d CNNPoolingLayer::backward(const Tensor3d&dl_da,float step){
		Tensor3d res(in_c,in_h,in_w);
		cudaMemset(res.getData(),0,in_c*in_h*in_w*sizeof(float));
		dim3 block(8,8,8);
		dim3 grid(
			(out_c+block.x-1)/block.x,
			(out_h+block.y-1)/block.y,
			(out_w+block.z-1)/block.z
		);
		pass_grad<<<grid,block>>>(
			res.getData(),
			dl_da.getData(),
			saved_max->getData(),
			in_h,
			in_w,
			out_c,
			out_h,
			out_w
		);
		return res;
	}

	CNNConvolutionLayer::CNNConvolutionLayer(
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
		size_t p,
		const std::function<Tensor3d(const Tensor3d&)>&a,
		const std::function<Tensor3d(const Tensor3d&)>&a_d
	)
	{
		this->in_c=in_c;
		this->in_h=in_h;
		this->in_w=in_w;
		this->out_c=out_c;
		this->out_h=out_h;
		this->out_w=out_w;
		stride=s;
		padding=p;
		type=CNNLayerType::Convolution;
		cudaMalloc(&b,out_c*sizeof(float));
		kernel=std::make_unique<Tensor4d>(out_c,k_c,k_h,k_w);
		last_input=std::make_unique<Tensor3d>(in_c,in_h,in_w);
		z=std::make_unique<Tensor3d>(out_c,out_h,out_w);
		activation=Activation::identity_t;
		activation_d=Activation::identity_t_d;
		setActivation(a,a_d);
	}

	void CNNConvolutionLayer::init(float high,float low){
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
				for(size_t k=0;k<kernel->getHeight();k++)
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

	static void __global__ add_bias(float*b,float*res,size_t o_c,size_t o_h,size_t o_w){
		size_t i=blockIdx.x*blockDim.x+threadIdx.x;
		size_t j=blockIdx.y*blockDim.y+threadIdx.y;
		size_t k=blockIdx.z*blockDim.z+threadIdx.z;
		if(i<o_c&&j<o_h&&k<o_w)
			res[i*o_h*o_w+j*o_w+k]+=b[i];
	}

	Tensor3d CNNConvolutionLayer::forward(const Tensor3d&input){
		Tensor3d res;
		if(input.getChannel()==in_c&&input.getHeight()==in_h&&in_w==input.getWidth()){
			*last_input=input;
			*z=input.convolution(*kernel,stride,padding);
			dim3 block(8,8,8);
			dim3 grid(
				(out_c+block.x-1)/block.x,
				(out_h+block.y-1)/block.y,
				(out_w+block.z-1)/block.z
			);
			add_bias<<<grid,block>>>(b,z->getData(),out_c,out_h,out_w);
			res=activation(*z);
		}
		return res;
	}

	static void __global__ get_dl_da(
		float*kernel,
		float*res,
		float*dl_dz,
		size_t i_c,
		size_t i_h,
		size_t i_w,
		size_t o_c,
		size_t o_h,
		size_t o_w,
		size_t stride,
		size_t padding,
		size_t k_c,
		size_t k_h,
		size_t k_w
	){
		size_t ii=blockIdx.x*blockDim.x+threadIdx.x;
		size_t jj=blockIdx.y*blockDim.y+threadIdx.y;
		size_t kk=blockIdx.z*blockDim.z+threadIdx.z;
		if(ii<i_c&&jj<i_h&&kk<i_w)
			for(size_t i=0;i<o_c;i++)
				for(size_t j=0;j<o_h;j++)
					for(size_t k=0;k<o_w;k++)
						if(
							!(
								(long long)jj-
								(long long)(stride*j-padding)<
								0||
								(long long)kk-
								(long long)(stride*k-padding)<
								0||
								(long long)jj-
								(long long)(stride*j-padding)>=
								(long long)k_h||
								(long long)kk-
								(long long)(stride*k-padding)>=
								(long long)k_w
							)
						)
							res[ii*i_h*i_w+jj*i_w+kk]+=
								dl_dz[i*o_h*o_w+j*o_w+k]*
								kernel[
									(i*k_c*k_h*k_w)+
									(ii*k_h*k_w)+
									(jj-j*stride+padding)*k_w+
									(kk-k*stride+padding)
								];
	}

	static void __global__ grad_update_ker(
		float*kernel,
		float*dl_dz,
		float*last_input,
		size_t i_c,
		size_t i_h,
		size_t i_w,
		size_t o_c,
		size_t o_h,
		size_t o_w,
		size_t stride,
		size_t padding,
		size_t k_c,
		size_t k_h,
		size_t k_w,
		float step
	){
		size_t ii=blockIdx.x*blockDim.x+threadIdx.x;
		size_t jj=blockIdx.y*blockDim.y+threadIdx.y;
		size_t kk=blockIdx.z*blockDim.z+threadIdx.z;
		if(ii<k_c&&jj<k_h&&kk<k_w)
			for(size_t i=0;i<o_c;i++)
				for(size_t j=0;j<o_h;j++)
					for(size_t  k=0;k<o_w;k++)
						if(
							!(
								(long long)(j*stride+jj)-
								(long long)padding<
								0||
								(long long)(k*stride+kk)-
								(long long)padding<
								0||
								(long long)(j*stride+jj)-
								(long long)padding>=
								(long long)i_h||
								(long long)(k*stride+kk)-
								(long long)padding>=
								(long long)i_w
							)
						)
							kernel[i*k_c*k_h*k_w+ii*k_h*k_w+jj*k_w+kk]-=
								step*(
									dl_dz[i*o_h*o_w+j*o_w+k]*
									last_input[ii*i_h*i_w+(j*stride+jj-padding)*i_w+(k*stride+kk-padding)]
								);
	}

	static void __global__ grad_update_bias(
		float*b,
		float*dl_dz,
		size_t o_c,
		size_t o_h,
		size_t o_w,
		float step
	){
		for(size_t i=0;i<o_c;i++)
			for(size_t j=0;j<o_h;j++)
				for(size_t k=0;k<o_w;k++)
					b[i]-=step*dl_dz[i*o_h*o_w+j*o_w+k];
	}

	Tensor3d CNNConvolutionLayer::backward(const Tensor3d&dl_da,float step){
		Tensor3d res(in_c,in_h,in_w);
		cudaMemset(res.getData(),0,in_c*in_h*in_w*sizeof(float));
		Tensor3d dl_dz=dl_da.hadamard(activation_d(*z));
		dim3 block(8,8,8);
		dim3 grid(
			(in_c+block.x-1)/block.x,
			(in_h+block.y-1)/block.y,
			(in_w+block.z-1)/block.z
		);
		get_dl_da<<<grid,block>>>(
			kernel->getData(),
			res.getData(),
			dl_dz.getData(),
			in_c,
			in_h,
			in_w,
			out_c,
			out_h,
			out_w,
			stride,
			padding,
			kernel->getChannel(),
			kernel->getHeight(),
			kernel->getWidth()
		);
		dim3 block2(8,8,8);
		dim3 grid2(
			(kernel->getChannel()+block.x-1)/block.x,
			(kernel->getHeight()+block.y-1)/block.y,
			(kernel->getWidth()+block.z-1)/block.z
		);
		grad_update_ker<<<grid2,block2>>>(
			kernel->getData(),
			dl_dz.getData(),
			last_input->getData(),
			in_c,
			in_h,
			in_w,
			out_c,
			out_h,
			out_w,
			stride,
			padding,
			kernel->getChannel(),
			kernel->getHeight(),
			kernel->getWidth(),
			step
		);
		grad_update_bias<<<1,1>>>(b,dl_dz.getData(),out_c,out_h,out_w,step);
		return res;
	}

	Tensor4d CNNConvolutionLayer::saveKernel()const{
		return*kernel;
	}

	std::vector<float>CNNConvolutionLayer::saveBias()const{
		float*tmp=(float*)malloc(out_c*sizeof(float));
		cudaMemcpy(tmp,b,out_c*sizeof(float),cudaMemcpyDeviceToHost);
		return std::vector<float>(tmp,tmp+out_c);
	}

	bool CNNConvolutionLayer::loadKernel(const Tensor4d&k){
		if(
			kernel->getBatch()==k.getBatch()&&
			kernel->getChannel()==k.getChannel()&&
			kernel->getHeight()==k.getHeight()&&
			kernel->getWidth()==k.getWidth()
		){
			*kernel=k;
			return true;
		}
		return false;
	}

	bool CNNConvolutionLayer::loadBias(const std::vector<float>&b){
		if(b.size()!=out_c)return false;
		cudaMemcpy(this->b,b.data(),out_c*sizeof(float),cudaMemcpyHostToDevice);
		return true;
	}

	void CNNConvolutionLayer::setActivation(
		const std::function<Tensor3d(const Tensor3d&)>&a,
		const std::function<Tensor3d(const Tensor3d&)>&a_d
	){
		activation=a;
		activation_d=a_d;
	}

	CNNConvolutionLayer::~CNNConvolutionLayer(){
		cudaFree(b);
	}

	CNNConvolutionLayer::CNNConvolutionLayer(const CNNConvolutionLayer&a){
		activation=a.activation;
		activation_d=a.activation_d;
		kernel=std::make_unique<Tensor4d>(*a.kernel);
		last_input=std::make_unique<Tensor3d>(*a.last_input);
		z=std::make_unique<Tensor3d>(*a.z);
		in_c=a.in_c;
		in_h=a.in_h;
		in_w=a.in_w;
		out_c=a.out_c;
		out_h=a.out_h;
		out_w=a.out_w;
		stride=a.stride;
		padding=a.padding;
		cudaMalloc(&b,out_c*sizeof(float));
		cudaMemcpy(b,a.b,out_c*sizeof(float),cudaMemcpyDeviceToDevice);
	}

	CNNConvolutionLayer&CNNConvolutionLayer::operator=(const CNNConvolutionLayer&a){
		activation=a.activation;
		activation_d=a.activation_d;
		kernel=std::make_unique<Tensor4d>(*a.kernel);
		last_input=std::make_unique<Tensor3d>(*a.last_input);
		z=std::make_unique<Tensor3d>(*a.z);
		in_c=a.in_c;
		in_h=a.in_h;
		in_w=a.in_w;
		out_c=a.out_c;
		out_h=a.out_h;
		out_w=a.out_w;
		stride=a.stride;
		padding=a.padding;
		cudaMalloc(&b,out_c*sizeof(float));
		cudaMemcpy(b,a.b,out_c*sizeof(float),cudaMemcpyDeviceToDevice);
		return*this;
	}

	size_t CNNLayer::getOutChannel()const{
		return out_c;
	}

	size_t CNNLayer::getOutHeight()const{
		return out_h;
	}

	size_t CNNLayer::getOutWidth()const{
		return out_w;
	}
}
