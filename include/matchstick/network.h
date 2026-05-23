#ifndef NETWORK_H
#define NETWORK_H

#include<functional>
#include<vector>

namespace LibMatchstick{
	class Matrix;
	class MLPLayer;
	class Tensor3d;
	class Tensor4d;
	class CNNLayer;

	class MLP{
	private:
		std::vector<MLPLayer>layers;
		float step;
		bool ce;
		std::function<float(const Matrix&,const Matrix&)>loss;
		std::function<Matrix(const Matrix&,const Matrix&)>loss_d;
	public:
		MLP();
		MLP(size_t layer_size,float step);
		void setLayer(size_t index,size_t in_size,size_t out_size);
		void setLayerActivation(
			size_t index,
			std::function<Matrix(const Matrix&)>a,
			std::function<Matrix(const Matrix&)>a_d
		);
		void setLoss(
			std::function<float(const Matrix&,const Matrix&)>loss,
			std::function<Matrix(const Matrix&,const Matrix&)>loss_d
		);
		float train(const Matrix&input,const Matrix&expected,Matrix&l_dl_da);
		Matrix use(const Matrix&input);
		void setSm();
		void setCe();
		void loadWeight(size_t index,const Matrix&W);
		void loadBias(size_t index,const Matrix&b);
		Matrix saveWeight(size_t index)const;
		Matrix saveBias(size_t index)const;
		void init(float high=1,float low=-1);
	};

	class CNN{
	private:
		std::vector<CNNLayer>layers;
		float step;
		MLP m;
	public:
		CNN();
		CNN(size_t layer_size,float step,size_t mlp_layer_size,float mlp_step);
		MLP&mlp();
		const MLP&mlp()const;
		void setLayer(
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
		);
		void setLayerActivation(
			size_t index,
			std::function<Tensor3d(const Tensor3d&)>activation,
			std::function<Tensor3d(const Tensor3d&)>activation_d
		);
		float train(const Tensor3d&input,const Matrix&expected);
		Matrix use(const Tensor3d&input);
		Tensor4d saveKernel(size_t index)const;
		std::vector<float>saveBias(size_t index)const;
		bool loadKernel(size_t index,const Tensor4d&k);
		bool loadBias(size_t index,const std::vector<float>&b);
		void init(float high=1,float low=-1);
	};
}

#endif
