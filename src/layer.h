#ifndef LAYER_H
#define LAYER_H

#include<functional>
#include<memory>

namespace LibMatchstick{
	class Matrix;
	class Tensor3d;
	class Tensor4d;

	class MLPLayer{
	private:
		std::function<Matrix(const Matrix&)>activation;
		std::function<Matrix(const Matrix&)>activation_d;
		std::unique_ptr<Matrix>W;
		std::unique_ptr<Matrix>b;
		std::unique_ptr<Matrix>last_input;
		std::unique_ptr<Matrix>z;
		bool sm;
		size_t in_size;
		size_t out_size;
	public:
		MLPLayer();
		MLPLayer(size_t in_size,size_t out_size);
		Matrix forward(const Matrix&input);
		Matrix backward(const Matrix&dl_da,float step);
		Matrix backward_dz(const Matrix&dl_dz,float step);
		void init(float low=-1,float high=1);
		Matrix saveWeight()const;
		Matrix saveBias()const;
		bool loadWeight(const Matrix&W);
		bool loadBias(const Matrix&b);
		void setActivation(
			const std::function<Matrix(const Matrix&)>&a,
			const std::function<Matrix(const Matrix&)>&a_d
		);
	};

	class CNNLayer{
	private:
		std::function<Tensor3d(const Tensor3d&)>activation;
		std::function<Tensor3d(const Tensor3d&)>activation_d;
		std::unique_ptr<Tensor4d>kernel;
		float*b;
		std::unique_ptr<Tensor3d>last_input;
		std::unique_ptr<Tensor3d>z;
		size_t in_c,in_h,in_w;
		size_t out_c,out_h,out_w;
		size_t stride;
		size_t padding;
	public:
		CNNLayer();
		CNNLayer(
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
		~CNNLayer();
		Tensor3d forward(const Tensor3d&input);
		Tensor3d backward(const Tensor3d&dl_da,float step);
		void init(float low=-1,float high=1);
		Tensor4d saveKernel()const;
		float*saveBias()const;
		bool loadKernel(const Tensor4d&k);
		bool loadBias(float*b);
		void setActivation(
			const std::function<Tensor3d(const Tensor3d&)>&a,
			const std::function<Tensor3d(const Tensor3d&)>&a_d
		);
	};
}

#endif
