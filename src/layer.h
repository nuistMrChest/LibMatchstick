#ifndef LAYER_H
#define LAYER_H

#include<functional>
#include<memory>

namespace LibMatchstick{
	class Matrix;

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
	};
}

#endif
