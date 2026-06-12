#ifndef TENSOR_3D_H
#define TENSOR_3D_H

#include<cstddef>
#include<initializer_list>
#include<iostream>
#include<vector>

namespace LibMatchstick{
	class Matrix;
	class Tensor4d;
	class Tensor3d{
		private:
			size_t c,h,w;
			float*data;
		public:
			Tensor3d();
			Tensor3d(size_t c,size_t h,size_t w);
			Tensor3d(std::initializer_list<std::initializer_list<std::initializer_list<float>>>a);
			~Tensor3d();
			friend std::ostream&operator<<(std::ostream&os,const Tensor3d&a);
			size_t getChannel()const;
			size_t getHeight()const;
			size_t getWidth()const;
			void resize(size_t c,size_t h,size_t w);
			Tensor3d(const Tensor3d&a);
			Tensor3d&operator=(const Tensor3d&a);
			Tensor3d operator+(const Tensor3d&a)const;
			Tensor3d operator-(const Tensor3d&a)const;
			Tensor3d&operator+=(const Tensor3d&a);
			Tensor3d&operator-=(const Tensor3d&a);
			Tensor3d hadamard(const Tensor3d&a)const;
			void set(size_t i,size_t j,size_t k,float v);
			float get(size_t i,size_t j,size_t k)const;
			float*getData();
			const float*getData()const;
			Tensor3d convolution(const Tensor4d&k,size_t stride,size_t padding)const;
			Matrix flatten();
			static Tensor3d deflatten(const Matrix&a,size_t c,size_t h,size_t w);
	};
}

#endif
