#ifndef TENSOR_4D_H
#define TENSOR_4D_H

#include"cstddef"
#include<initializer_list>

namespace LibMatchstick{
	class Tensor4d{
	private:
		float*data;
		size_t b;
		size_t c;
		size_t h;
		size_t w;
	public:
		Tensor4d();
		Tensor4d(size_t b,size_t c,size_t h,size_t w);
		Tensor4d(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<float>>>>a);
		size_t getBatch()const;
		size_t getChannel()const;
		size_t getHeight()const;
		size_t getWidth()const;
		void resize(size_t b,size_t c,size_t h,size_t w);
		Tensor4d(const Tensor4d&a);
		Tensor4d&operator=(const Tensor4d&a);
		void set(size_t i,size_t j,size_t k,size_t l,float v);
		float get(size_t i,size_t j,size_t k,size_t l)const;
		float*getData();
		const float*getData()const;
	};
}

#endif
