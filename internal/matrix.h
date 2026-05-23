#ifndef MATRIX_H
#define MATRIX_H

#include<stddef.h>
#include<initializer_list>
#include<iostream>

namespace LibMatchstick{
	class Matrix{
	private:
		float*data;
		size_t h,w;
	public:
		Matrix();
		Matrix(size_t h,size_t w);
		Matrix(std::initializer_list<std::initializer_list<float>>a);
		~Matrix();
		friend std::ostream&operator<<(std::ostream&os,const Matrix&a);
		size_t getHeight()const;
		size_t getWidth()const;
		void resize(size_t h,size_t w);
		Matrix(const Matrix&a);
		Matrix&operator=(const Matrix&a);
		Matrix operator+(const Matrix&a)const;
		Matrix operator-(const Matrix&a)const;
		Matrix&operator+=(const Matrix&a);
		Matrix&operator-=(const Matrix&a);
		Matrix hadamard(const Matrix&a)const;
		Matrix operator*(const Matrix&a)const;
		void set(size_t i,size_t j,float v);
		float get(size_t i,size_t j)const;
		float*getData();
		const float*getData()const;
		Matrix transpose()const;
		Matrix operator*(float a)const;
		Matrix operator*=(float a);
		friend Matrix operator*(float a,const Matrix&b);
	};

	typedef Matrix Tensor2d;
}

#endif


