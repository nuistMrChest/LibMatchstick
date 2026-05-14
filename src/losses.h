#ifndef LOSSES_H
#define LOSSES_H

namespace LibMatchstick{
	class Matrix;

	namespace Losses{
		float MSE(const Matrix&x,const Matrix&e);
		Matrix MSE_d(const Matrix&x,const Matrix&e);
		float MAE(const Matrix&x,const Matrix&e);
		Matrix MAE_d(const Matrix&x,const Matrix&e);
		float cross_entropy(const Matrix&x,const Matrix&e);
		Matrix cross_entropy_d(const Matrix&x,const Matrix&e);
	}
}

#endif
