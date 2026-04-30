#include<iostream>
#include"src/matrix.h"

using namespace LibMatchstick;

int main(){
	Matrix a{
		{1,2,3},
		{4,5,6},
		{7,8,9}
	};
	std::cout<<a<<"\n";
	Matrix b{
		{0,0,1},
		{0,1,0},
		{1,0,0}
	};
	std::cout<<a*b<<"\n";
}
