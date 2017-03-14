#pragma once
#include "basic_types.h"
#include <iostream>
using namespace std;
inline void printMatrix(MatrixXX& _in)
{
	int n_row = _in.rows();
	int n_col = _in.cols();
	for( int i  = 0; i < n_row ;++i)
	{
		for( int j = 0; j < n_col ;++j )
		{
			cout<< _in(i ,j)<<" ";
		}
		cout<<std::endl;
	}
}
inline void printMatrix(const MatrixXXF& _in)
{
	int n_row = _in.rows();
	int n_col = _in.cols();
	for( int i  = 0; i < n_row ;++i)
	{
		for( int j = 0; j < n_col ;++j )
		{
			cout<< _in(i ,j).val()<<" ";
		}
		cout<<std::endl;
	}
}
