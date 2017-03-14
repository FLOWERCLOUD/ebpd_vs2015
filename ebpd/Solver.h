#ifndef __M_SOLVER__
#define __M_SOLVER__
#include "basic_types.h"
#include "solver/fadiff.h"
#include "solver/badiff.h"
#include <functional>
namespace ebpd_solver
{

	//计算雅克比矩阵的函数

	void ffgradient(MatrixXX& out_, std::function< fadbad::F<double> (const fadbad::F<double>& x, const fadbad::F<double>& y, const fadbad::F<double>& z)> func,double x0,double x1,double x2);
	void ffgradientr(MatrixXX& out_, std::function< fadbad::F<double> (fadbad::F<double> x, fadbad::F<double> y, fadbad::F<double> z)> func,double x0,double x1,double x2);
	void ffgradientB(MatrixXX& out_, std::function< fadbad::B<double> (const fadbad::B<double>& x, const fadbad::B<double>& y, const fadbad::B<double>& z)> func);
	void val(double& out_, std::function< fadbad::B< fadbad::F<double> > (const fadbad::B< fadbad::F<double> >& , const fadbad::B< fadbad::F<double> >& , const fadbad::B< fadbad::F<double> >& )> func ,
		double x0,double x1,double x2 );
	void ffgradient2(MatrixXX& out_, std::function< fadbad::B< fadbad::F<double> > (const fadbad::B< fadbad::F<double> >& , const fadbad::B< fadbad::F<double> >& , const fadbad::B< fadbad::F<double> >& )> func,
		double x0,double x1,double x2);
	void ffHesse(MatrixXX& out_,std::function< fadbad::B< fadbad::F<double> > (const fadbad::B< fadbad::F<double> >& , const fadbad::B< fadbad::F<double> >& , const fadbad::B< fadbad::F<double> >& )> func,
		double x0,double x1,double x2	);
	//void ffHesseB(MatrixXX& out_, std::function< fadbad::B<double> (const fadbad::B<double>& x, const fadbad::B<double>& y, const fadbad::B<double>& z)> func);
	void newtonSolver( std::function< fadbad::B< fadbad::F<double> > (const fadbad::B< fadbad::F<double> >& , const fadbad::B< fadbad::F<double> >& , const fadbad::B< fadbad::F<double> >& )> func );
}


#endif // !__M_SOLVER__
