#include "ebpd.h"
#include <QtWidgets/QApplication>
#include "Solver.h"
#include "print_util.h"

#if 0
#include "skinning_model.h"
using namespace fadbad;

F<double> func(const F<double>& x1, const F<double>& x2,const F<double>& x3)
{
	F<double> f= pow((3*x1-cos(x2*x3)-1.0/2),2)+
		pow(( pow(x1,2)-81*(x2+0.1)+sin(x3)+1.06),2)+
		pow( (exp(-x1*x2)+20*x3+1.0/3*(10*3.14159-3)),2);
	return f;
}
F<double> funr(F<double> x1,  F<double> x2,F<double> x3)
{
	F<double> f= pow((3*x1-cos(x2*x3)-1.0/2),2)+
		pow(( pow(x1,2)-81*(x2+0.1)+sin(x3)+1.06),2)+
		pow( (exp(-x1*x2)+20*x3+1.0/3*(10*3.14159-3)),2);
	return f;
}
B< F<double> > funcjac2(const B< F<double> >& x1, const B< F<double> >& x2,const B< F<double> >& x3)
{
	B< F<double> > f= pow((3*x1-cos(x2*x3)-1.0/2),2)+
		pow(( pow(x1,2)-81*(x2+0.1)+sin(x3)+1.06),2)+
		pow( (exp(-x1*x2)+20*x3+1.0/3*(10*3.14159-3)),2);
	return f;
}
F<double> func1(const F<double>& x, const F<double>& y,const F<double>& z)
{
	F<double> f= 1*x*y+ 2*y + 3*x*z;
	return f;
}
B<double> func2(const B<double>& x1, const B<double>& x2,const B<double>& x3)
{
	B<double> f= pow((3*x1-cos(x2*x3)-1/2),2)+
		pow(( pow(x1,2)-81*(x2+0.1)+sin(x3)+1.06),2)+
		pow( (exp(-x1*x2)+20*x3+1/3*(10*3.14159-3)),2);
	return f;
}
B< F<double> > funcjac(const B< F<double> >& x1, const B< F<double> >& x2,const B< F<double> >& x3)
{
	B< F<double> > f= 1*x1*x2+ 2*x2 + 3*x1*x3;
	return f;
}
B< F<double> > func3(const B< F<double> >& x1, const B< F<double> >& x2,const B< F<double> >& x3)
{
	B< F<double> > f= 3*pow(x1,2) - 2*pow(x2,2) + 5*pow(x3,2) +5*x1+3*x2-4*x3- x1*x2 -5*x1*x3 + x2*x3+5;
	return f;
}

F<double> funcf(const F<double>* x1, int num)
{
	F<double> f;
	return f;
}


int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	ebpd w;
	w.show();
	MatrixXX out_;
	out_.resize(1,3);
	ebpd_solver::ffgradientr(out_,funr,0,0,0);
	printMatrix(out_);
	MatrixXX out2_;
	out2_.resize(1,3);
	ebpd_solver::ffgradientB(out2_,func2);
	printMatrix(out2_);
	//MatrixXX out3_;
	//out3_.resize(3,3);
    //ebpd_solver::ffHesse(out3_,funcjac,1,2,3);
//	printMatrix(out3_);
	//cout<<"solve 1"<<endl;
	//ebpd_solver::newtonSolver(funcjac ,func1);
	cout<<"solve 2"<<endl;
	ebpd_solver::newtonSolver(funcjac2 );
	//cout<<"solve 3"<<endl;
	//ebpd_solver::newtonSolver(func3 );
	Skinning sk1;
	int num_vtx = 5;
	int num_bone = 3;
	int num_example = 4;
	MatrixXX weight(num_vtx ,num_bone);
	weight.setZero();
	MatrixXX tf(4*num_bone ,4);
	tf.setZero();
	MatrixXX ori_vtx(num_vtx,3);
	ori_vtx.setZero();
	MatrixXX eweight;
	eweight.resize( num_vtx ,num_example);
	eweight.setZero();
	MatrixXX tf2(7*num_bone ,num_example);
	tf2.setZero();
//    MatrixXX out = sk1.caculateTraditionalSkinng( weight ,tf,ori_vtx,num_vtx,num_bone);
	MatrixXX out2 = sk1.caculateQLEPSkinng( weight ,tf2,ori_vtx,eweight,num_vtx,num_bone);

	return a.exec();
}

#endif