#include "Solver.h"
#include "print_util.h"
#include <iostream>
using namespace std;
using namespace fadbad;


void ebpd_solver::ffgradient(MatrixXX& out_, std::function< F<double> (const F<double>& x, const F<double>& y, const F<double>& z)> func ,double x0,double x1,double x2)
{
	F<double> x,y,z,f;     // Declare variables x,y,f
	x=x0;                 // Initialize variable x
 	x.diff(0,3);         // Differentiate with respect to x (index 0 of 2)
	y=x1;                 // Initialize variable y
 	y.diff(1,3);         // Differentiate with respect to y (index 1 of 2)
	z=x1;
 	z.diff(2,3);         // Differentiate with respect to y (index 1 of 2)
	cout<<"x "<<x.x()<< endl;
	cout<<"y "<<y.x()<< endl;
	cout<<"y "<<z.x()<< endl;
	f=func(x,y,z);         // Evaluate function and derivatives
 	double fval=f.x();   // Value of function
	double dfdx=f.d(0);  // Value of df/dx (index 0 of 2)
	double dfdy=f.d(1);  // Value of df/dy (index 1 of 2)
	double dfdz=f.d(2);
//	cout << "f(x,y,z)=" << fval << endl;
	cout << "df/dx(x,x)=" << dfdx << endl;
	cout << "df/dy(x,y)=" << dfdy << endl;
	cout << "df/dy(x,z)=" << dfdz << endl;
	out_(0,0) = (float)dfdx;
	out_(0,1) = (float)dfdy;
	out_(0,2) = (float)dfdz;
}
void ebpd_solver::ffgradientr(MatrixXX& out_, std::function< fadbad::F<double> (fadbad::F<double> x, fadbad::F<double> y, fadbad::F<double> z)> func,double x0,double x1,double x2)
{
	F<double> x,y,z,f;     // Declare variables x,y,f
	x=x0;                 // Initialize variable x
	x.diff(0,3);         // Differentiate with respect to x (index 0 of 2)
	y=x1;                 // Initialize variable y
	y.diff(1,3);         // Differentiate with respect to y (index 1 of 2)
	z=x1;
	z.diff(2,3);         // Differentiate with respect to y (index 1 of 2)
	cout<<"x "<<x.x()<< endl;
	cout<<"y "<<y.x()<< endl;
	cout<<"y "<<z.x()<< endl;
	f=func(x,y,z);         // Evaluate function and derivatives
	double fval=f.x();   // Value of function
	double dfdx=f.d(0);  // Value of df/dx (index 0 of 2)
	double dfdy=f.d(1);  // Value of df/dy (index 1 of 2)
	double dfdz=f.d(2);
	//	cout << "f(x,y,z)=" << fval << endl;
	cout << "df/dx(x,x)=" << dfdx << endl;
	cout << "df/dy(x,y)=" << dfdy << endl;
	cout << "df/dy(x,z)=" << dfdz << endl;
	out_(0,0) = (float)dfdx;
	out_(0,1) = (float)dfdy;
	out_(0,2) = (float)dfdz;
}
void ebpd_solver::ffgradientB(MatrixXX& out_, std::function< B<double> (const B<double>& x, const B<double>& y, const B<double>& z)> func)
{
	B<double> x,y,z,f;     // Declare variables x,y,f
	x=0;                 // Initialize variable x
	y=0;                 // Initialize variable y
	z=0;
	f=func(x,y,z);         // Evaluate function and derivatives


	double fval=f.x();   // Value of function
	f.diff(0,1);
	cout<<"x "<<x.x()<< endl;
	cout<<"y "<<y.x()<< endl;
	cout<<"y "<<z.x()<< endl;
	double dfdx=x.d(0);  // Value of df/dx (index 0 of 2)
	f.diff(1,3);
	double dfdy=y.d(0);  // Value of df/dy (index 1 of 2)
	f.diff(2,3);
	double dfdz=z.d(0);
	cout << "f(x,y,z)=" << fval << endl;
	cout << "df/dx(x,x)=" << dfdx << endl;
	cout << "df/dy(x,y)=" << dfdy << endl;
	cout << "df/dy(x,z)=" << dfdz << endl;
	out_(0,0) = (float)dfdx;
	out_(0,1) = (float)dfdy;
	out_(0,2) = (float)dfdz;
}

void ebpd_solver::val(double& out_, std::function< B< F<double> > (const B< F<double> >& x, const B< F<double> >& y, const B< F<double> >& z)> func ,double x0,double x1,double x2 )
{
	B< F<double> > x,y,z,f;     // Declare variables x,y,f
	x=x0;                 // Initialize variable x
	y=x1;                 // Initialize variable y
	z=x2;

	f=func(x,y,z);               // Evaluate function and record DAG
	double fval=f.x().x();     // Value of function
	out_ = fval;
}
void ebpd_solver::ffgradient2(MatrixXX& out_, std::function< B< F<double> > (const B< F<double> >& x, const B< F<double> >& y, const B< F<double> >& z)> func ,double x0,double x1,double x2 )
{
	B< F<double> > x,y,z,f;     // Declare variables x,y,f
	x=x0;                 // Initialize variable x
	y=x1;                 // Initialize variable y
	z=x2;
	cout<<"x "<<x.x().x()<< endl;
	cout<<"y "<<y.x().x()<< endl;
	cout<<"z "<<z.x().x()<< endl;
	x.x().diff(0,3);           // Second order wrt. x
	y.x().diff(1,3);           // Second order wrt. y
	z.x().diff(2,3);           // Second order wrt. y
	f=func(x,y,z);               // Evaluate function and record DAG
	f.diff(0,1);               // Differentiate f
	double fval=f.x().x();     // Value of function
	double dfdx=x.d(0).x();    // Value of df/dx
	double dfdy=y.d(0).x();    // Value of df/dy
	double dfdz=z.d(0).x();    // Value of df/dz
	//	cout << "f(x,y,z)=" << fval << endl;
	cout << "df/dx(x,y,z)=" << dfdx << endl;
	cout << "df/dy(x,y,z)=" << dfdy << endl;
	cout << "df/dy(x,y,z)=" << dfdz << endl;
	out_(0,0) = (float)dfdx;
	out_(0,1) = (float)dfdy;
	out_(0,2) = (float)dfdz;

}
void ebpd_solver::ffHesse(MatrixXX& out_,std::function< B< F<double> > (const B< F<double> >& x, const B< F<double> >& y, const B< F<double> >& z)> func ,double x0,double x1,double x2	)
{
	B< F<double> > x,y,z,f;     // Declare variables x,y,f
	x=x0;                 // Initialize variable x
	y=x1;                 // Initialize variable y
	z=x2;
	cout<<"x "<<x.x().x()<< endl;
	cout<<"y "<<y.x().x()<< endl;
	cout<<"z "<<z.x().x()<< endl;
	x.x().diff(0,3);           // Second order wrt. x
	y.x().diff(1,3);           // Second order wrt. y
	z.x().diff(2,3);           // Second order wrt. y
	f=func(x,y,z);               // Evaluate function and record DAG
	f.diff(0,1);               // Differentiate f
	double fval=f.x().x();     // Value of function
	double dfdx=x.d(0).x();    // Value of df/dx
	double dfdy=y.d(0).x();    // Value of df/dy
	double dfdz=z.d(0).x();    // Value of df/dz
	double dfdxdx=x.d(0).d(0); // Value of df/dxdx
	double dfdxdy=x.d(0).d(1); // Value of df/dxdy
	double dfdxdz=x.d(0).d(2); // Value of df/dxdz
	double dfdydx=y.d(0).d(0); // Value of df/dydx
	double dfdydy=y.d(0).d(1); // Value of df/dydy
	double dfdydz=y.d(0).d(2); // Value of df/dydz
	double dfdzdx=z.d(0).d(0); // Value of df/dzdx
	double dfdzdy=z.d(0).d(1); // Value of df/dzdy
	double dfdzdz=z.d(0).d(2); // Value of df/dzdz
	//	cout << "f(x,y,z)=" << fval << endl;
	cout << "df/dx(x,y,z)=" << dfdx << endl;
	cout << "df/dy(x,y,z)=" << dfdy << endl;
	cout << "df/dz(x,y,z)=" << dfdz << endl;
	cout << "df/dxdx=" << dfdxdx << endl;
	cout << "df/dxdy=" << dfdxdy << endl;
	cout << "df/dxdz=" << dfdxdz << endl;
	cout << "df/dydx=" << dfdydx << endl;
	cout << "df/dydy=" << dfdydy << endl;
	cout << "df/dydz=" << dfdydz << endl;
	cout << "df/dzdx=" << dfdzdx << endl;
	cout << "df/dzdy=" << dfdzdy << endl;
	cout << "df/dzdz=" << dfdzdz << endl;
	out_(0,0) = (float)dfdxdx;
	out_(0,1) = (float)dfdxdy;
	out_(0,2) = (float)dfdxdz;
	out_(1,0) = (float)dfdydx;
	out_(1,1) = (float)dfdydy;
	out_(1,2) = (float)dfdydz;
	out_(2,0) = (float)dfdzdx;
	out_(2,1) = (float)dfdzdy;
	out_(2,2) = (float)dfdzdz;
}
/*void ebpd_solver::ffHesseB(MatrixXX& out_, std::function< B<double> (const B<double>& x, const B<double>& y, const B<double>& z)> func)
{
	B<double> x,y,z,f;     // Declare variables x,y,f
	x=1;                 // Initialize variable x
	y=2;                 // Initialize variable y
	z=3;
	f=func(x,y,z);         // Evaluate function and derivatives


	double fval=f.x();   // Value of function
	f.diff(0,1);
	cout<<"x "<<x.x()<< endl;
	cout<<"y "<<y.x()<< endl;
	cout<<"y "<<z.x()<< endl;
	double dfdx=x.d(0);  // Value of df/dx (index 0 of 2)
	f.diff(1,3);
	double dfdy=y.d(0);  // Value of df/dy (index 1 of 2)
	f.diff(2,3);
	double dfdz=z.d(0);
	cout << "f(x,y,z)=" << fval << endl;
	cout << "df/dx(x,x)=" << dfdx << endl;
	cout << "df/dy(x,y)=" << dfdy << endl;
	cout << "df/dy(x,z)=" << dfdz << endl;
	out_(0,0) = (float)dfdx;
	out_(0,1) = (float)dfdy;
	out_(0,2) = (float)dfdz;
}
*/
void ebpd_solver::newtonSolver(std::function< B< F<double> > (const B< F<double> >& , const B< F<double> >& , const B< F<double> >& )> func )
{
	double stoperror = 1e-15;
	int iter_count = 0;
	double error = 1000;
	double x0,x1,x2;
	x0 = x1 = x2 =0;
	
	MatrixXX gradient;
	gradient.resize(1,3);
	MatrixXX hesse;
	hesse.resize(3,3);
	Vec3 xx ,t ,x_start;
	x_start(0,0) = x0;
	x_start(1,0) = x1;
	x_start(2,0) = x2;
	while( error > stoperror && iter_count < 100)
	{
	 	ffgradient2(gradient,func,x_start(0,0),x_start(1,0),x_start(2,0) );
	 	ffHesse(hesse,func, x_start(0,0),x_start(1,0),x_start(2,0) );
		printMatrix(gradient);
		printMatrix(hesse);
		if( hesse.determinant() == 0) break;
		else
		{
			xx = x_start -hesse.inverse()*gradient.transpose();
		}
		t = xx -x_start;
		x_start = xx;
		iter_count++;
		error = t.norm();
		cout<<"error "<<error<<endl;

	};
	cout<<"iter count : "<< iter_count <<endl;
	double minval;
	val(minval ,func,x_start(0,0),x_start(1,0),x_start(2,0));
	cout<<"min: "<<minval<<endl;
	cout<<"¼«Öµµã: "<<x_start(0,0)<<" "<<x_start(1,0)<<" "<<x_start(2,0)<<endl;
}
