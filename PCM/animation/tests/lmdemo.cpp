
#if 0
// A little test to deform a line mesh composed of several segments
// we optimize the energy function depending on the edges and angles

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/AutoDiff>
#include <iostream>
#include <QTime>
#include "toolbox/portable_includes/port_glew.h"


// -----------------------------------------------------------------------------

/// Computes the energy terms into costs.
/// @param costs: vector of costs.
/// The first part contains terms related to edge length variation,
/// and the second to angle variations.
template<typename Scalar>
void ikLikeCosts(const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& curve,
                 const Eigen::VectorXd& targetAngles,
                 const Eigen::VectorXd& targetLengths,
                 double beta,
                 Eigen::Matrix<Scalar,Eigen::Dynamic,1>& costs)
{
    using namespace Eigen;
    using std::atan2;

    typedef Matrix<Scalar,2,1> Vec2;
    int nb = curve.size()/2;

    costs.setZero();
//    std::cout << "costs size: " << costs.rows() << std::endl;

    for(int k = 1; k < nb - 1; ++k)
    {
        // Look up curves values three by three
        Vec2 pk0 = curve.template segment<2>( 2 * (k-1) );
        Vec2 pk  = curve.template segment<2>( 2 * k     );
        Vec2 pk1 = curve.template segment<2>( 2 * (k+1) );

        if(k + 1 < nb - 1) {
            // Compute edge length costs store it in the second part of costs
            costs((nb-2) + (k-1)) = (pk1 - pk).norm() - targetLengths(k-1);
        }

        // Compute edge angles costs store it in the second part of costs
        Vec2 v0 = (pk  - pk0).normalized();
        Vec2 v1 = (pk1 - pk ).normalized();

        // signed angle between v0, v1
        Scalar at = atan2(-v0.y() * v1.x() + v0.x() * v1.y(),
                           v0.x() * v1.x() + v0.y() * v1.y() );

        costs(k-1) = beta * at - targetAngles(k-1);
    }
}

// -----------------------------------------------------------------------------

// Generic functor
template<typename _Scalar>
struct Functor
{
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = Eigen::Dynamic,
        ValuesAtCompileTime = Eigen::Dynamic
                          };
    typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

    const int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}

    /// @param input : total number of unkown in x vector.
    /// @param values : total number of terms in the energy sum.
    Functor(int inputs,
            int values) :
        m_inputs(inputs),
        m_values(values)
    {

    }

    int inputs() const { return m_inputs; } // number of degree of freedom (= 2*nb_vertices)

    // This method must be defined for LevenbergMarquardt
    int values() const { return m_values; } // number of energy terms (= nb_vertices(for angles) + nb_edges(for lengths))

    // you should define that in the subclass :
    //    void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
};

// -----------------------------------------------------------------------------

// Specialized functor warping the ikLikeCosts function
struct iklike_functor : Functor<double>
{
    typedef Eigen::AutoDiffScalar<Eigen::Matrix<Scalar,Eigen::Dynamic,1> > ADS;
    typedef Eigen::Matrix<ADS, Eigen::Dynamic, 1> VectorXad;

    // pfirst and plast are the two extremities of the curve
    iklike_functor(const Eigen::VectorXd& targetAngles,
                   const Eigen::VectorXd& targetLengths,
                   double beta,
                   const Eigen::Vector2d pfirst,
                   const Eigen::Vector2d plast)
        :
          Functor<double>(targetAngles.size()*2-4, targetAngles.size()*2-1),
          m_targetAngles(targetAngles),
          m_targetLengths(targetLengths),
          m_beta(beta),
          m_pfirst(pfirst),
          m_plast(plast)
    {

    }

    // This method must be defined for LevenbergMarquardt
    // input = x = {  ..., x_i, y_i, ....}
    // output = fvec = the value of each term of the energy sum
    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec)
    {

        std::cout << "x size: " << x.rows() << std::endl;
        std::cout << "fvec size: " << fvec.rows() << std::endl;
        std::cout << "values : " << values() << std::endl;

        using namespace Eigen;
        // Build curve:
        //  { m_pfirst - (1, 0), m_pfirst, x, m_plast, m_plast + (1, 0) }

        VectorXd curves(this->inputs()+8/*2 extra points at first and last position*/);

        curves.segment(4, this->inputs()) = x;

        Vector2d d(1,0);
        curves.segment<2>(0)                = m_pfirst - d;
        curves.segment<2>(2)                = m_pfirst;
        curves.segment<2>(this->inputs()+4) = m_plast;
        curves.segment<2>(this->inputs()+6) = m_plast + d;

        ikLikeCosts(curves, m_targetAngles, m_targetLengths, m_beta, fvec);
        return 0;
    }

    // This method must be defined for LevenbergMarquardt
    /// Compute the jacobian into fjac for the current solution x
    /// @param fjac : jacobian matrix of f()
    int df(const Eigen::VectorXd& x, Eigen::MatrixXd& fjac)
    {
        using namespace Eigen;
        VectorXad curves(this->inputs()+8);

        // Compute the derivatives of each degree of freedom
        // -> grad( x_i ) = (0, ..., 0, 1, 0, ..., 0) ; 1 is in position i
        for(int i = 0; i < this->inputs(); ++i)
            curves(4+i) = ADS(x(i), this->inputs(), i);

        Vector2d d(1,0);
        curves.segment<2>(0)                  = (m_pfirst - d).cast<ADS>();
        curves.segment<2>(2)                  = (m_pfirst).cast<ADS>();
        curves.segment<2>(this->inputs() + 4) = (m_plast).cast<ADS>();
        curves.segment<2>(this->inputs() + 6) = (m_plast + d).cast<ADS>();

        VectorXad v(this->values());

        ikLikeCosts(curves, m_targetAngles, m_targetLengths, m_beta, v);

        // copy the gradient of each energy term into the Jacobian
        for(int i = 0; i < this->values(); ++i)
            fjac.row(i) = v(i).derivatives();

        return 0;
    }

    const Eigen::VectorXd& m_targetAngles;
    const Eigen::VectorXd& m_targetLengths;
    double m_beta;
    Eigen::Vector2d m_pfirst, m_plast;
};

// -----------------------------------------------------------------------------

void draw_vecX(const Eigen::VectorXd& res, double depth = 4.)
{
    Eigen::VectorXd x(res.rows());
    for(int i = 0; i < res.rows(); i++){
        x(i) = res(i);
    }

    glLineWidth(2.f );
    glPointSize(5.f);
    glColor3f(1.f, 0.f, 0.f);
    glBegin(GL_LINES);
    for( int i = 0; i < (x.rows()/2) - 1; i++ )
    {
        int a = i;
        int b = i + 1;
        glVertex3d(x(a*2), x(a*2 + 1), depth);
        glVertex3d(x(b*2), x(b*2 + 1), depth);
    }
    glEnd();

    glColor3f(1.f, 1.f, 0.f);
    glBegin(GL_POINTS);
    for( int i = 0; i < (x.rows()/2); i++ )
    {
        int a = i;
        glVertex3d(x(a*2), x(a*2 + 1), depth);
    }
    glEnd();
}

// -----------------------------------------------------------------------------

void ikInterpolation()
{
    Eigen::Vector2d pfirst(-5., 0.);
    Eigen::Vector2d plast ( 5., 0.);

    // rest pose is a straight line starting between first and last point
    const int nb_points = 30;
    Eigen::VectorXd targetAngles (nb_points);
    targetAngles.fill(0);

    Eigen::VectorXd targetLengths(nb_points-1);
    double val = (pfirst-plast).norm() / (double)(nb_points-1);
    targetLengths.fill(val);

    // get initial solution
    Eigen::VectorXd x((nb_points-2)*2);
    for(int i = 1; i < (nb_points - 1); i++)
    {
        double s = (double)i / (double)(nb_points-1);
        x.segment<2>((i-1)*2) = plast * s + pfirst * (1. - s);
    }

    draw_vecX( x );

    static double dx = 0.;
    // move last point
    plast = Eigen::Vector2d(4.+dx, 1.-dx);

    dx += 0.1f;
    dx = dx > 5.f ? 0.f : dx;

    // Create the functor object
    iklike_functor func(targetAngles, targetLengths, /*0.1*/0.1, pfirst, plast);

    // Build the solver
    Eigen::LevenbergMarquardt<iklike_functor> lm( func );

    // adjust tolerance
    lm.parameters.ftol *= 1e-2;
    lm.parameters.xtol *= 1e-2;
    lm.parameters.maxfev = 2000;

    QTime t;
    t.start();
    /*int a = */lm.minimize(x);
    std::cout << "time:" << t.elapsed() << std::endl;
//    std::cerr << "info = " << a  << " " << lm.nfev << " " << lm.njev << " "  <<  "\n";

//    std::cout << pfirst.transpose() << "\n";
    draw_vecX( x, 0.);
//    std::cout << plast.transpose() << "\n";
}

#endif
