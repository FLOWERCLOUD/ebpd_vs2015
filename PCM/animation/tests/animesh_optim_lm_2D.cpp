#if 1
 /*
 * Optimisation test:
 * With bending / streching / area energies available
 * Optim done in 2D with the Levenberg Marquardth method
 */


#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/AutoDiff>
#include <unsupported/Eigen/NumericalDiff>
#include <iostream>
#include <QTime>

// -----------------------------------------------------------------------------

#include "toolbox/utils.hpp"
#include "toolbox/std_utils/vector.hpp"
#include "animesh.hpp"
#include "mesh_utils.hpp"
#include "toolbox/portable_includes/port_glew.h"
#include "debug_ctrl.hpp"
#include "cuda_ctrl.hpp"
using namespace Cuda_ctrl;

// -----------------------------------------------------------------------------

// Forward def from animesh_optim_kers.cu
void compute_pot_launch_ker(const Cuda_utils::DA_Vec3& d_verts,
                            Cuda_utils::DA_float& d_pot);

// -----------------------------------------------------------------------------

// Specialized functor warping the ikLikeCosts function
template<typename _Scalar>
struct Lm_functor{

    typedef _Scalar Scalar;

    const bool _solve_in_2D;

    enum {
        InputsAtCompileTime = Eigen::Dynamic,
        ValuesAtCompileTime = Eigen::Dynamic
    };

    typedef Eigen::Matrix<Scalar, 3, 3> Matrix3x3;
    typedef Eigen::Matrix<Scalar, 3, 1> Vec3_ei;

    typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

    typedef Eigen::AutoDiffScalar<Eigen::Matrix<Scalar,Eigen::Dynamic,1> > ADS;
    typedef Eigen::Matrix<ADS, Eigen::Dynamic, 1> VectorXad;

    typedef Eigen::NumericalDiff<Lm_functor<Scalar>, Eigen::Central > My_numerical_diff;

    // -------------------------------------------------------------------------

    static Vec3_ei to_eigen_vec(const Vec3&    v) { return Vec3_ei(  v.x,  v.y,  v.z ); }
    static Vec3    to_vec_cu   (const Vec3_ei& v) { return Vec3   ( v(0), v(1), v(2) ); }

    // -------------------------------------------------------------------------

    void fill_to_3d_mats()
    {
        const int nb_verts = _animesh->hd_free_vertices.size();
        _to_3d_mats.resize( nb_verts );

        for(int i = 0; i < nb_verts; ++i)
        {

            Matrix3x3 eigen_syst = Matrix3x3::Identity();
            if(_solve_in_2D)
            {
                const int vert_idx = _animesh->hd_free_vertices[i];
                Vec3 grad = _animesh->hd_gradient[vert_idx];
                // Build coord syst with grad aligned with x axis.
                Mat3 syst = Mat3::coordinate_system( -grad );
                eigen_syst.col(0) = to_eigen_vec( syst.z() );
                eigen_syst.col(1) = to_eigen_vec( syst.y() );
                eigen_syst.col(2) = to_eigen_vec( _init_guess[vert_idx] );
            }

            _to_3d_mats[i] = eigen_syst;
        }
    }


    // -------------------------------------------------------------------------

    /// transform the optimised unknown 'x' into the 3D position of the mesh
    /// vertices in h_verts. if not needed just copy 'x' in 'h_verts'
    void convert_solution(const Eigen::VectorXd& x,
                           Cuda_utils::HA_Point3& h_verts)
    {
        const int nb_verts = _animesh->hd_free_vertices.size();

        for(int i = 0; i < nb_verts; ++i)
        {
            const int vert_idx = _animesh->hd_free_vertices[i];
            if(_solve_in_2D) {
                Vec3_ei pos_2d( x( i * 2 ), x( i * 2 + 1 ), 1. );
                h_verts[vert_idx] = to_vec_cu(_to_3d_mats[i] * pos_2d).to_point3();
            } else {
                assert(nb_verts*3 == x.rows() );
                Point3 pt(x( i * 3 ), x( i * 3 + 1 ), x( i * 3 + 2));
                h_verts[vert_idx] = pt;
            }
        }
    }

    // -------------------------------------------------------------------------

    void init_solution(Eigen::VectorXd& x,
                       const Cuda_utils::HA_Point3& h_verts)
    {
        const int nb_free_verts = _animesh->hd_free_vertices.size();
        if(!_solve_in_2D)
        {
            assert(nb_free_verts*3 == x.rows());
            for(int i = 0; i < nb_free_verts; ++i)
            {
                const int vert_id = _animesh->hd_free_vertices[i];
                Point3 v = h_verts[vert_id];
                x(i*3) = v.x; x(i*3+1) = v.y; x(i*3+2) = v.z;
            }

        } else {
            assert(nb_free_verts*2 == x.size());
            x = Eigen::VectorXd::Zero( nb_free_verts * 2 );
        }
    }

    // -------------------------------------------------------------------------

    // pfirst and plast are the two extremities of the curve
    /// @param init_guess :
    /// every mesh vertices in initial position before optimizing
    Lm_functor( Animesh* animesh,
                const Vec3* init_guess,
                bool solve) :
        _solve_in_2D(solve),
        _animesh(animesh),
        _init_guess(init_guess)
    {
        const int nb_verts = _animesh->hd_free_vertices.size();

        fill_to_3d_mats();

        // Number of energy terms
        _values = 0.;
        const int nb_tris  = _animesh->hd_free_triangles.size();
        _values += nb_tris;  // Area
        const int nb_edges = _animesh->hd_free_edges.size();
//      _values += nb_edges; // Bending
        _values += nb_edges; // Edge length
//        _values += nb_verts; // vertex deviation
//       _values += nb_verts; // vertex mvc dev
//        _values += nb_verts; // vertex angular

        // Number of unknown
        _inputs = nb_verts * (_solve_in_2D ? 2 : 3);

        // We have to garantee this:
        assert(_values >= _inputs);

        // Some intermediate buffers
        // Allocate host mem
        hd_pot.  malloc( nb_verts );
        hd_verts.malloc( nb_verts );
        // Allocate device mem
        hd_pot.  update_device_mem();
        hd_verts.update_device_mem();

        //_x_auto_diff.resize( inputs() );
        _f_terms_buff.resize( values() );
        _f_x_minus_h .resize( values() );
        _f_x_plus_h  .resize( values() );
        _x_buff      .resize( inputs() );
        _grad_buff   .resize( inputs() );
    }

    // -------------------------------------------------------------------------

    /// Retreive vertex position in the vector 'x'. If the vertex is not in 'x'
    /// it means we are at a boundary.
    template<typename Real>
    Vec3 vert_anim_pos(const Eigen::Matrix<Real, Eigen::Dynamic, 1>& x,
                          int vert_id) const
    {
        int free_id = _animesh->hd_map_verts_to_free_vertices[vert_id];

        if( free_id != -1 )
        {
            if(_solve_in_2D){
                Vec3_ei pos( x(free_id * 2), x(free_id * 2 + 1), 1.);
                return to_vec_cu( _to_3d_mats[free_id] * pos );
            }else{
                return Vec3(x( free_id * 3 ), x( free_id * 3 + 1 ), x( free_id * 3 + 2));
            }
        }
        else
            return _init_guess[vert_id];
    }

    // -------------------------------------------------------------------------

    template<typename Real>
    float tri_area(const Eigen::Matrix<Real, Eigen::Dynamic, 1>& x,
                   int tri_idx) const
    {
        const Mesh& m = *(_animesh->get_mesh());
        const EMesh::Tri_face t = m.get_tri( tri_idx );
        const Vec3 va = vert_anim_pos(x, t.a );
        const Vec3 vb = vert_anim_pos(x, t.b );
        const Vec3 vc = vert_anim_pos(x, t.c );

        const Vec3 edge0 = vb - va;
        const Vec3 edge1 = vc - va;

        return (edge0.cross(edge1)).norm() / 2.f;
    }

    // -------------------------------------------------------------------------

    template<typename Real>
    Vec3 tri_normal(const Eigen::Matrix<Real, Eigen::Dynamic, 1>& x,
                       int tri_idx) const
    {
        const Mesh& m = *(_animesh->get_mesh());
        const EMesh::Tri_face t = m.get_tri( tri_idx );
        const Vec3 va = vert_anim_pos(x, t.a );
        const Vec3 vb = vert_anim_pos(x, t.b );
        const Vec3 vc = vert_anim_pos(x, t.c );

        const Vec3 edge0 = vb - va;
        const Vec3 edge1 = vc - va;

        return (edge0.cross(edge1)).normalized();
    }

    // -------------------------------------------------------------------------

    template<typename Real>
    float edge_len(const Eigen::Matrix<Real, Eigen::Dynamic, 1>& x,
                   int edge_idx) const
    {
        const Mesh& m = *(_animesh->get_mesh());
        const EMesh::Edge e = m.get_edge( edge_idx );
        return (vert_anim_pos(x, e.b) - vert_anim_pos(x, e.a)).norm();
    }

    // -------------------------------------------------------------------------

    float edge_len(const Vec3* pos, int edge_idx) const
    {
        const Mesh& m = *(_animesh->get_mesh());
        const EMesh::Edge e = m.get_edge( edge_idx );
        return (pos[e.b] - pos[e.a]).norm();
    }

    // -------------------------------------------------------------------------

    template<typename Real>
    float edge_bending(const Eigen::Matrix<Real, Eigen::Dynamic, 1>& x,
                       int edge_idx) const
    {
        const Mesh& m = *(_animesh->get_mesh());
        const EMesh::Edge e = m.get_edge( edge_idx );

        const std::vector<int>& tris = m.get_edge_shared_tris( edge_idx );
        if( tris.size() != 2) return 0.f;

        Vec3 n0 = tri_normal(x, tris[0]);
        Vec3 n1 = tri_normal(x, tris[1]);

        Vec3 ref = vert_anim_pos(x, e.b) - vert_anim_pos(x, e.a);
        // Signed angle between n0 and n1 projected on normal plane 'ref'
        return ref.signed_angle( n0, n1 );
    }

    // -------------------------------------------------------------------------

    // evaluate the potential at 'x' and return result in 'pot'
    template<typename Real>
    void compute_potential( const Eigen::Matrix<Real,Eigen::Dynamic,1>& x,
                            Cuda_utils::HDA_float& pot) const
    {
        const int nb_verts = _animesh->hd_free_vertices.size();
        for(int i = 0; i < nb_verts; ++i)
            hd_verts[i] = vert_anim_pos(x, _animesh->hd_free_vertices[i]);

        hd_verts.update_device_mem();
        compute_pot_launch_ker(hd_verts.device_array(), pot.device_array());
        pot.update_host_mem();
    }

    // -------------------------------------------------------------------------

    template<typename Real>
    int fill_area_energy(Eigen::Matrix<Real,Eigen::Dynamic,1>& fvec,
                          const Eigen::Matrix<Real, Eigen::Dynamic, 1>& x,
                          float weight,
                          int offset) const
    {
        const int nb_tris = _animesh->hd_free_triangles.size();
        for(int i = 0; i < nb_tris; ++i)
        {
            int   tri_idx   = _animesh->hd_free_triangles[i];
            float rest_area = _animesh->hd_tri_areas[tri_idx];
            float curr_area = tri_area(x, tri_idx);

            fvec(offset + i) = weight * (curr_area - rest_area);
        }
        return nb_tris;
    }

    // -------------------------------------------------------------------------

    template<typename Real>
    int fill_length_energy(Eigen::Matrix<Real,Eigen::Dynamic,1>& fvec,
                           const Eigen::Matrix<Real, Eigen::Dynamic, 1>& x,
                           float weight,
                           int offset) const
    {
        const int nb_edges = _animesh->hd_free_edges.size();
        for(int i = 0; i < nb_edges; ++i)
        {
            int   edge_idx = _animesh->hd_free_edges[i];
            float rest_len = _animesh->hd_edge_lengths[edge_idx];
            float curr_len = edge_len(x, edge_idx);
            float diff     = (curr_len - rest_len);
            fvec( offset + i ) = diff * weight;
        }
        return nb_edges;
    }

    // -------------------------------------------------------------------------

    template<typename Real>
    int fill_bend_energy(Eigen::Matrix<Real,Eigen::Dynamic,1>& fvec,
                          const Eigen::Matrix<Real, Eigen::Dynamic, 1>& x,
                          float weight,
                          int offset) const
    {
        const int nb_edges = _animesh->hd_free_edges.size();
        for(int i = 0; i < nb_edges; ++i)
        {
            int   edge_idx = _animesh->hd_free_edges[i];
            float rest_bend = _animesh->hd_edge_bending[edge_idx];
            float curr_bend = edge_bending(x, edge_idx);
            fvec( offset + i ) = weight * (curr_bend - rest_bend)*(curr_bend - rest_bend); //* std::sqrt( edge_len(x, edge_idx) );
        }

        return nb_edges;
    }

    // -------------------------------------------------------------------------

    template<typename Real>
    int fill_deviation_energy(Eigen::Matrix<Real,Eigen::Dynamic,1>& fvec,
                               const Eigen::Matrix<Real, Eigen::Dynamic, 1>& x,
                               float weight,
                               int offset) const
    {
        const int nb_free_verts = _animesh->hd_free_vertices.size();
        for(int i = 0; i < nb_free_verts; ++i)
        {
            const int vert_idx = _animesh->hd_free_vertices[i];
            fvec( offset + i ) = weight * ( _init_guess[vert_idx] - vert_anim_pos(x, vert_idx) ).norm();
        }

        return nb_free_verts;
    }

    // -------------------------------------------------------------------------

    template<typename Real>
    int fill_potential_energy(Eigen::Matrix<Real,Eigen::Dynamic,1>& fvec,
                               const Eigen::Matrix<Real,Eigen::Dynamic,1>& x,
                               float weight,
                               int offset) const
    {
        const int nb_verts = _animesh->hd_free_vertices.size();
        compute_potential( x, hd_pot );
        for(int i = 0; i < nb_verts; ++i){
            fvec( offset + i ) = weight * (hd_pot[i] - _animesh->hd_free_base_potential[i]);
        }
        return nb_verts;
    }

    // -------------------------------------------------------------------------

    template<typename Real>
    int fill_angular_energy(Eigen::Matrix<Real,Eigen::Dynamic,1>& fvec,
                            const Eigen::Matrix<Real,Eigen::Dynamic,1>& x,
                            float weight,
                            int offset) const
    {
        const int nb_free_verts = _animesh->hd_free_vertices.size();
        for(int i = 0; i < nb_free_verts; ++i)
        {
            const int vert_idx = _animesh->hd_free_vertices[i];
            const Mesh& m = *(_animesh->get_mesh());

            Vec3 pos = vert_anim_pos(x, vert_idx);

            float sum = 0.f;
            int dep      = m.get_1st_ring_offset(vert_idx*2    );
            int nb_neigh = m.get_1st_ring_offset(vert_idx*2 + 1);
            int end      = dep + nb_neigh;
            for(int n = dep; n < (dep+nb_neigh); n++)
            {
                int curr = m.get_1st_ring( n );
                int next = m.get_1st_ring( (n+1) >= end  ? dep : n+1 );

                Vec3 edge0 = vert_anim_pos(x, curr) - pos;
                Vec3 edge1 = vert_anim_pos(x, next) - pos;

                sum += acos(edge0.normalized().dot(edge1.normalized())); //_animesh->hd_gradient[vert_idx].signed_angle( edge0.normalized(), edge1.normalized() );
            }

//            std::cout << "base" << _animesh->d_base_gradient.fetch(vert_idx) << std::endl;

//            std::cout << "f" << _animesh->hd_gradient[vert_idx] - _animesh->d_base_gradient.fetch(vert_idx) << std::endl;
//            std::cout << "a" << sum << std::endl;
//            std::cout << "b" << _animesh->hd_sum_angles[vert_idx] << std::endl;

            fvec( offset + i ) = weight * (_animesh->hd_sum_angles[vert_idx] - sum);
        }
        return nb_free_verts;
    }

    // -------------------------------------------------------------------------

    template<typename Real>
    int fill_dev_mvc_energy(Eigen::Matrix<Real,Eigen::Dynamic,1>& fvec,
                            const Eigen::Matrix<Real,Eigen::Dynamic,1>& x,
                            float weight,
                            int offset) const
    {
        const int nb_verts = _animesh->hd_free_vertices.size();
        for(int i = 0; i < nb_verts; ++i)
        {
            const int vert_idx = _animesh->hd_free_vertices[i];
            const Mesh& m = *(_animesh->get_mesh());

            Vec3 grad = _animesh->hd_gradient[vert_idx];
            Vec3 cog(0.f, 0.f, 0.f);
            float sum = 0.f;
            int dep      = m.get_1st_ring_offset(vert_idx*2    );
            int nb_neigh = m.get_1st_ring_offset(vert_idx*2 + 1);
            for(int n = dep; n < (dep+nb_neigh); n++)
            {
                int   index_neigh = m.get_1st_ring(n);
                float mvc         = _animesh->hd_1st_ring_mvc[n];

                Vec3 v = vert_anim_pos(x, index_neigh);
                sum += mvc;
                cog =  cog + grad.proj_on_plane(_init_guess[vert_idx].to_point3(), v.to_point3()) * mvc;
            }

            // Check mvc values are correct
            if( fabs(sum) < 0.00001f ){
                fvec( offset + i ) = 0.;
            } else {
                cog = cog / sum;
                // this force the smoothing to be only tangential :
                Vec3 cog_proj = cog; //grad.proj_on_plane(_init_guess[vert_idx].to_point3(), cog.to_point3());

                Real val = (cog_proj - vert_anim_pos(x, vert_idx)).norm();
                fvec( offset + i ) = weight * val/**val*val*val*/;
            }

            //std::cout << _animesh->hd_velocity[vert_idx].norm()/*fvec( offset + i )*/ << ", ";
        }
        //std::cout << std::endl;

        return nb_verts;
    }

    // -------------------------------------------------------------------------

    /// This method must be defined for LevenbergMarquardt
    /// @param input = x = {  ..., x_i, y_i, z_i ....} mesh vert
    /// output = fvec = the value of each term of the energy sum fvec size == values()
    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
    {
        int offset = 0;
        offset += fill_length_energy   ( fvec, x, /*_debug._val0     */ 1. , offset );
//        offset += fill_angular_energy  ( fvec, x, 5./*_debug._val0       */, offset );
//        offset += fill_bend_energy     ( fvec, x, 1./*_debug._tab_vals[0]*/, offset );
//        offset += fill_dev_mvc_energy  ( fvec, x, 1./*_debug._tab_vals[1]*/, offset );
//        offset += fill_deviation_energy( fvec, x, _debug._tab_vals[1], offset );
        offset += fill_area_energy( fvec, x, 1./*_debug._val1*/, offset);
//        offset += fill_potential_energy( fvec, x, _debug._val1, offset);

        return 0;
    }

    // -------------------------------------------------------------------------

    /// This method must be defined for LevenbergMarquardt
    /// Compute the jacobian into fjac for the current solution x
    /// @param fjac : jacobian matrix of f()
    int df(const Eigen::VectorXd& x, Eigen::MatrixXd& fjac) const
    {
        //QTime t; t.start();
#if 0
        My_numerical_diff num_diff( *this, 0.00001 );
        num_diff.df(x, fjac);
#endif
        Scalar h = 0.0001;
        _x_buff = x;

        double tmp;
        for(int j = 0; j < inputs(); ++j)
        {
            tmp = x(j);
            // compute f( x + h )
            _x_buff[j] = tmp + h;
            (*this)(_x_buff, _f_x_plus_h);
            // compute f( x - h )
            _x_buff[j] = tmp - h;
            (*this)(_x_buff, _f_x_minus_h);

            _x_buff[j] = x[j];
            fjac.col(j) = (_f_x_plus_h - _f_x_minus_h) / (2. * h);
        }
        //std::cout << "jac time: " << (double)t.elapsed() << " ms" << std::endl;

        return 0;
    }

    // -------------------------------------------------------------------------

    float eval_sum(const Eigen::VectorXd& x)
    {
        (*this)(x, _f_terms_buff);
        Scalar s = 0.;
        for (int i = 0; i < values(); ++i)
            s += _f_terms_buff(i) * _f_terms_buff(i); //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$DEBUG
        return s;
    }

    // -------------------------------------------------------------------------

    void eval_grad_sum(const Eigen::VectorXd& x, Eigen::VectorXd& grad)
    {
        Scalar h = 0.001;
        _x_buff = x;

        double tmp;
        for(int j = 0; j < inputs(); ++j)
        {
            tmp = x(j);
            // compute f( x + h )
            _x_buff[j] = tmp + h;
            Scalar f_x_plus_h = eval_sum(_x_buff);
            // compute f( x - h )
            _x_buff[j] = tmp - h;
            Scalar f_x_minus_h = eval_sum(_x_buff);

            _x_buff[j] = x[j];
            grad(j) = (f_x_plus_h - f_x_minus_h) / (2. * h);
        }
    }

    // -------------------------------------------------------------------------

    void time_step(Eigen::VectorXd& x)
    {

#if 1
        const float damping = 0.5;//Cuda_ctrl::_debug._val0;
        const float step    = 0.01;//Cuda_ctrl::_debug._val1;
        const float force   = Cuda_ctrl::_debug._tab_vals[0];

        eval_grad_sum(x, _grad_buff);

        //std::cout << "grad " << _grad_buff << std::endl;
        //float n = _grad_buff.norm();
        //_grad_buff /= n;
        //_grad_buff *= 0.1f;

        const int nb_free_verts = _animesh->hd_free_vertices.size();
        for(int i = 0; i < nb_free_verts; ++i)
        {
            const int vert_idx = _animesh->hd_free_vertices[i];

            Vec3 pos( x(i*3), x(i*3 + 1), x(i*3 + 2) );

            Vec3& old_pos = _animesh->hd_prev_output_vertices[vert_idx];
            Vec3 tmp = pos;
            Vec3 grad( _grad_buff[i*3], _grad_buff[i*3 + 1], _grad_buff[i*3 + 2] );
            Vec3 a = -Vec3::unit_z() * force;
            a = -grad*0.5;

            Vec3 new_pos = pos + /* (pos - old_pos) * (1.-damping)  + */ a * step;
            old_pos = tmp;

            x(i*3    ) = new_pos.x;
            x(i*3 + 1) = new_pos.y;
            x(i*3 + 2) = new_pos.z;
        }

        _animesh->hd_prev_output_vertices.update_device_mem();
#endif
    }

    // -------------------------------------------------------------------------

    bool is_deformable(int vert_id) const {
        return _animesh->hd_map_verts_to_free_vertices[vert_id] != -1;
    }

    // -------------------------------------------------------------------------

    /// number of degree of freedom (= (_solve_in_2D ? 2 : 3 ) * nb_vertices)
    int inputs() const { return _inputs; }

    /// This method must be defined for LevenbergMarquardt
    /// to know the number of terms of the energy sum
    /// number of energy terms == nb_terms areas + nb termes edges + nb terms potential)
    int values() const { return _values; }

    // -------------------------------------------------------------------------
    /// @name attributes
    // -------------------------------------------------------------------------
private:
    int _inputs; ///< total number of unkown in x vector. (i.e nb_verts*(2|3))
    int _values; ///< total number of terms in the energy sum.

    /// GPU Buffer to store the potential at each vertex (size == _inputs/(2|3))
    mutable Cuda_utils::HDA_float   hd_pot;
    /// GPU Buffer to store the vertex position (size == _inputs/(2|3))
    mutable Cuda_utils::HDA_Vec3 hd_verts;

    /// Buffer of unkown with auto diff capabilities
    //VectorXad _x_auto_diff;
    mutable Eigen::VectorXd _f_terms_buff;
    mutable Eigen::VectorXd _f_x_minus_h, _f_x_plus_h;
    mutable Eigen::VectorXd _x_buff;
    mutable Eigen::VectorXd _grad_buff;

    /// Mesh to be deformed
    Animesh* _animesh;

    /// every vertices positions before optimizing
    /// (used to get boundary positions which are not part of the unknown 'x')
    const Vec3* _init_guess;

    /// The matrices to transform 2d points to 3d is _solve_in_2D true
    std::vector<Matrix3x3> _to_3d_mats;
};

// -----------------------------------------------------------------------------

static inline
std::string to_string( Eigen::LevenbergMarquardtSpace::Status state)
{
    std::string str;
    using namespace Eigen::LevenbergMarquardtSpace;
    switch(state){
    case NotStarted:                        str = "NotStarted";                        break;
    case Running:                           str = "Running";                           break;
    case ImproperInputParameters:           str = "ImproperInputParameters";           break;
    case RelativeReductionTooSmall:         str = "RelativeReductionTooSmall";         break;
    case RelativeErrorTooSmall:             str = "RelativeErrorTooSmall";             break;
    case RelativeErrorAndReductionTooSmall: str = "RelativeErrorAndReductionTooSmall"; break;
    case CosinusTooSmall:                   str = "CosinusTooSmall";                   break;
    case TooManyFunctionEvaluation:         str = "TooManyFunctionEvaluation";         break;
    case FtolTooSmall:                      str = "FtolTooSmall";                      break;
    case XtolTooSmall:                      str = "XtolTooSmall";                      break;
    case GtolTooSmall:                      str = "GtolTooSmall";                      break;
    case UserAsked:                         str = "UserAsked";                         break;
    }
    return str;
}

// -----------------------------------------------------------------------------

void print_infos(Animesh* animesh)
{
    const int nb_tris  = animesh->hd_free_triangles.size();
    const int nb_edges = animesh->hd_free_edges.size();
    const int nb_verts = animesh->hd_free_vertices.size();

    std::cout << "nb tris"  << nb_tris  << std::endl;
    std::cout << "nb_edges" << nb_edges << std::endl;
    std::cout << "nb_verts" << nb_verts << std::endl;
}

// -----------------------------------------------------------------------------

/// @param verts : list of vertices with incremental position, they will be
/// deformed in place
void test_non_linear_deform_phyX(Animesh* animesh,
                                 Cuda_utils::Device::Array<Point3>& d_verts)
{
    //print_infos( animesh );
    QTime t; t.start();

    const int nb_free_verts = animesh->hd_free_vertices.size();

    /////////
    // Allocate memory if needed
    static Cuda_utils::HA_Point3 h_verts;
    if( h_verts.size() != d_verts.size() )
        h_verts.malloc( d_verts.size() );
    h_verts.copy_from( d_verts );

    const bool solve_in_2D = true;

    // alloc vec of unknowns
    static Eigen::VectorXd x;
    if( x.rows() != nb_free_verts * (solve_in_2D ? 2 : 3 ) )
        x.resize( nb_free_verts * (solve_in_2D ? 2 : 3 ) );

    // Create the functor object
    Lm_functor<double> func(animesh, (const Vec3*)h_verts.ptr(), solve_in_2D);
    // init vec of unknowns
    func.init_solution(x, h_verts);

    //func.time_step( x );
#if 1
    #if 0
    // Build the solver
    Eigen::LevenbergMarquardt<Lm_functor<double> > lm( func );
    // adjust tolerance
    lm.parameters.ftol = 1e-3;
    lm.parameters.xtol = 1e-3;
    lm.parameters.maxfev = 200;
    //////////
    // Start solving
    //Eigen::LevenbergMarquardtSpace::Status a = lm.lmdif1(func, x, 0);
    Eigen::LevenbergMarquardtSpace::Status a = lm.minimize( x );
    std::cerr << "info = " << to_string(a)  << " " << lm.nfev << " " << lm.njev << " "  <<  "\n";
    #else

    static Eigen::VectorXd grad;
    if( grad.rows() != func.inputs() )
        grad.resize( func.inputs() );

    func.eval_grad_sum(x, grad);
    x -= grad * 0.5 * 0.01;

    /*
    int acc = 0;
    double curr = func.eval_sum(x);
    double prev = 0.;
    double n = 10.;
    while( n > 0.001 && std::fabs(curr-prev) > 0.01 && acc < 10)
    {
        func.eval_grad_sum(x, grad);
        //n = grad.norm();
        //grad *= 1./n;

        x -= grad * 0.5 * 0.01;

        prev = curr;
        curr = func.eval_sum(x);
        acc++;
    }

    std::cout << "grad descent nb iters: " << acc << std::endl;
    */
    #endif
#endif

    ///////////
    // Retreive verts pos
    func.convert_solution(x, h_verts);
    d_verts.copy_from( h_verts );

    //std::cout << "time:" << (double)t.elapsed()/1000. << "sec" << std::endl;
}

// -----------------------------------------------------------------------------

#include <Eigen/SVD>

void Animesh::compute_SVD_rotations(Vec3* verts)
{
    Cuda_utils::HA_float smooth_fact(_mesh->get_nb_vertices());
    smooth_fact.copy_from( d_smooth_factors_conservative );

    Eigen::Matrix3d eye = Eigen::Matrix3d::Identity();

    for(int i = 0; i < _mesh->get_nb_vertices(); ++i)
    {
        //if( smooth_fact[i] < 0.5f ) continue;

        int valence = _mesh->get_1st_ring_offset( 2 * i + 1);
        int degree = 0;

        Eigen::MatrixXd P(3, valence), Q(3, valence);

        const int dep = _mesh->get_1st_ring_offset( 2 * i );
        const int end = dep + valence;
        int acc = 0;
        for(int n = dep; n < end; n++, acc++)
        {
            int j = _mesh->get_1st_ring( n );

            // eij = pi - pj
            //compute: P_i * D_i in the paper
            Vec3 v = (_mesh->get_vertex(i) - _mesh->get_vertex(j)) * hd_1st_ring_cotan[n];
            P.col(degree) = Eigen::Vector3d(v.x, v.y, v.z);

            // eij' = pi' - pj'
            //compute: P_i'
            v = verts[i] - verts[j];
            Q.col(degree++) = Eigen::Vector3d(v.x, v.y, v.z);
        }
        // Compute the 3 by 3 covariance matrix:
        // actually S = (P * W * Q.t()); W is already considerred in the previous step (P=P*W)
        Eigen::MatrixXd S = (P * Q.transpose());


        // Compute the singular value decomposition S = UDV.t
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeThinU | Eigen::ComputeThinV); // X = U * D * V.t()

        Eigen::MatrixXd V = svd.matrixV();
        Eigen::MatrixXd Ut = svd.matrixU().transpose();

        eye(2,2) = (V * Ut).determinant();	// remember: Eigen starts from zero index

        // V*U.t may be reflection (determinant = -1). in this case, we need to change the sign of
        // column of U corresponding to the smallest singular value (3rd column)
        Eigen::Matrix3d rot = (V * eye * Ut);


        hd_verts_3drots[i] = Mat3(rot(0,0), rot(0,1), rot(0,2),
                                  rot(1,0), rot(1,1), rot(1,2),
                                  rot(2,0), rot(2,1), rot(2,2) ); //Ri = (V * eye * U.t());
    }

    hd_verts_3drots.update_device_mem();
}

// -----------------------------------------------------------------------------

void Animesh::spring_relaxation(Vec3* h_verts, int nb_iter)
{
    for(int n = 0; n < nb_iter; ++n)
    {
        const Mesh& m = *get_mesh();

        const int nb_edges = hd_free_edges.size();
        for(int i = 0; i < nb_edges; ++i)
        {
            int edge_idx = hd_free_edges[i];

            const EMesh::Edge e = m.get_edge( edge_idx );

            float rest_len = hd_edge_lengths[edge_idx];
            Vec3& posA = h_verts[e.a];
            Vec3& posB = h_verts[e.b];

            float curr_len = (posA - posB).norm();

            Vec3 dir = posB - posA;
            Vec3 correction = dir * (1. - rest_len/curr_len);

            bool state_a = hd_map_verts_to_free_vertices[e.a] != -1;
            bool state_b = hd_map_verts_to_free_vertices[e.b] != -1;

            Vec3 half = correction * 0.5;

            if(state_a) posA += half;
            if(state_b) posB -= half;
        }
    }
}

// Forward def from animesh_optim.cpp
void test_non_linear_deform(Animesh* animesh,
                            Cuda_utils::Device::Array<Point3>& d_verts);

void test_non_linear_deform_phyX(Animesh* animesh,
                                 Cuda_utils::Device::Array<Point3>& d_verts);

// -----------------------------------------------------------------------------

#include "skeleton.hpp"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

Eigen::SparseLU< Eigen::SparseMatrix<float> > g_solver;
Eigen::SparseMatrix<float> g_mat_to_solve;
Eigen::SparseMatrix<float> g_mat_At;
Eigen::VectorXf g_vec_B[3];
Eigen::VectorXf g_vec_X[3];


//void test_mult(Cuda_utils::HDA_float& B, int size)
//{
//    B.update_host_mem();
//    Eigen::VectorXd ei_vec_B(size);
//    for(int i = 0; i < size; i++) {
//        ei_vec_B[i] = B[i];
//    }

//    ei_vec_B = g_mat_A.transpose() * ei_vec_B;

//    for(int i = 0; i < size; i++) {
//        B[i] = ei_vec_B[i];
//    }

//    B.update_device_mem();
//}

void Animesh::init_cotan_weights()
{
    // Compute upload contan weights
    std::vector<std::vector<float> > cotans;
    Mesh_utils::laplacian_cotan_weights(*_mesh, cotans);
    std::vector<float> cotans_flat;
    Std_utils::flatten(cotans, cotans_flat);
    hd_1st_ring_cotan.malloc_hd( cotans_flat.size() );
    hd_1st_ring_cotan.copy_from_hd( cotans_flat );

#if 0

    std::cout << "WARNING trying to factor cotan matrix" << std::endl;
    ///////////////////
    // Fill matrix
    ///////////////////

    // Fill sparse laplacian matrix (without areas)
    typedef Eigen::Triplet<float> ETriplet;
    std::vector< ETriplet > triplet_list;
    int nb_verts = _mesh->get_nb_vertices();
    for(int i = 0; i < nb_verts; ++i)
    {
        if( hd_ssd_interpolation_factor[i] < 0.9f )
        {

            float weight = 0.0;
            int dep = _mesh->get_1st_ring_offset(i*2);
            for(unsigned j = 0; j < cotans[i].size(); j++)
            {
                weight += cotans[i][j];
                triplet_list.push_back(ETriplet(i, _mesh->get_1st_ring(dep + j), -cotans[i][j]));
            }
            triplet_list.push_back( ETriplet(i, i, weight) );
        }
        else
            triplet_list.push_back( ETriplet(i, i, 1.f) );
    }

    #if 1
    for(int i = 0; i < nb_verts; ++i)
    {
        triplet_list.push_back( ETriplet(i+nb_verts, i, 1.f) );
        triplet_list.push_back( ETriplet(i+nb_verts, i+nb_verts, -1.f) );
        triplet_list.push_back( ETriplet(i+nb_verts*2, i+nb_verts, 1.f) );
    }
    #endif

    Eigen::SparseMatrix<float> A(nb_verts * 3, nb_verts * 2);
    A.setFromTriplets(triplet_list.begin(), triplet_list.end());

    // Preconditionning the matrix with At;
    g_mat_At = A.transpose();
    g_mat_to_solve = g_mat_At * A;

    g_solver.compute( g_mat_to_solve );

    for(int i = 0; i < 3; ++i){
        g_vec_B[i] = Eigen::VectorXf::Zero(nb_verts * 3);
        g_vec_X[i] = Eigen::VectorXf::Zero(nb_verts * 2);
    }
    hd_vec_B.malloc_hd( nb_verts * 3 );

    //std::cout << "A" << Eigen::MatrixXd( A ) << std::endl << std::endl;
    //std::cout << "B" << Eigen::MatrixXd( B ) << std::endl << std::endl;


    ///////////////////
    // Upload matrix to GPU
    ///////////////////
    #if 0
    std::vector<float> diag_vals( nb_verts );
    std::vector<std::vector< std::pair<float, int> > > row_vals(nb_verts);

    for(int k = 0; k < g_mat_to_solve.outerSize(); ++k)
    {
        for(Eigen::SparseMatrix<float>::InnerIterator it(g_mat_to_solve, k); it; ++it)
        {
            if(it.row() == it.col())
                diag_vals[it.row()] = it.value();
            else
                row_vals[it.row()].push_back( std::make_pair(it.value(), it.col()) );
        }
    }
    // upload B to GPU:
    std::vector< std::pair<float, int> > cotans_squared_flat;
    std::vector<int> cotans_squared_flat_offset;
    Std_utils::flatten(row_vals, cotans_squared_flat, cotans_squared_flat_offset);
    hd_sparse_mat_vals.malloc_hd( cotans_squared_flat.size() );
    hd_sparse_mat_vals.copy_from_hd( cotans_squared_flat );

    hd_sparse_mat_offset.malloc_hd( cotans_squared_flat_offset.size() );
    hd_sparse_mat_offset.copy_from_hd( cotans_squared_flat_offset );

    hd_sparse_mat_diag.malloc_hd( nb_verts );
    hd_sparse_mat_diag.copy_from_hd( diag_vals );
    #endif
#endif
}

// -----------------------------------------------------------------------------

Eigen::Vector3d to_ei(const Vec3& v){
    return Eigen::Vector3d(v.x, v.y, v.z);
}

// -----------------------------------------------------------------------------



#endif // Deactivate cpp

