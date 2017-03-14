#ifndef ANIMATED_MESH__
#define ANIMATED_MESH__

// -----------------------------------------------------------------------------

#include "animesh_enum.hpp"
#include "toolbox/maths/selection_heuristic.hpp"
#include "toolbox/maths/mat2.hpp"
#include "../meshes/gl_mesh.hpp"
#include "../meshes/mesh.hpp"
#include "../rendering/camera.hpp"
#include "../animation/bone.hpp"
// -----------------------------------------------------------------------------

#include <map>
#include <vector>

// Forward def -----------------------------------------------------------------
//#include "skeleton.hpp"
#include "bone.hpp"
struct Skeleton;
// End Forward def -------------------------------------------------------------

/** @brief Class of animated meshes

    As the joints of the skeleton rotates, the mesh is deformed accordingly.

    @warning the class might reorder the mesh vertices when creating the
    Animesh object instance. Methods output and input in the old order
    are suffixed 'aifo' that is to say 'as in imported file order'. Methods not
    marked are suppose to use the new order.
    A mapping between old and new order is stored in 'vmap_old_new' :
    vmap_old_new[old_idx] == new_idx
*/

struct Animesh{
public:

    template<typename _Scalar>
    friend struct iklike_functor;

    // -------------------------------------------------------------------------
    /// @name Inner class
    // -------------------------------------------------------------------------

    /// @brief base class to find HRBF samples on the mesh given a bone
    class HRBF_sampling {
    public:
        HRBF_sampling( Animesh* am ) :
            _bone_id(-1),
            _factor_siblings(false),
            _am(am)
        { }

        virtual ~HRBF_sampling(){ }

        /// Given the mesh and the defined bone _bone_id samples the mesh
        /// surface.
        virtual void sample(std::vector<Tbx::Vec3>& out_verts,
                            std::vector<Tbx::Vec3>& out_normals) = 0;

        /// factor the samples if '_factor_siblings' is true. This returns
        /// The samples factored in the first bone for bones in the same level
        /// of the skeleton tree. The other children don't have samples
        void factor_samples(std::vector<int>& vert_ids,
                            std::vector<Tbx::Vec3>& verts,
                            std::vector<Tbx::Vec3>& normals);

        /// Eliminates samples too far from the bone using parameters "_jmax"
        /// and "_pmax"
        /// @warning reset skeleton before using this
        void clamp_samples(std::vector<int>& vert_ids,
                           std::vector<Tbx::Vec3>& verts,
                           std::vector<Tbx::Vec3>& normals);

        /// Bone to base the heuristic on
        int _bone_id;

        /// consider bones with multiple children has a single bone
        /// Samples will be added to the first child and other bone on the same
        /// level of the skeleton tree will be ignored.
        bool _factor_siblings;

        float _jmax; ///< percentage max dist from joint (range [-1 1])
        float _pmax; ///< percentage max dist from joint parent (range [-1 1])
        float _fold; ///< threshold scalar product dir projection/mesh normal

        Animesh* _am;
    };

    // -------------------------------------------------------------------------

    /// @brief sampling the mesh using an adhoc heuristic
    /// based on the mesh vertices
    class Adhoc_sampling : public HRBF_sampling {
    public:

        Adhoc_sampling( Animesh* am ) :
            HRBF_sampling(am),
            _mind(0.f)
        {}

        void sample(std::vector<Tbx::Vec3>& out_verts,
                    std::vector<Tbx::Vec3>& out_normals);

        float _mind; ///< minimal distance between two samples
    };

    /// @brief sampling the mesh surface with a poisson disk strategy
    class Poisson_disk_sampling : public HRBF_sampling {
    public:

        Poisson_disk_sampling( Animesh* am ) :
            HRBF_sampling(am),
            _min_d(0.f),
            _nb_samples(0)
        {}

        void sample(std::vector<Tbx::Vec3>& out_verts,
                    std::vector<Tbx::Vec3>& out_normals);
        ///< minimal distance between two samples (if == 0 e use _nb_samples)
        float _min_d;
        int _nb_samples;
    };

    // -------------------------------------------------------------------------

    /// @brief some parameters to give when painting the mesh
    /// @see paint()
    struct Paint_setup{
        bool _rest_pose;      ///< Use verts in rest pose
        bool _backface_cull;  ///< Paint only front face
        int _brush_radius;    ///< Brush radius
        int _x, _y;           ///< Brush center
        float _val;           ///< Value to paint (depends on the painting mode)
    };

    // -------------------------------------------------------------------------

    /*-------*/
    /*       */
    /*-------*/

    Animesh(Mesh* m_, Skeleton* s_);

    ~Animesh();

    /// Computes the potential at each vertex of the mesh. When the mesh is
    /// animated, if implicit skinning is enabled, vertices move so as to match
    /// that value of the potential.
    void update_base_potential();

    /// Transform the vertices of the mesh given the rotation at each bone.
    /// Transformation is computed from the initial position of the mesh
    /// @param type specify the technic used to compute vertices deformations
    /// @param refresh : must be true if the skeleton is not moving
    void transform_vertices(EAnimesh::Blending_type type, bool refresh = false);

    /// All vertices are set to their rest position, this does not effect
    /// the skeleton which bones position may have to be reset has well
    void reset_vertices();

    /// Compute one step of deformation with an incremental algorithm
    /// @param refresh : must be true if the skeleton is not moving
    /// it will continue one step time of elastic deformation
    /// @warning incremental means that that the skeleton must be animated
    /// with SMALL steps for the deformation algorithm to work
    void transform_vertices_incr(bool refresh = false);

    /// Another method to deform the mesh incrementaly
    void transform_vertices_incr_energy(bool refresh);

    /// Draw the mesh for the current position of the skeleton
    /// @deprecated
    void draw(bool use_color_array = true, bool use_point_color = false) const;

    /// @deprecated
    void draw_rest_pose(bool use_color_array = true, bool use_point_color = false) const;

    /// @deprecated
    void draw_points_rest_pose() const;

    /// @param rest_pose wether the selection is made for animated pose
    /// or rest pose
    void select(int x, int y, Tbx::Select_type<int>* selection_set, bool rest_pose);

    /// Paint the attribute defined by 'mode'. Painting is done by projecting
    /// mesh's vertices with cuda
    void paint(EAnimesh::Paint_type mode,
               const Paint_setup& setup,
               const Tbx::Camera& cam);

    // -------------------------------------------------------------------------
    /// @name Smooth parts of the limbs
    // -------------------------------------------------------------------------

    /// Compute the vertices/triangles/edges which will be deformed by our
    /// non linear optim problem
    void compute_smooth_parts();

    // -------------------------------------------------------------------------
    /// @name HRBF
    // -------------------------------------------------------------------------

    /// replace the ith bone samples with the samples in 'nodes' and 'n_nodes'
    /// This converts the bone to an Hrbf_bone
    void update_bone_samples(EBone::Id id_bone,
                             const std::vector<Tbx::Vec3>& nodes,
                             const std::vector<Tbx::Vec3>& n_nodes);

    float compute_nearest_vert_to_bone(int bone_id);

    /// Compute caps at the tip of the bone to close the hrbf
    /// @param use_parent_dir: add the cap following the parent direction and
    /// not the direction of 'bone_id'
    void compute_pcaps(int bone_id,
                       bool use_parent_dir,
                       std::vector<Tbx::Vec3>& out_verts,
                       std::vector<Tbx::Vec3>& out_normals);

    /// Compute caps at the tip of the bone to close the hrbf
    void compute_jcaps(int bone_id,
                       std::vector<Tbx::Vec3>& out_verts,
                       std::vector<Tbx::Vec3>& out_normals);

    /// Update the mesh clusters according to the skeleton
    void clusterize(EAnimesh::Cluster_type type);

    // -------------------------------------------------------------------------
    /// @name Import export
    /// @deprecated You should use the parsers module
    // -------------------------------------------------------------------------

    /// Reads the weight file that define how each joint of the skeleton
    /// deforms each vertex of the mesh
    /// @deprecated
    void read_weights_from_file(const char* filename, bool file_has_commas);

    /// Export the ssd weights associated to each vertices
    /// @deprecated
    void export_weights(const char* filename);

    /// Export the mesh for the current position of the skeleton
    /// (i.e mesh is export from the current vbo used to draw it)
    /// @deprecated
    void export_off(const char* filename) const;

    void export_cluster(const char* filename);

    void import_cluster(const char* filename);

    // -------------------------------------------------------------------------
    /// @name SSD Related
    // -------------------------------------------------------------------------

    /// Initialize the interpolation weights that define the proportion of
    /// implicit skinning vs ssd. Each vertex is associated to a weight equals
    /// to 0 (full implicit skinning) when it is inside the implicit primitive
    /// or 1 (full SSD) when it is outside. These weights are then diffused
    /// along the mesh to smoothen transition between SSD animation and IS.
    void init_ssd_interpolation_weights();

    /// The ssd weights are initialize with the bone's cluster. It will result
    /// in a rigid animation of the mesh : each vertex is only influenced by
    /// the nearest bone
    void init_rigid_ssd_weights();

    /// diffuse the ssd weights thanks to the heat difusion equation
    /// @warning Only works with closed 2-manifold triangular meshes
    void heat_diffuse_ssd_weights(float heat);

    /// Some diffusion but not topology based
    /// @warning experimental
    void geodesic_diffuse_ssd_weights(int nb_iter, float strength);

    /// Diffuse the ssd weights along the mesh and normalize them
    void topology_diffuse_ssd_weights(int nb_iter, float strength);

    /// Set the weight of the ith vertex associated to the if joint,
    /// the value is clamped between [0, 1], and the value associated to the
    /// other joints are normalized.
    void set_ssd_weight(int id_vertex, int id_joint, float weight);

    float get_ssd_weight(int id_vertex, int id_joint);

    /// @param weights Array vector of ssd weights per vertices and per bone:
    /// vec[vert_id][...].first == bone_id, vec[vert_id][...].second == weight
    /// use case:
    /** @code
        std::vector<std::map<int, float> > weights;
        get_ssd_weights(weights);
        std::map<int, float>& map = weights[index_vert];
        std::map<int, float>::iterator it;
        for(it = map.begin(); it != map.end(); ++it){
            const int   bone_id = it->first;
            const float weight  = it->second;
        }
        @endcode
    */
    void get_ssd_weights(std::vector<std::map<int, float> >& weights);

    /// copy device ssd weights on host side
    void update_host_ssd_weights();

    void set_ssd_weights(const std::vector<std::map<int, float> >& weights);

    void update_device_ssd_weights();

    // -------------------------------------------------------------------------
    /// @name Getter & Setters
    // -------------------------------------------------------------------------

    /// Compute the potential for a subset of vertices in '_mesh'.
    /// The subset is defined by a list of indices in 'vert_list'.
    /// Potential and gradient (optionnal) are stored in the same order found
    /// in 'vert_list'
    void compute_potential(const Tbx::Vec3 *vert_pos,
                           const int nb_vert,
                           float* d_base_potential,
                           Tbx::Vec3* d_base_grad = 0);

    // -------------------------------------------------------------------------
    /// @name Getter & Setters
    // -------------------------------------------------------------------------

    /// Fill the vector as in the imported file order
    void get_anim_vertices_aifo(std::vector<float>& anim_vert);

    /// Set bone type
    void set_bone_type(int id, int bone_type);

    void get_ssd_lerp(std::vector<float>& ssd_to_is_lerp) const{
        const std::vector<float>& tab = hd_ssd_interpolation_factor;
        ssd_to_is_lerp.resize( tab.size() );
        std::copy(tab.begin(), tab.end() ,ssd_to_is_lerp.begin() );
    }

    void set_ssd_is_lerp(const std::vector<float>& vec) {
        assert( (int)vec.size() == hd_ssd_interpolation_factor.size());
//        hd_ssd_interpolation_factor.copy_from_hd( vec );
		hd_ssd_interpolation_factor =  vec ;
        init_cotan_weights();
    }

    inline void set_smooth_factor(int i, float val){
        d_input_smooth_factors[i] = val;
    }

    void set_smoothing_weights_diffusion_iter(int nb_iter){
        diffuse_smooth_weights_iter = nb_iter;
    }
    void set_smoothing_iter (int nb_iter ) { smoothing_iter = nb_iter;   }
    void set_smooth_mesh    (bool state  ) { do_smooth_mesh = state;     }
    void set_local_smoothing(bool state  ) { do_local_smoothing = state; }
    void set_smooth_force_a (float alpha ) { smooth_force_a = alpha;     }
//    void set_smooth_force_b (float beta  ) { smooth_force_b = beta;      }
//    void set_smooth_smear   (float v     ) { smooth_smear   = v;         }


//    void set_smoothing_type (EAnimesh::Smooth_type type ) {
//        mesh_smoothing = type;
//    }

    void set_enable_update_base_potential(bool s){ do_update_potential = s; }

    /// Switch between implicit skinning and basic ssd skinning
    inline void set_implicit_skinning  ( bool s) { do_implicit_skinning = s;                     }
    inline void switch_implicit_skinning(      ) { do_implicit_skinning = !do_implicit_skinning; }

    inline const std::vector<Tbx::Vec3>& get_ssd_normals() const {
        return d_ssd_normals;
    }

    inline const std::vector<Tbx::Vec3>& get_rot_axis() const {
        return d_rot_axis;
    }

    inline const std::vector<Tbx::Vec3>& get_gradient() const {
        return hd_gradient;
    }

    inline int get_nearest_bone(int vert_idx){
        return h_vertices_nearest_bones[vert_idx];
    }

    float get_junction_radius(int bone_id);

    // TO be deleted and moved in the ctrl
    void set_junction_radius(int bone_id, float rad);

    void set_flip_propagation(int vid, bool s){
        d_flip_propagation[vid] =s;
    }

    void reset_flip_propagation();

    inline EAnimesh::Color_type get_color_type(){ return mesh_color; }

    /// Compute colors for the animated mesh according to the color type
    /// @warning for SSD_WEIGHTS use set_color_ssd_weight() method
    void set_colors(EAnimesh::Color_type type = EAnimesh::BASE_POTENTIAL,
                    float r = 0.7f,
                    float g = 0.7f,
                    float b = 0.7f,
                    float a = 1.f );

    ///< color the mesh according to its ssd weights associated to the ith joint
    void set_color_ssd_weight(int joint_id);

    Tbx::GlBuffer_obj<Tbx::Vec3>* get_vbo_rest_pose(){
        return _vbo_input_vert;
    }

    Tbx::GlBuffer_obj<Tbx::Vec3>* get_nbo_rest_pose(){
        return _nbo_input_normal;
    }

    const Mesh*     get_mesh() const { return _mesh; }
    const Skeleton* get_skel() const { return _skel; }

    Skeleton* get_skel(){ return _skel; }

private:
    // -------------------------------------------------------------------------
    /// @name Tools
    // -------------------------------------------------------------------------

    void compute_SVD_rotations(Tbx::Vec3 *verts);

    /// Compute the nearest cluster of bones for each vertex. We use euclidean
    /// distance to determine the nearest bone
    void clusterize_euclidean(std::vector<int>& h_nearest_bones,
                              std::vector<int>& h_nearest_joint,
                              std::vector<int>& nb_vert_by_bone);

    /// Compute the nearest cluster of bones for each vertex. We use current
    /// SSD weights. Each vertex is associated to the highest bone weight
    void clusterize_weights(std::vector<int>& h_nearest_bones,
                            std::vector<int>& h_nearest_joint,
                            std::vector<int>& nb_vert_by_bone);

    /// Update the attributes 'd_nearest_bone_in_device_mem' and
    /// 'd_nearest_joint_in_device_mem'
    void update_nearest_bone_joint_in_device_mem();

    /// Compute a radius for each bone, given the distance of each vertex to
    /// their closest bone
    void set_default_bones_radius();

    /// make the mesh smooth with the smoothing technique specified by
    /// mesh_smoothing
    void smooth_mesh(Tbx::Vec3* output_vertices,
                     Tbx::Vec3* verts_buffer,
                     float* factors,
                     int nb_iter,
                     bool local_smoothing = true);


    /// Treats edges like springs, 'h_verts' is modified in place.
    /// 'nb_iter' defines the stifness of the springs. Computation is CPU
    void spring_relaxation(Tbx::Vec3* h_verts, int nb_iter);

    /// Compute normals in 'normals' and the vertices position in 'vertices'
    void compute_normals(const std::vector<Tbx::Vec3>& vertices, std::vector<Tbx::Vec3>& normals);

    void compute_tangents(const std::vector<Tbx::Vec3>& vertices, std::vector<Tbx::Vec3>& tangents);

    /// Compute geometric deformation of verts 'd_in' (with SSD, dual quat ...)
    /// @param t : type of the deformation
    /// @param d_in : input vertices to deform
    /// @param out : output vertices deformed with 't' method
    /// @param out2: identical as 'out'
    /// @param transfos : tranformations at each joints (indexed by EBone::Id)
    /// Type is interpreted as Dual_quat_cu for dual quats deformations
    /// or Transfo for other geometric deformations. If not specified
    /// the current skeleton transformations are used
    void geometric_deformation(EAnimesh::Blending_type t,
                               const std::vector<Tbx::Vec3> &d_in,
                               std::vector<Tbx::Vec3>& out,
                               std::vector<Tbx::Vec3>& out2,
                               const void* transfos = 0);

    void compute_blended_dual_quat_rots();

    /// Interpolates between out_verts and ssd_position (in place)
    void ssd_lerp(Tbx::Vec3* out_verts);

    void fit_mesh(int nb_vert_to_fit,
                  int* d_vert_to_fit,
                  bool full_eval,
                  bool smooth_fac_from_iso,
                  Tbx::Vec3 *d_vertices,
                  int nb_steps, float smooth_strength);

    void fit_mesh_std(int nb_vert_to_fit,
                      int* d_vert_to_fit,
                      bool full_eval,
                      bool smooth_fac_from_iso,
                      Tbx::Vec3* d_vertices,
                      int nb_steps,
                      float smooth_strength);

    /// diffuse values over the mesh on GPU
    void diffuse_attr(int nb_iter, float strength, float* attr);

    int pack_vert_to_fit(std::vector<int>& in,
                         std::vector<int>& out,
                         int last_nb_vert_to_fit);

    /// Pack negative index in 'd_vert_to_fit' (done on gpu)
    /// @param d_vert_to_fit list of vertices to fit negative indices are to be
    /// eliminated and indices regrouped to the begining of this array
    /// @param buff_prefix_sum intermediate array filled with the prefix sum of
    /// 'd_vert_to_fit' where negative indices are considered as zeros positives
    /// as ones. buff_prefix_sum.size() == 1+d_vert_to_fit.size()
    /// @param packed_vert_to_fit result of the packing of 'd_vert_to_fit'.
    /// packed_vert_to_fit contains the positive indices regrouped at its
    /// beginning. packed_vert_to_fit.size() == d_vert_to_fit.size()
    /// @return the number of vertices to fit (i.e number of index >= 0)
    /// its the size of packed_vert_to_fit which holds the positive indices of
    /// d_vert_to_fit
    int pack_vert_to_fit_gpu(
            std::vector<int>& d_vert_to_fit,
            std::vector<int>& buff_prefix_sum,
            std::vector<int>& packed_vert_to_fit,
            int nb_vert_to_fit);

    /// Updating vbo nbo tbo with d_vert, d_normals and d_tangents.
    /// vertices with multiple texture coordinates are duplicated
    /// @param nb_vert size of the arrays. It's not necesarily the same as the
    /// buffer objects
    /// @param d_tangents array of tangents of size nb_vert if equal 0 this
    /// parameter is ignored
    /// @param tbo buffer object of tangents if equal 0 this parameter is
    /// ignored
    /// @warning buffer objects are to be registered in cuda context
    void update_opengl_buffers(int nb_vert,
                               const std::vector<Tbx::Vec3>& d_vert,
                               const std::vector<Tbx::Vec3>& d_normals,
                               const std::vector<Tbx::Vec3>& d_tangents,
                               Tbx::GlBuffer_obj<Tbx::Vec3>* vbo,
                               Tbx::GlBuffer_obj<Tbx::Vec3>* nbo,
                               Tbx::GlBuffer_obj<Tbx::Vec3>* tbo);

    /// Copy the attributes of 'a_mesh' into the attributes of the animated
    /// mesh in device memory
    void copy_mesh_data(const Mesh& a_mesh);

    /// Compute the mean value coordinates (mvc) of every vertices in rest pose
    /// @note : in some special cases the sum of mvc will be exactly equal to
    /// zero. This will have to be dealt with properly  when using them. For
    /// instance when smoothing we will have to check that.
    /// Cases :
    /// - Vertex is a side of the mesh
    /// - one of the mvc coordinate is negative.
    /// (meaning the vertices is outside the polygon the mvc is expressed from)
    /// - Normal of the vertices has norm == zero
    void compute_mvc();

    /// Allocate and initialize 'd_vert_to_fit' and 'd_vert_to_fit_base'.
    /// For instance lonely vertices are not fitted with the implicit skinning.
    void init_vert_to_fit();

    /// Initialize attributes h_input_verts_per_bone and
    /// h_input_normals_per_bone which stores vertices and normals by nearest
    /// bones
    void init_verts_per_bone();

    void init_vertex_bone_proj();

    void init_smooth_factors(std::vector<float>& d_smooth_factors);

    void init_sum_angles();

    void init_cotan_weights();

    /// @return true if mesh color enum belongs to the subset of colors that
    /// needs to be updated dynamically
    bool is_dynamic_color( EAnimesh::Color_type mesh_col );

    // -------------------------------------------------------------------------
    /// @name Attributes
    // -------------------------------------------------------------------------
public:
    /// The mesh 'm' is deformed after each call of transform_vertices().
    /// deformation is computed from the initial position of the mesh stored in
    /// d_input_vertices. The mesh buffer objects attributes defines the animated
    /// mesh
    Mesh*      _mesh;
    Skeleton*  _skel;

    /// Does the bone intended to deform the mesh or only to be used for
    /// kinematic purpose ? (meaning the bone weights/clusters will be ignored)
    std::vector<bool> _do_bone_deform;

    EAnimesh::Color_type  mesh_color;
//    EAnimesh::Smooth_type mesh_smoothing;

    bool do_implicit_skinning;
    bool do_smooth_mesh;
    bool do_local_smoothing;
    bool do_interleave_fitting;
    bool do_update_potential;

    /// Smoothing strength after animation

    int smoothing_iter;
    int diffuse_smooth_weights_iter;
    float smooth_force_a; ///< must be between [0 1]
//    float smooth_force_b; ///< must be between [0 1] only for humphrey smoothing
//    float smooth_smear;   ///< between [-1 1]

    /// Smoothing weights associated to each vertex
    std::vector<float> d_input_smooth_factors;
    /// Animated smoothing weights associated to each vertex
    std::vector<float> d_smooth_factors_conservative;

    /// Smooth factor at each vertex depending on SSD
    std::vector<float> d_smooth_factors_laplacian;

    /// Initial vertices in their "resting" position. animation is compute with
    /// these points
    std::vector<Tbx::Vec3> d_input_vertices;

    /// VBO of the Mesh in resting position
    Tbx::GlBuffer_obj<Tbx::Vec3>* _vbo_input_vert;
    /// NBO of the mesh in resting position
    Tbx::GlBuffer_obj<Tbx::Vec3>* _nbo_input_normal;

    std::vector<float> h_junction_radius;

    /// triangle index in device mem. We don't use the mesh's vbos because
    /// there are different from the mesh's real topology as some of the vertices
    /// are duplicated for rendering because of textures
    std::vector<int> d_input_tri;
    /// quad index in device mem. @see  d_input_tri
    std::vector<int> d_input_quad;

    // -------------------------------------------------------------------------
    /// @name Per vertex datas
    // -------------------------------------------------------------------------

public:
    /// Rotation for the 2D arap
    std::vector<Tbx::Mat2> hd_verts_rots;

    /// Rotation for the 3D arap
    std::vector<Tbx::Mat3> hd_verts_3drots;

    /// Vertex projected on their nearest bone
    std::vector<Tbx::Vec3> hd_verts_bone_proj;

    /// Stores in which state a vertex is when fitted into the implicit surface
    std::vector<EAnimesh::Vert_state> d_vertices_state;
    /// Colors associated to the enum field EAnimesh::Vert_state
    std::vector<Tbx::Vec4> d_vertices_states_color;


    std::vector<Tbx::Vec3>  hd_prev_output_vertices;
    /// Animated vertices in their final position.
    std::vector<Tbx::Vec3>  hd_output_vertices;

    std::vector<Tbx::Vec3>  hd_tmp_vertices;
    /// final normals
    std::vector<Tbx::Vec3> d_output_normals;
    std::vector<Tbx::Vec3> d_output_tangents;

    /// Points of the mesh animated by ssd
    std::vector<Tbx::Vec3>  d_ssd_vertices;
    /// Normals of the mesh animated by ssd
    std::vector<Tbx::Vec3> d_ssd_normals;

public:
    /// Gradient of the implicit surface at each vertices when animated
    std::vector<Tbx::Vec3> hd_gradient;

    /// Velocity of vertices at each vertex for one animation step
    std::vector<Tbx::Vec3> hd_velocity;

    /// Base potential associated to the ith vertex (i.e in rest pose of skel)
    std::vector<float> d_base_potential;

    /// Base gradient associated to the ith vertex (i.e in rest pose of skel)
    std::vector<Tbx::Vec3> d_base_gradient;

    /// Buffer used to compute normals on GPU. this array holds normals for each
    /// face. d_unpacked_normals[vert_id*nb_max_face_per_vert + ith_face_of_vert]
    /// == normal_at_vert_id_for_its_ith_face
    std::vector<Tbx::Vec3> d_unpacked_normals;
    /// same as 'd_unpacked_normals' but with tangents
    std::vector<Tbx::Vec3> d_unpacked_tangents;
    /// ?
    std::vector<EMesh::Prim_idx_vertices> d_piv;

    /// Map the ith packed vertex to the unpacked array form used for rendering
    /// d_packed_vert_map[packed_vert] = unpacked_vert
    /// @see Mesh
    std::vector<EMesh::Packed_data> d_packed_vert_map;

    /// Vector representing the rotation axis of the nearest joint for each
    /// vertex. d_rot_axis[vert_id] == vec_rotation_axis
    std::vector<Tbx::Vec3>  d_rot_axis;

    /// Vertices are moved based on a linear interpolation between ssd and
    /// implicit skinning this array map for each vertex index its interpolation
    /// weight : 1 is full ssd and 0 full implicit skinning
    std::vector<float> hd_ssd_interpolation_factor;

    // -------------------------------------------------------------------------
    /// @name Edge datas
    // -------------------------------------------------------------------------
public:

    /// List of mesh edges of the mesh
    std::vector<EMesh::Edge> hd_edge_list;

    /// Associated mesh bending at every edges(for the rest pose).
    /// Bending is measured by the dihedral angle between two triangles shared
    /// at the edge. At mesh boundary bending is null as well as edge with
    /// non manifold features (i.e. More than 2 triangles shared at one edge)
    std::vector<float> hd_edge_bending;

    /// Sum at every vertices of their incident edge angles (at rest pose)
    std::vector<float> hd_sum_angles;

    // -------------------------------------------------------------------------
    /// @name Deformed data
    // -------------------------------------------------------------------------
    /// subList of all edges index to be deformed
    std::vector<int> hd_free_edges;

    /// subList of all triangle index to be deformed
    std::vector<int> hd_free_triangles;

    /// subList of vertices index to be deformed
    std::vector<int> hd_free_vertices;

    /// Map index of every vertices to free (subset of deformed) vertices.
    /// if not in subset == -1
    std::vector<int> hd_map_verts_to_free_vertices;

    /// subList of vertices potential at rest pose
    std::vector<float> hd_free_base_potential;
    // -------------------------------------------------------------------------
    /// @name Rest pose datas
    // -------------------------------------------------------------------------

    /// Lengths per edges at rest pose
    std::vector<float> hd_edge_lengths;

    /// Per triangle areas in rest pose
    std::vector<float> hd_tri_areas;

public:
    // -------------------------------------------------------------------------
    /// @name 1st Ring data
    // -------------------------------------------------------------------------
    /// Store for the angle between each ring vertex
    /// @note to look up this list you need to use 'd_1st_ring_list_offsets'
    std::vector<float> hd_1st_ring_angle;

    /// Store for each 1st ring vertex the length between the central vertex
    /// (at rest pose)
    /// @note to look up this list you need to use 'd_1st_ring_list_offsets'
    std::vector<float> d_1st_ring_lengths;

    /// Store for each 1st ring vertex the laplacian cotan weight
    /// @note to look up this list you need to use 'd_1st_ring_list_offsets'
    std::vector<float> hd_1st_ring_cotan;

    /// Sparse matrix for every vertex of the squared laplacian
    std::vector< std::pair<float, int> > hd_sparse_mat_vals;

    std::vector<float> hd_sparse_mat_diag;

    std::vector<int> hd_sparse_mat_offset;

    std::vector<Tbx::Vec3> hd_vec_B;

    /// Store for each 1st ring vertex the mean value coordinate of the central
    /// vertex (Mean value coordinates are barycentric
    /// coordinates where we can express v as the sum of its neighborhood
    /// neigh_i. v = sum from 0 to nb_neigh { mvc_i * neigh_i } )
    /// @note to look up this list you need to use 'd_1st_ring_list_offsets'
    std::vector<float> hd_1st_ring_mvc;

    /// List of first ring edges indices
    std::vector<int> hd_1st_ring_edges;

    /// List of first ring neighborhoods for a vertices, this list has to be
    /// read with the help of d_1st_ring_list_offsets[] array @see d_1st_ring_list_offsets
    std::vector<int> d_1st_ring_list;
    /// Table of indirection in order to read d_1st_ring_list[] array.
    /// For the ith vertex d_1st_ring_list_offsets[2*ith] gives the offset from
    /// which d_1st_ring_list as to be read and d_1st_ring_list_offsets[2*ith+1] gives
    /// the number of neighborhood for the ith vertex.
    std::vector<int> d_1st_ring_list_offsets;

    // -------------------------------------------------------------------------
    /// @name 2nd Ring data
    // -------------------------------------------------------------------------

    /// Store for each 2nd ring vertex the length between the central vertex
    /// (at rest pose)
    /// @note to look up this list you need to use 'd_2nd_ring_list_offsets'
    std::vector<float> d_2nd_ring_lengths;

    /// List of second ring neighborhoods of vertices, this list has to be
    /// read with the help of d_2nd_ring_list_offsets[] array @see d_2nd_ring_list_offsets
    std::vector<int> d_2nd_ring_list;
    /// Table of indirection in order to read d_2nd_ring_list[] array.
    /// For the ith vertex d_2nd_ring_list_offsets[2*ith] gives the offset from
    /// which d_2nd_ring_list as to be read and d_2nd_ring_list_offsets[2*ith+1] gives
    /// the number of neighborhood for the ith vertex.
    std::vector<int> d_2nd_ring_list_offsets;

    // -------------------------------------------------------------------------
    /// @name SSD Weights
    // -------------------------------------------------------------------------

    /// List of joint numbers associated to a vertex. You must use d_jpv to
    /// access the list of joints associated to the ith vertex
    /// @see d_jpv
    std::vector<int> d_joints;

    /// List of ssd weights associated to a vertex for animation. You must use
    /// d_jpv to access the list of ssd weights associated to the ith vertex
    /// @see d_jpv
    std::vector<float> d_weights;

    /// Table of indirection which associates for the ith vertex its list of
    /// weights and joint IDs.
    /// For the ith vertex d_jpv gives for (ith*2) the starting index in
    /// d_joints and d_weights. The (ith*2+1) element gives the number of joint/
    /// weigths associated to the vertex in d_joints and d_weights arrays.
    std::vector<int> d_jpv;

    /// SSD weigths on CPU
    /// h_weights[ith_vert] = pair(first:joint, second:weight)
    std::vector<std::map<int, float> > h_weights;

    // -------------------------------------------------------------------------
    /// @name CLUSTER
    // -------------------------------------------------------------------------

 //   typedef Skeleton_env::DBone_id DBone_id;

    /// Mapping of mesh points with there nearest bone
    /// (i.e tab[vert_idx]=bone_idx)
    std::vector<EBone::Id>  h_vertices_nearest_bones;
    std::vector<EBone::Id>  d_vertices_nearest_bones;
    //std::vector<DBone_id>  d_nearest_bone_in_device_mem;
    //std::vector<DBone_id>  d_nearest_joint_in_device_mem;
    std::vector<float> hd_nearest_joint_dist;

    /// Initial vertices in their "resting" position. sorted by nearest bone.
    /// h_input_vertices[nearest][ith_vert] = vert_coord
    std::vector< std::vector<Tbx::Vec3> > h_input_verts_per_bone;
    std::vector< std::vector<Tbx::Vec3> > h_input_normals_per_bone;

    /// h_input_vertices[nearest][ith_vert] = vert_id_in_mesh
    std::vector< std::vector<int> > h_verts_id_per_bone;

    /// Distance from the nearest bone to the mesh's vertices
    /// (distance is either euclidean or geodesic depending on the
    /// clusterisation)
    /// h_bone_dist[ver_id][bone_id] = dist bone to vert
    std::vector< double > h_nearest_bone_dist;

    /// nb_vertices_by_bones[bone_idx] = nb_vertices
    std::vector<int>    nb_vertices_by_bones;

    /// Mapping of mesh points with there nearest joint
    /// (i.e tab[vert_idx] = joint_idx)
    std::vector<int>   h_vertices_nearest_joint;
    std::vector<int> d_vertices_nearest_joint;

    // END CLUSTER -------------------------------------------------------------

    /// Map old vertex index from Mesh class to the new Vertex index in
    /// Animesh. Because Animesh changes the order of vertex indices
    /// we need to keep track of that change. It is usefull to load from a file
    /// the ssd weight which are in the old order for instance.
    /// vmap_old_new[old_order_idx] == new_order_idx
    std::vector<int> vmap_old_new;

    /// Map new vertex index from Animesh class to the old Vertex index in
    /// Mesh. Because Animesh changes the order of vertex indices
    /// we need to keep track of that change. It is usefull to export a file
    /// of ssd weights which usually must be stored as imported
    /// (in the old order).
    /// vmap_new_old[new_order_idx] == old_order_idx
    std::vector<int> vmap_new_old;

    /// Vertices behind a joint when it flex
    std::vector<bool> d_rear_verts;

    /// (joint idx in host mem)
    std::vector<Tbx::Vec3> h_half_angles;
    std::vector<Tbx::Vec3> d_half_angles;

    /// orthogonal vector at each joint considering the adjacents bones
    /// (joint idx in host mem)
    std::vector<Tbx::Vec3> h_orthos;
    std::vector<Tbx::Vec3> d_orthos;

    /// Array used to debug. Tells if a vertices needs to be fitted by following,
    /// the inverse direction it should take.
    /// d_flipp_propagation[vert_id] = if we flip the propagation dir of the fit
    std::vector<bool> d_flip_propagation;
public:
    // -------------------------------------------------------------------------
    /// @name Pre allocated arrays to store intermediate results of the mesh
    // -------------------------------------------------------------------------
    /// @{
    std::vector<Tbx::Vec3>  d_grad_transfo;
    std::vector<Tbx::Vec3>    h_vert_buffer;
    std::vector<Tbx::Vec3>  d_vert_buffer;
    std::vector<Tbx::Vec3>  d_vert_buffer_2;
    std::vector<float>    d_vals_buffer;

    std::vector<int>      d_vert_to_fit;
    std::vector<int>      d_vert_to_fit_base;
    std::vector<int>      d_vert_to_fit_buff_scan;
    std::vector<int>      d_vert_to_fit_buff;

    std::vector<int>        h_vert_to_fit_buff;
    std::vector<int>        h_vert_to_fit_buff_2;
    /// @}

    // -------------------------------------------------------------------------
    /// @name Buffer energy terms
    // -------------------------------------------------------------------------


    std::vector<float> hd_energy_terms;
    std::vector<float> hd_energy_grad;
    std::vector<float> hd_energy_verts;

    /// vertex displacement between two relaxation steps
    std::vector<float> hd_displacement;
};
// END ANIMATEDMESH CLASS ======================================================

#endif // ANIMATED_MESH__
