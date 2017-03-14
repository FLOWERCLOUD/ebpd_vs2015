#ifndef MESH_HPP__
#define MESH_HPP__

#include "mesh_unpacked_attr.hpp"
#include "mesh_buffer_objects.hpp"
#include "mesh_static.hpp"
#include "mesh_half_edge.hpp"
#include "mesh_types.hpp"

class Mesh {
public:
	/// @warning this don't copy texture but only pointer to texture
	Mesh(const Mesh& m);
	Mesh();
	/// Build mesh from triangle index and vertex position
	Mesh(const std::vector<EMesh::Vert_idx>& tri, const std::vector<float>& vert);
	~Mesh();

	/// Free memory for every attributes
	void clear_data();

	/// Check for data corruptions in the mesh  and exit programm if there is any.
	/// This is a debug function, if it triggers it is likely this class is bugged.
	void check_integrity();

	//  ------------------------------------------------------------------------
	/// @name Import/export
	//  ------------------------------------------------------------------------

	/// Create a mesh from an OFF file
	Mesh(const char* filename);

	/// @param filename      path and name for the file to be writen
	/// @param invert_index  wether the quads and triangles index are to be
	/// inverted (i.e write in clockwise order or counter clockwise)
	void export_off(const char* filename, bool invert_index = false) const{
		_mesh_static.export_off(filename, invert_index, _scale, _offset);
	}

	//  ------------------------------------------------------------------------
	/// @name Drawing the mesh
	//  ------------------------------------------------------------------------

	// Highly inneficient used for debug purpose
	void debug_draw_edges() const;

	/// Draw the mesh using the given vertex and normal buffer objects
	void draw_using_buffer_object(const Tbx::GlBuffer_obj<Tbx::Vec3>& vbo,
		const Tbx::GlBuffer_obj<Tbx::Vec3>& n_bo,
		bool use_color_array = true) const;

	/// Draw point mesh using the given vertex and normal buffer objects
	void draw_points_using_buffer_object(const Tbx::GlBuffer_obj<Tbx::Vec3>& vbo,
		const Tbx::GlBuffer_obj<Tbx::Vec3>& n_bo,
		const Tbx::GlBuffer_obj<Tbx::Vec4>& c_bo,
		bool use_color_array = true) const;

	/// draw mesh points using color and buffer objects
	void draw_points() const;

	/// Draw the mesh using the given vertex, normal and color buffer objects
	void draw_using_buffer_object(const Tbx::GlBuffer_obj<Tbx::Vec3>& vbo,
		const Tbx::GlBuffer_obj<Tbx::Vec3>& n_bo,
		const Tbx::GlBuffer_obj<Tbx::Vec4>& c_bo,
		bool use_color_array = true) const;

	/// Draw the mesh
	/// @param use_color_array use the color array to draw the mesh
	/// @param use_point_color There are two buffers objects that stores the
	/// color. One is the points color and the other the mesh color. this
	/// parameter choose between them
	void draw(bool use_color_array = true, bool use_point_color = false) const;

	/// Enable openGL client state for vertices/texture UVs/normals
	void enable_client_state() const;
	/// Disable openGL client state for vertices/texture UVs/normals
	void disable_client_state() const;

	//  ------------------------------------------------------------------------
	/// @name Surface deformations
	//  ------------------------------------------------------------------------

	/// Center about the center of gravity and resize the mesh to the given
	/// length 'size'
	// TODO: should return the mesh offset and scale computed and let the user
	// decided if he wants to apply it directly to the mesh with set_scale()
	// or while display with matrices
	void center_and_resize(float size);

	/// Smoothen the mesh (make the surface less rough)
	void smoothen_mesh(float smooth_factor,
		int nb_iter,
		int nb_min_neighbours = 0);

	/// Diffuse some vertex attributes along the mesh.
	/// Each vertex is associated to an attribute defined by the array
	/// "vertices_attributes[]". For each vertex the mean sum of its first ring
	/// of neighborhood is computed, the operation is repeated "nb_iter" times.
	/// @return The new vertices attributes in "vertices_attributes[]" array
	void diffuse_along_mesh(float* vertices_attributes, int nb_iter) const;

	/// Diffuse some vertex attributes along the mesh. see method above.
	/// this method is the same but does not change attributes values which
	/// equals to "locked_value"
	void diffuse_along_mesh(float* vertices_attributes,
		float locked_value,
		int nb_iter) const;

	/// Add noise to the mesh
	void add_noise(int fq, float amp);

	//  ------------------------------------------------------------------------
	/// @name Colors accessors
	//  ------------------------------------------------------------------------

	/// Set the color of the ith vertices when the mesh points are displayed
	/// @warning slow method for large data don't use this use directly th  e
	/// buffer object
	void set_point_color_bo(int i, float r, float g, float b, float a);

	/// Set the color_bo attribute to the given rgba color.
	/// @see color_bo
	void set_color_bo(float r, float g, float b, float a);

	/// Set the point_color_bo attribute to the given rgba color.
	/// @see point_color_bo
	void set_point_color_bo(float r, float g, float b, float a);

	//  ------------------------------------------------------------------------
	/// @name Update attributes
	//  ------------------------------------------------------------------------

	void compute_normals();

	void compute_tangents();

	//  ------------------------------------------------------------------------
	/// @name
	//  ------------------------------------------------------------------------

	void resize_tris(int nb_tris)
	{
		_mesh_static.resize_tris( nb_tris );
		_mesh_attr.resize_tris( nb_tris );
		_mesh_gl.resize_tris( nb_tris );
	}


	//  ------------------------------------------------------------------------
	/// @name
	//  ------------------------------------------------------------------------

	// TODO: to be deleted offset and scale
	/// Get the offset
	inline const Tbx::Vec3& get_offset() const{ return _offset; }

	/// Get the scale
	inline float get_scale() const { return _scale; }

	//  ------------------------------------------------------------------------
	/// @name Vertices and associated datas accessors
	//  ------------------------------------------------------------------------

	int get_nb_vertices() const { return _mesh_static.get_nb_vertices(); }

	/// Get ith vertex position
	Tbx::Vec3 get_vertex(EMesh::Vert_idx i) const { return _mesh_static.get_vertex(i); }

	/// Clear and fill 'verts' vector with vertex positions
	void get_vertices( std::vector<Tbx::Vec3>& verts ) const;

	/// List of vertex indices which are connected to not manifold triangles
	const std::vector<EMesh::Vert_idx>& get_not_manifold_list() const { return _mesh_he.get_not_manifold_list(); }

	/// List of vertices on a side of the mesh
	const std::vector<EMesh::Vert_idx>& get_on_side_list() const { return _mesh_he.get_on_side_list(); }

	/// Is the ith vertex on the mesh boundary
	bool is_vert_on_side(EMesh::Vert_idx i) const { return _mesh_he.is_vert_on_side(i); }

	/// @return false if the ith vertex belongs to at least one primitive
	/// i.e triangle or quad
	bool is_disconnect(EMesh::Vert_idx i) const { return _mesh_he.is_disconnect(i); }

	/// Get map (packed vertices) -> (unpacked vertices). Telling how much
	/// each vertex is duplicated (e.g, when there has multiple tex coords)
	/// It's used to fill the VBO correctly from 'get_vertex()'
	const EMesh::Packed_data* get_packed_vert_map() const { return &(_mesh_attr._packed_vert_map.front()); }

	/// Get the normal at vertex no. i
	/// @param n sometimes a vertex has multiples normals
	/// (either duplicated because of texture coordinates or different for the shading)
	/// the parameter is a way to fetch at the ith vertex the nth normal.
	/// @see get_nb_normals()
	Tbx::Vec3 get_normal(EMesh::Vert_idx i, int n = 0) const;

	/// number of normals associated to the ith vertex
	int get_nb_normals(EMesh::Vert_idx i) const { return _mesh_attr._packed_vert_map[i]._nb_ocurrence; }

	/// Get the normal at vertex no. i
	/// @return the mean normal at vertex 'i'
	/// (sometimes a vertex has multiples normals)
	/// @see get_normal()
	Tbx::Vec3 get_mean_normal(EMesh::Vert_idx i) const;

	/// Get the list of normals (average them if multiple normals at one vertex)
	/// @see get_normal()
	void get_mean_normals( std::vector<Tbx::Vec3>& normals ) const;

	/// Get the texture coordinates at vertex no. i
	/// @param n sometimes a vertex has multiples texture coordinates
	/// the parameter 'n' is a way to fetch at the ith vertex the
	/// nth texture coordinate.
	/// @see get_nb_tex_coords()
	EMesh::Tex_coords get_tex_coords(EMesh::Vert_idx i, int n = 0) const;

	/// number of texture coordinates associated to the ith vertex
	int get_nb_tex_coords(EMesh::Vert_idx i) const { return _mesh_attr._packed_vert_map[i]._nb_ocurrence; }

	/// Total number of unpacked vertices
	int get_vbos_size() const { return _mesh_attr._size_unpacked_verts; }

	//  ------------------------------------------------------------------------
	/// @name Faces accessors
	//  ------------------------------------------------------------------------

	int get_nb_tris()   const { return _mesh_static.get_nb_tris();   }
	int get_nb_quads()  const { return _mesh_static.get_nb_quads();  }
	int get_nb_faces() const { return _mesh_static.get_nb_faces(); }

	/// Get the offsets for primitive no. i
	EMesh::Prim_idx_vertices get_piv(int i) const { return _mesh_he.get_piv(i); }

	/// Get a triangle described by a list of three edge indices.
	/// You can then retreive the edge with #get_edge()
	EMesh::Tri_edges get_tri_edges(int tri_index) const { return _mesh_he.get_tri_edges( tri_index ); }

	/// Get three triangle index for the ith triangle
	EMesh::Tri_face get_tri(EMesh::Tri_idx i) const { return _mesh_static.get_tri(i); }

	/// Get four quad index for the ith quad
	EMesh::Quad_face get_quad(EMesh::Quad_idx i) const { return _mesh_static.get_quad(i); }

	//  ------------------------------------------------------------------------
	/// @name Materials accessors
	//  ------------------------------------------------------------------------

	const std::vector<EMesh::Mat_grp>& get_mat_grps_tri() const { return _mesh_attr._material_grps_tri; }

	const std::vector<EMesh::Material>& get_mat_list() const { return _mesh_attr._material_list; }

	//  ------------------------------------------------------------------------
	/// @name 1st ring neighborhood accessors
	//  ------------------------------------------------------------------------

	const std::vector<EMesh::Vert_idx>& get_1st_ring_verts(EMesh::Vert_idx i) const {
		return _mesh_he.get_1st_ring_verts(i);
	}

	/// clear and fill 'first_ring' with the ordered first neighborhood ring of
	/// the ith vertex
	void get_1st_ring( std::vector< std::vector<EMesh::Vert_idx> >& first_ring ) const { _mesh_he.get_1st_ring(first_ring); }

	/// list_size == #get_size_1st_ring_list()
	/// Contigus list of 1st ring neighborhood of every vertices
	EMesh::Vert_idx get_1st_ring(int i) const { return _mesh_he.get_1st_ring(i); }

	int get_size_1st_ring_list() const { return _mesh_he.get_size_1st_ring_list(); }

	/// Size == 2 * get_nb_vertices()
	int get_1st_ring_offset(EMesh::Vert_idx i) const { return _mesh_he.get_1st_ring_offset(i); }

	/// Valence of the ith vertex
	int get_valence(EMesh::Vert_idx i) const { return _mesh_he.get_valence(i); }

	/// Number of 1st ring neighborhood at the ith vertex "same as valence"
	int get_nb_neighbors(EMesh::Vert_idx i) const { return _mesh_he.get_nb_neighbors(i); }

	/// Get edge indices at the first ring neighborhood of the ith vertex.
	/// @note same order as with first ring of vertices (#get_1st_ring())
	const std::vector<int>& get_1st_ring_edges(EMesh::Vert_idx i) const { return _mesh_he.get_1st_ring_edges(i); }

	/// Get triangle indices at the first ring neighborhood of the ith vertex.
	/// @warning !! this list is unordered unlike the other 1st ring list !!
	// FIXME: add a post-process that re-order this list c.f compute_face_index()
	const std::vector<EMesh::Tri_idx>& get_1st_ring_tris(EMesh::Vert_idx i) const { return _mesh_he.get_1st_ring_tris(i); }

	//  ------------------------------------------------------------------------
	/// @name 2nd ring neighborhood accessors
	//  ------------------------------------------------------------------------

	const std::vector<EMesh::Vert_idx>& get_2nd_ring_verts(EMesh::Vert_idx i) { return _mesh_he.get_2nd_ring_verts(i); }

	//  ------------------------------------------------------------------------
	/// @name Edges accessors
	//  ------------------------------------------------------------------------

	EMesh::Edge get_edge(EMesh::Edge_idx i) const { return _mesh_he.get_edge(i); }

	int get_nb_edges() const { return (int)_mesh_he.get_nb_edges(); }

	/// List of triangles shared by the edge 'i'. It can be one for boundaries
	/// two for closed objects or another number if the mesh is not 2-manifold
	const std::vector<EMesh::Tri_idx>& get_edge_shared_tris(EMesh::Edge_idx i) const { return _mesh_he.get_edge_shared_tris(i); }

	bool is_side_edge(EMesh::Edge_idx i) const { return _mesh_he.is_side_edge(i); }

	//  ------------------------------------------------------------------------
	/// @name Various accessors
	//  ------------------------------------------------------------------------

	bool is_closed()      const { return _mesh_he.is_closed();   }
	bool is_manifold()    const { return _mesh_he.is_manifold(); }
	bool has_tex_coords() const { return _has_tex_coords;        }
	bool has_normals()    const { return _has_normals;           }
	bool has_materials()  const { return _has_materials;         }
	bool has_bumpmap()    const { return _has_bumpmap;           }

	/// updates the vbo with the vertex position in '_vert'.
	/// @warning slow method it is done on CPU
	//void update_vertex_buffer_object();

private:



	// TODO: everything must be private below here


	//  ------------------------------------------------------------------------
	/// @name global properties
	//  ------------------------------------------------------------------------
public:
	bool _is_initialized;    ///< is the mesh renderable (means every attributes is filled correctly)
	bool _has_tex_coords;    ///< do the mesh has texture coordinates loaded
	bool _has_normals;       ///< do the mesh has normals loaded
	bool _has_materials;     ///< do we apply the matterial list
	bool _has_bumpmap;

	// TODO: to be deleted offset and scale. We should use transformation matrices to do that while display.
	// Or if the user want to revert mesh changes its his responsibility
	Tbx::Vec3  _offset;
	float _scale;

public:
	Mesh_static  _mesh_static;
	Mesh_half_edge _mesh_he;
public:
	Mesh_unpacked_attr _mesh_attr;

	Mesh_gl _mesh_gl;

};
#endif