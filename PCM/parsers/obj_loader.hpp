#ifndef OBJ_LOADER_HPP__
#define OBJ_LOADER_HPP__

#include <fstream>
#include <string>
#include <set>
#include <map>
#include <vector>
#include <iostream>

#include "parsers/loader.hpp"

/**	\file	obj_loader.hpp
 *  \author	Rob Bateman [mailto:robthebloke@hotmail.com] [http://robthebloke.org]
 *  \date	11-4-05
 *  \brief	A C++ loader for the alias wavefront obj file format.
 *
 * This loader is intended to be as complete as possible
 * (ie, it handles everything that Maya can spit at it,
 * as well as all the parametric curve and surface stuff not supported
 * by Maya's objexport plugin).
 *
 * If you want to add obj import/export capabilities to your applications,
 * the best method is probably to derive a class from Obj::File and then
 * use the data in the arrays provided (or fill the data arrays with the
 * correct info, then export).
 *
 * Generally speaking, the obj file format is actually very very complex.
 * As a result, not all of the Curve and Surface types are fully supported
 * in the openGL display parts of this code. Bezier, Bspline & NURBS curves
 * and surface should be fine. Cardinal, taylor and basis matrix types
 * will be displayed as CV's / HULLS only. Trim curves and holes don't get
 * displayed either. I'll leave that as an excersice for the reader.
 * The material lib files (*.mtl) are also supported and parsed, this
 * includes a user extensible interface to load the image files as textures.
 * See the texture manager example for more info.
 *
 * One nice aspect of this code is the ability to calculate normals for
 * the polygonal surfaces, and the ability to convert the objfile into
 * surfaces and /or vertex arrays. Hopefully it may be useful to someone.
 *
 * If you wish to use this code within a game, I **STRONGLY** RECOMMEND
 * converting the data to Vertex Arrays and saving as your own custom
 * file format. The code to handle your own objects will be substantially
 * more efficient than this code which is designed for completeness, not
 * speed! See the obj->game converter for a quick example of this.
*/

// TODO: Handle the smoothing groups ('s off' 's 1' 's 2' 's n'...)
// delete sscanf usage in the methods

// =============================================================================
namespace Loader {
// =============================================================================

// structs internal to obj loader ----------------------------------------------

namespace In {
struct Surface;
struct Vertex_buffer;
}

struct Wavefront_mesh;
struct Wavefront_data;
//------------------------------------------------------------------------------

/** @class Obj_File
    @brief main interface for an alias wavefront obj file.

    Note on the data structure :
    The provided list of data  (i.e : _vertices, _VertexParams, _normals,
    _texCoords, _points, _lines, _triangles, _groups, _mat_groups etc.)
    respect the .obj file layout. There is no garante that a list
    reference all the elements of another list.

    In other words : _triangles does not necessarily points to every vertices
    indices if there are lonely vertices. Another example : some groups don't
    have materials. this often means the material of the previous group should
    be used. We don't make that asumption and leave it at the user discretion,
    thus some group's material are empty.

    for faces without groups we make an exception and create  default group
    with name "" in order to reference every faces.

    _triangles layout is the same as the obj layout. Material groups are not
    necessarily contigus in _triangles. Therefore when rendering, doing a loop
    over the material groups doesn't garantee to be efficient as one material
    might be activated and desactivated several time.

*/
class Obj_file : public Base_loader {
    // vertex buffer needs to access custom materials
    friend struct Vertex_buffer;
    // surfaces may also need to access custom materials if split.
    friend struct Surface;
public:

    Obj_file();

    /// Parse and load file use 'get_mesh()' to get the abstract representation
    /// of the mesh
    Obj_file(const std::string& filename);

    ~Obj_file();

    /// The loader type
    Loader_t type() const { return OBJ; }

    bool import_file(const std::string& file_path);

    bool export_file(const std::string& file_path);

    //EObj::Flags fill_scene(Scene_tree& tree, EObj::Flags flag = 0)
    //{
    //    // Since obj loader only handles meshes we check only for meshes types
    //    if( flag == 0 || EObj::test(flag, EObj::MESH) )
    //    {
    //        Obj_mesh* o = new Obj_mesh();
    //        Abs_mesh mesh;
    //        get_mesh(mesh);
    //        o->load( mesh );
    //        tree.register_obj( o );
    //        return EObj::MESH;
    //    }

    //    return EObj::NONE;
    //}

    /// Obj files have no animation frames
    void get_anims(std::vector<Base_anim_eval*>& anims) const { anims.clear(); }

    /// transform internal representation into generic representation
    void get_mesh(Abs_mesh& mesh);
    /// transform generic representation into internal representation
    void set_mesh(const Abs_mesh& mesh);


private:

    // -------------------------------------------------------------------------
    /// @name class tools
    // -------------------------------------------------------------------------

    /// Init data pointers
    void init();

    /// releases all object data
    void release();

    /// loads the specified material file
    bool load_mtl(const char filename[]);

    /// saves the mtl file
    bool save_mtl(const char filename[]) const;

    /// this function will generate vertex normals for the current
    /// surface and store those within the m_Normals array
    void compute_normals();

    /// splits the obj file into seperate surfaces based upon object grouping.
    /// The returned list of surfaces will use indices relative to the start of
    /// this surface.
    void groups_to_surfaces(std::vector<In::Surface>& surface_list);

    /// splits the obj file into sets of vertex arrays for quick rendering.
    void groups_to_vertex_arrays(std::vector<In::Vertex_buffer>& surface_list);

    /// Eat the current line from the stream untill the character new line
    /// is reached
    /// @return The string of all eaten characters
    std::string eat_line(std::istream& ifs);

    std::string read_chunk(std::istream& ifs);

    /// utility function
    void read_points(std::istream&);

    /// utility function to parse
    void read_line(std::istream&);

    /// utility function to parse a face
    void read_face(std::istream&);

    /// a utility function to parse a group
    void read_group(std::istream&);

    /// a utility function to parse material file
    void read_material_lib(std::istream& ifs);

    /// a utility function to parse a use material statement
    void read_use_material(std::istream& ifs);

     // ------------------------------------------------------------------------
    /// @name Attributes
    // -------------------------------------------------------------------------
    /// Obj representation of the mesh (or multiples meshes)
    Wavefront_mesh* _mesh;
    Wavefront_data* _data;
};

//------------------------------------------------------------------------------

}// END Loader NAMESPACE =======================================================

#endif // OBJ_LOADER_HPP__
