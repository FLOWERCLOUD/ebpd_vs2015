
/** \file obj_loader.cpp
 * \author Rob Bateman, mailto:robthebloke@hotmail.com
 * \date 11-4-05
 * \brief A C++ objloader supporting materials, and any face data going, groups
 * (ie, different meshes), calculation of normals. All face data will be
 * triangulated (badly, though should work 99% of the time).
 */
//------------------------------------------------------------------------------

#include "parsers/obj_loader.hpp"
#include <algorithm>
#include <string>
#include <math.h> // TODO: use cmath
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>

#define FCOMPARE(x,y) (((x)-0.0001f)<(y) && ((x)+0.00001f)>(y))

// =============================================================================
namespace Loader {
// =============================================================================

// forward declarations --------------------------------------------------------

class Obj_file;

// structs internal to obj loader ----------------------------------------------

/// @brief holds internal data structure for wavefront objs
// =============================================================================
namespace In {
// =============================================================================

struct Vert {
    float x; ///< x component of point
    float y; ///< y component of point
    float z; ///< z component of point


    /// ctors
    Vert() : x(0),y(0),z(0) { }

    Vert(float _x, float _y, float _z) : x(_x), y(_y), z(_z) { }

    Vert( const Vert& v ) : x(v.x),y(v.y),z(v.z) { }

    /// ctor from stream
    Vert( std::istream& ifs ) { ifs >> x >> y >> z; }


    friend std::ostream& operator<< ( std::ostream& ofs, const Vert& v ) {
        return ofs << "v " << v.x << " " << v.y << " " << v.z << std::endl;
    }

    bool operator== ( const Vert& v ) const {
        return FCOMPARE(x,v.x) && FCOMPARE(y,v.y) && FCOMPARE(z,v.z);
    }
};

// -----------------------------------------------------------------------------

struct Vertex_param {
    float u; ///< x component of point
    float v; ///< y component of point
    float w; ///< z component of point


    /// ctors
    Vertex_param() : u(0),v(0),w(0) { }
    Vertex_param(float _u, float _v, float _w): u(_u), v(_v), w(_w) { }
    Vertex_param(const Vertex_param& v_) : u(v_.u), v(v_.v), w(v_.w) {  }

    /// ctor from input stream
    Vertex_param( std::istream& ifs )
    {
        std::stringbuf current_line_sb(std::ios::in);
        ifs.get( current_line_sb );
        std::istream current_line(&current_line_sb);
        current_line >> u >> v;
        if(!current_line.eof()) current_line >> w;
    }


    friend std::ostream& operator<< ( std::ostream& ofs, const Vertex_param& v_ ) {
        return ofs << "vp " << v_.u << " " << v_.v << " " << v_.w << std::endl;
    }
    bool operator== ( const Vertex_param& v_ ) const {
        return FCOMPARE(u,v_.u) && FCOMPARE(v,v_.v) && FCOMPARE(w,v_.w) ;
    }
};

// -----------------------------------------------------------------------------

struct Normal {
    float x; ///< x component of vector
    float y; ///< y component of vector
    float z; ///< z component of vector


    /// ctors
    Normal() : x(0),y(0),z(0) { }
    Normal(float _x, float _y, float _z) : x(_x), y(_y), z(_z) { }
    Normal(const In::Normal& n) : x(n.x), y(n.y), z(n.z) { }
    /// ctor from input stream
    Normal( std::istream& ifs ) { ifs >> x >> y >> z; }


    friend std::ostream& operator<< ( std::ostream& ofs, const In::Normal& n ) {
        return ofs << "vn " << n.x << " " << n.y << " " << n.z << std::endl;
    }
    bool operator== ( const In::Normal& n ) const {
        return FCOMPARE(x,n.x) && FCOMPARE(y,n.y) && FCOMPARE(z,n.z);
    }

};

// Normal Calculation Utilities ================================================

static void do_face_calc(const In::Vert& v1,
                         const In::Vert& v2,
                         const In::Vert& v3,
                         In::Normal& n1,
                         In::Normal& n2,
                         In::Normal& n3)
{
    // calculate vector between v2 and v1
    In::Vert e1;
    e1.x=	v1.x - v2.x;
    e1.y=	v1.y - v2.y;
    e1.z=	v1.z - v2.z;

    // calculate vector between v2 and v3
    In::Vert e2;
    e2.x=	v3.x - v2.x;
    e2.y=	v3.y - v2.y;
    e2.z=	v3.z - v2.z;

    // cross product them
    In::Vert e1_cross_e2;
    e1_cross_e2.x=	e2.y*e1.z - e2.z*e1.y;
    e1_cross_e2.y=	e2.z*e1.x - e2.x*e1.z;
    e1_cross_e2.z=	e2.x*e1.y - e2.y*e1.x;

    float itt = 1.0f/((float)sqrt( e1_cross_e2.x*e1_cross_e2.x +
                                  e1_cross_e2.y*e1_cross_e2.y +
                                  e1_cross_e2.z*e1_cross_e2.z ));

    // normalize
    e1_cross_e2.x *= itt;
    e1_cross_e2.y *= itt;
    e1_cross_e2.z *= itt;

    // sum the face normal into all the vertex normals this face uses
    n1.x += e1_cross_e2.x;
    n1.y += e1_cross_e2.y;
    n1.z += e1_cross_e2.z;

    n2.x += e1_cross_e2.x;
    n2.y += e1_cross_e2.y;
    n2.z += e1_cross_e2.z;

    n3.x += e1_cross_e2.x;
    n3.y += e1_cross_e2.y;
    n3.z += e1_cross_e2.z;
}

//------------------------------------------------------------------------------

static void normalize_normals(std::vector<In::Normal>& norms)
{
    std::vector<In::Normal>::iterator itn = norms.begin();
    for( ; itn != norms.end(); ++itn )
    {
        float itt = 1.0f/((float)sqrt( itn->x*itn->x +
                                       itn->y*itn->y +
                                       itn->z*itn->z ));
        itn->x *= itt;
        itn->y *= itt;
        itn->z *= itt;
    }
}

//------------------------------------------------------------------------------

static void zero_normals(std::vector<In::Normal>& norms)
{
    // zero normals
    std::vector<In::Normal>::iterator itn = norms.begin();
    for( ; itn != norms.end(); ++itn )
        itn->x = itn->y = itn->z = 0;
}

// END Normal Calculation Utilities ============================================

struct Tex_coord {
    float u; ///< u tex coord
    float v; ///< v tex coord


    /// ctors
    Tex_coord() : u(0),v(0) { }
    Tex_coord(float _u, float _v) : u(_u), v(_v) { }
    Tex_coord(const In::Tex_coord& uv) : u(uv.u),v(uv.v) { }
    /// ctor from stream
    Tex_coord( std::istream& ifs ) { ifs >> u >> v; }


    friend std::ostream& operator<< ( std::ostream& ofs, const In::Tex_coord& uv ) {
        return ofs << "vt " << uv.u << " " << uv.v << std::endl;
    }
    bool operator== ( const In::Tex_coord& uv ) const {
        return FCOMPARE(u,uv.u) && FCOMPARE(v,uv.v);
    }
};

// -----------------------------------------------------------------------------

struct Line {
    /// the material applied to this line
    unsigned short _material;
    /// the group to which this line belongs
    unsigned short _group;
    /// the vertex indices for the line
    std::vector<unsigned> _vertices;
    /// the texture coord indices for the line
    std::vector<unsigned> _tex_coords;


    /// ctors
    Line() : _vertices(), _tex_coords() { }
    Line(const In::Line& l) :
        _vertices(l._vertices),
        _tex_coords(l._tex_coords)
    {

    }


    friend std::ostream& operator<< ( std::ostream& ofs, const In::Line& l )
    {
        ofs << "l";
        if ( l._tex_coords.size() ) {
            std::vector<unsigned>::const_iterator itv= l._vertices.begin();
            std::vector<unsigned>::const_iterator itt= l._tex_coords.begin();
            for( ; itv != l._vertices.end();++itv) {
                ofs << " " << *itv << "/" << *itt;
            }
            ofs << "\n";
        } else {
            std::vector<unsigned>::const_iterator itv= l._vertices.begin();
            for( ; itv != l._vertices.end();++itv) {
                ofs << " " << *itv ;
            }
            ofs << "\n";
        }
        return ofs;
    }
};

// -----------------------------------------------------------------------------

struct Face {
    unsigned v[3]; ///< vertex indices for the triangle
    int n[3];      ///< normal indices for the triangle
    int t[3];      ///< texture coordinate indices for the triangle


    /// ctor
    Face() {
        v[0] = v[1] = v[2] =  0;
        // -1 indicates not used
        n[0] = n[1] = n[2] = -1;
        t[0] = t[1] = t[2] = -1;
    }

    friend std::ostream& operator<< ( std::ostream& ofs,const In::Face& f );
};

// -----------------------------------------------------------------------------

struct Material {
    /// material name
    std::string name;
    /// don't know :| Seems to always be 4
    int illum;
    float Ka[4]; ///< ambient
    float Kd[4]; ///< diffuse
    float Ks[4]; ///< specular
    float Tf[3]; ///< transparency
    float Ni;    ///< intensity
    float Ns;    ///< specular power
    std::string map_Ka;   ///< ambient texture map
    std::string map_Kd;   ///< diffuse texture map
    std::string map_Ks;   ///< specular texture map
    std::string map_Bump; ///< bump texture map
    /// bump map depth. Only used if bump is relevent.
    float Bm;


    /// ctors
    Material() : name(), illum(4), Ni(1), Ns(10), map_Ka(), map_Kd(), map_Ks(), map_Bump(), Bm(1)
    {
        Ka[0] = Ka[1] = Ka[2] = Kd[0] = Kd[1] = Kd[2] = Ks[0] = Ks[1] = Ks[2] = 0;
        Ka[3] = Kd[3] = Ks[3] = 1;
        Tf[0] = Tf[1] = Tf[2] = 1;
        illum = 4;
        Ni = Ns = 0.5f;
        map_Ka = map_Kd = map_Ks = map_Bump = "";
        Bm = 0;
    }

    Material(const In::Material& mat)
    {
        Ka[0] = mat.Ka[0]; Ka[1] = mat.Ka[1]; Ka[2] = mat.Ka[2]; Ka[3] = mat.Ka[3];
        Kd[0] = mat.Kd[0]; Kd[1] = mat.Kd[1]; Kd[2] = mat.Kd[2]; Kd[3] = mat.Kd[3];
        Ks[0] = mat.Ks[0]; Ks[1] = mat.Ks[1]; Ks[2] = mat.Ks[2]; Ks[3] = mat.Ks[3];
        Tf[0] = mat.Tf[0]; Tf[1] = mat.Tf[1]; Tf[2] = mat.Tf[2];
        Ni = mat.Ni;
        Ns = mat.Ns;
        name = mat.name;
        map_Ka = mat.map_Ka;
        map_Kd = mat.map_Kd;
        map_Ks = mat.map_Ks;
        map_Bump = mat.map_Bump;
        illum = mat.illum;
        Bm = mat.Bm;
    }

    /// dtor
    ~Material(){    }

    friend std::ostream& operator<< ( std::ostream& ofs, const In::Material& f );
};

// -----------------------------------------------------------------------------

struct Material_group {
    /// the material applied to a set of faces
    unsigned _material_idx;
    /// the starting index of the face to which the material is applied
    unsigned _start_face;
    /// the ending index of the face to which the material is applied
    unsigned _end_face;
    /// start index for points to which the material is applied
    unsigned _start_point;
    /// end index for points to which the material is applied
    unsigned _end_point;

    /// ctors
    Material_group() : _material_idx(0), _start_face(0), _end_face(0) {}
    Material_group(const In::Material_group& mg) :
        _material_idx(mg._material_idx),
        _start_face(mg._start_face),
        _end_face(mg._end_face)
    {    }
};

// -----------------------------------------------------------------------------

struct Group {
    /// start index for faces in the group (surface)
    unsigned _start_face;
    /// end index for faces in the group (surface)
    unsigned _end_face;
    /// start index for points in the group (surface)
    unsigned _start_point;
    /// end index for points in the group (surface)
    unsigned _end_point;
    /// name of the group
    std::string _name;
    /// a set of material groupings within this surface. ie, which
    /// materials are assigned to which faces within this group.
    std::vector<In::Material_group> _assigned_materials;

    /// ctors
    Group() : _start_face(0),_end_face(0),_name(""),_assigned_materials() { }
    Group(const In::Group& g)
        : _start_face(g._start_face),
          _end_face(g._end_face),
          _name(g._name),
          _assigned_materials(g._assigned_materials)
    {
    }
};

// -----------------------------------------------------------------------------

struct Bezier_patch {
    /// the material applied to this patch
    unsigned short _material;
    /// the group to which this patch belongs
    unsigned short _group;
    /// a set of 16 vertex indices
    int _vertex_indices[4][4];
    /// an array of vertices/normals/texcoords. Each vertex has 8 floats
    float* _vertex_data;
    /// an array of vertices/normals/texcoords. Each vertex has 8 floats
    float* _blend_funcs;
    /// an array of vertex indices for triangle strips
    unsigned* _index_data;
    /// the level of detail.
    unsigned _LOD;


    /// ctors
    Bezier_patch(){
        memset(this,0,sizeof(In::Bezier_patch));
        set_LOD(10);
    }

    Bezier_patch(const In::Bezier_patch& bzp)
    {
        // copy over all indices
        for(int i = 0; i != 4; ++i)
            for(int j = 0; j != 4; ++j)
                _vertex_indices[i][j] = bzp._vertex_indices[i][j];

        /// prevents SetLOD() from attempting to delete invalid data
        _index_data  = 0;
        _vertex_data = 0;
        _blend_funcs = 0;
        /// set level of detail of surface
        set_LOD(bzp._LOD);
    }

    /// dtor
    ~Bezier_patch(){
        delete [] _index_data;
        delete [] _vertex_data;
        delete [] _blend_funcs;
        _index_data  = 0;
        _vertex_data = 0;
        _blend_funcs = 0;
    }


    /// sets the level of detail and does a bit of internal caching to
    /// speed things up a little.
    void set_LOD(unsigned new_lod)
    {
        delete [] _vertex_data;
        delete [] _blend_funcs;
        delete [] _index_data;
        _LOD = new_lod;
        // allocate new blend funcs array. This just caches the values for tesselation
        _blend_funcs = new float[4*(_LOD+1)];
        float* ptr = _blend_funcs;
        for(unsigned i=0;i<=_LOD;++i) {
            float t = static_cast<float>(i/_LOD);
            float t2 = t*t;
            float t3 = t2*t;
            float it =1.0f-t;
            float it2 = it*it;
            float it3 = it2*it;

            *ptr = t3; ++ptr;
            *ptr = 3*it*t2; ++ptr;
            *ptr = 3*it2*t; ++ptr;
            *ptr = it3; ++ptr;

            // calculate texture coordinates since they never change
            {

            }
            // calculate texture coordinates since they never change
            {

            }
        }

        // allocate vertex data array
        _vertex_data = new float[ 8*(_LOD+1)*(_LOD+1) ];

        // allocate indices for triangle strips to render the patch
        _index_data = new unsigned [ (_LOD+1)*_LOD*2 ];

        {
            // calculate the vertex indices for the triangle strips.
            unsigned *iptr = _index_data;
            unsigned *end = _index_data + (_LOD+1)*_LOD*2;
            unsigned ii=0;
            for( ; iptr != end; ++ii) {
                *iptr = ii;       ++iptr;
                *iptr = ii+_LOD+1; ++iptr;
            }
        }
    }

    friend std::ostream& operator<< ( std::ostream& ofs, const In::Bezier_patch& bzp ) {
        ofs << "bzp ";
        for(int i=0;i!=4;++i)
            for(int j=0;j!=4;++j)
                ofs << " " << bzp._vertex_indices[i][j];
        return ofs << "\n";
    }
};

/// the obj file can be split into seperate surfaces
/// where all indices are relative to the data in the surface,
/// rather than all data in the obj file.
struct Surface {
    friend class Loader::Obj_file;
public:

    /// ctor
    Surface():
        _name(), _vertices(), _normals(),
        _tex_coords(), _triangles(), _assigned_materials()
    {

    }

    /// copy ctor
    Surface(const In::Surface& surface):
        _name(surface._name),
        _vertices(surface._vertices),
        _normals(surface._normals),
        _tex_coords(surface._tex_coords),
        _triangles(surface._triangles),
        _assigned_materials(surface._assigned_materials)
    {

    }

    /// this function will generate vertex normals for the current
    /// surface and store those within the m_Normals array
    void calculate_normals()
    {
        // resize normal array if not present
        if(!_normals.size())
            _normals.resize(_vertices.size());

        zero_normals( _normals );

        // loop through each triangle in face
        std::vector<In::Face>::iterator it = _triangles.begin();
        for( ; it != _triangles.end(); ++it )
        {
            // if no indices exist for normals, create them
            if(it->n[0] == -1) it->n[0] = it->v[0];
            if(it->n[1] == -1) it->n[1] = it->v[1];
            if(it->n[2] == -1) it->n[2] = it->v[2];

            // calc face normal and sum into normal array
            do_face_calc( _vertices[ it->v[0] ], _vertices[ it->v[1] ], _vertices[ it->v[2] ],
                          _normals[ it->n[0] ],_normals[ it->n[1] ],_normals[ it->n[2] ]);
        }

        normalize_normals( _normals );
    }

public:

    /// the name of the surface
    std::string _name;

    /// the vertices in the obj file
    std::vector<In::Vert> _vertices;
    /// the normals from the obj file
    std::vector<In::Normal> _normals;
    /// the tex coords from the obj file
    std::vector<In::Tex_coord> _tex_coords;
    /// the triangles in the obj file
    std::vector<In::Face> _triangles;
    /// the lines in the obj file
    std::vector<In::Line> _lines;
    /// the points in the obj file
    std::vector<unsigned> _points;
    /// a set of material groupings within this surface. ie, which
    /// materials are assigned to which faces within this group.
    std::vector<In::Material_group> _assigned_materials;


private:
    /// pointer to file to access material data
    Loader::Obj_file* _pfile;
};

//------------------------------------------------------------------------------

struct GL_Line {
    struct {
        unsigned int _num_verts:16;
        unsigned int _has_uvs:1;
        unsigned int _material:15;
    };
    /// the line indices in the obj file
    std::vector<unsigned int> _indices;
};

//------------------------------------------------------------------------------

/// The obj file can be split into seperate vertex arrays,
/// ie each group is turned into a surface which uses a
/// single index per vertex rather than seperate vertex,
/// normal and uv indices.
struct Vertex_buffer {
    friend class Loader::Obj_file ;
public:

    /// ctor
    Vertex_buffer():
        _name(),
        _vertices(),
        _normals(),
        _tex_coords(),
        _indices(),
        _assigned_materials(),
        _pfile(0)
    {

    }

    /// copy ctor
    Vertex_buffer(const In::Vertex_buffer& surface) :
        _name(surface._name),
        _vertices(surface._vertices),
        _normals(surface._normals),
        _tex_coords(surface._tex_coords),
        _indices(surface._indices),
        _assigned_materials(surface._assigned_materials),
        _pfile(surface._pfile)
    {

    }

    /// this function will generate vertex normals for the current
    /// surface and store those within the m_Normals array
    void calculate_normals()
    {
        // resize normal array if not present
        if(_normals.size()!=_vertices.size())
            _normals.resize(_vertices.size());

        zero_normals( _normals );

        // loop through each triangle in face
        std::vector<unsigned int>::const_iterator it = _indices.begin();
        for( ; it < _indices.end(); it+=3 )
            do_face_calc(_vertices[ *it ],_vertices[ *(it+1) ],_vertices[ *(it+2) ],
                       _normals[ *it ],_normals[ *(it+1) ],_normals[ *(it+2) ]);

        normalize_normals( _normals );
    }

public:

    /// the name of the surface
    std::string _name;

    /// the vertices in the obj file
    std::vector<In::Vert> _vertices;
    /// the normals from the obj file
    std::vector<In::Normal> _normals;
    /// the tex coords from the obj file
    std::vector<In::Tex_coord> _tex_coords;
    /// the triangles in the obj file
    std::vector<unsigned int> _indices;
    /// a set of material groupings within this surface. ie, which
    /// materials are assigned to which faces within this group.
    std::vector<In::Material_group> _assigned_materials;
    /// the lines in the obj file.
    std::vector<In::GL_Line> _lines;

private:
    /// pointer to file to access material data
    Loader::Obj_file* _pfile;
};



// File Reading Utils ==========================================================

std::ostream& operator << (std::ostream& ofs,const In::Face& f) {
    ofs << "f ";
    for(int i=0;i!=3;++i) {
        ofs << (f.v[i]+1);
        if(f.n[i] != -1 || f.t[i] != -1) {
            ofs << "/";
            if(f.t[i] != -1)
                ofs << (f.t[i]+1);
            ofs << "/";
            if(f.n[i] != -1)
                ofs << (f.n[i]+1);
        }
        ofs << " ";
    }
    return ofs << std::endl;
}

// -----------------------------------------------------------------------------

std::ostream& operator << (std::ostream& ofs, const In::Material& f)
{
    ofs << "newmtl " << f.name << "\n";
    ofs << "illum " << f.illum << "\n";
    ofs << "Kd " << f.Kd[0] << " " << f.Kd[1] << " " << f.Kd[2] << "\n";
    ofs << "Ka " << f.Ka[0] << " " << f.Ka[1] << " " << f.Ka[2] << "\n";
    ofs << "Ks " << f.Ks[0] << " " << f.Ks[1] << " " << f.Ks[2] << "\n";
    ofs << "Tf " << f.Tf[0] << " " << f.Tf[1] << " " << f.Tf[2] << "\n";
    ofs << "Ni " << f.Ni << "\n";
    ofs << "Ns " << f.Ns << "\n";
    if(f.map_Kd.size())
        ofs << "map_Kd " << f.map_Kd << "\n";
    if(f.map_Ka.size())
        ofs << "map_Ka " << f.map_Ka << "\n";
    if(f.map_Ks.size())
        ofs << "map_Ks " << f.map_Ks << "\n";
    if(f.map_Bump.size())
        ofs <<"bump " << f.map_Bump << " -bm " << f.Bm << "\n";
    return ofs;
}

// -----------------------------------------------------------------------------

static bool has_only_vertex(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    for( ; it != s.end(); ++it ) {
        if(*it=='/')
            return false;
    }
    return true;
}

// -----------------------------------------------------------------------------

static bool missing_uv(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while( *it != '/') {
        if(it == s.end())
            return true;
        ++it;
    }
    return *(it+1)=='/';
}

// -----------------------------------------------------------------------------

static bool missing_normal(const std::string& s)
{
    return s[s.size()-1] == '/';
}

// -----------------------------------------------------------------------------

/// quick utility function to copy a range of data from the obj file arrays
/// into a surface array.
template< typename T >
static void copy_array(std::vector< T >& output,
                       const std::vector< T >& input,
                       unsigned start,
                       unsigned end)
{
    output.resize( end-start+1 );
    typename std::vector< T >::iterator       ito   = output.begin();
    typename std::vector< T >::const_iterator it    = input.begin() + start;
    typename std::vector< T >::const_iterator itend = input.begin() + end + 1;
    for( ; it != itend; ++it,++ito )
    {
        *ito = *it;
    }
}

// -----------------------------------------------------------------------------

static
void determine_index_range(unsigned int& s_vert,
                           unsigned int& e_vert,
                           int& s_norm,
                           int& e_norm,
                           int& s_uv,
                           int& e_uv,
                           std::vector<In::Face>::const_iterator it,
                           std::vector<In::Face>::const_iterator end)
{
    // need to determine start and end vertex/normal and uv indices
    s_vert=0xFFFFFFF;
    s_norm=0xFFFFFFF;
    s_uv  =0xFFFFFFF;
    e_vert= 0;
    e_norm=-1;
    e_uv  =-1;

    // loop through faces to find max/min indices
    for( ; it != end; ++it ) {
        for(int i=0;i!=3;++i) {
            if(it->v[i]<s_vert)	s_vert = it->v[i];
            if(it->v[i]>e_vert)	e_vert = it->v[i];
            if(it->n[i]!=-1) {
                if(it->n[i]<s_norm) s_norm = it->n[i];
                if(it->n[i]>e_norm) e_norm = it->n[i];
            }
            if(it->t[i]!=-1) {
                if(it->t[i]<s_uv) s_uv = it->t[i];
                if(it->t[i]>e_uv) e_uv = it->t[i];
            }
        }
    }
}

// -----------------------------------------------------------------------------

template< typename T >
static void write_array_range(std::ostream& ofs,
                              const std::vector< T >& the_array,
                              unsigned start,
                              unsigned end)
{
    typename std::vector< T >::const_iterator it    = the_array.begin() + start;
    typename std::vector< T >::const_iterator itend = the_array.begin() + end + 1;
    for( ; it != itend; ++it )
        ofs << *it;
}

// END File Reading Utils ======================================================

}// End In namespace ===========================================================

/**
  @class Wavefront_mesh
  @brief Internal representation of a mesh for the Obj_file class
  The class Obj_file in charge of loading '.obj' files use this structure
  to store the mesh once the file is parsed
  @see Obj_file
*/
struct Wavefront_mesh {

    Wavefront_mesh(){}

    std::vector<In::Vert>      _vertices;  ///< the vertices in the .obj
    std::vector<In::Normal>    _normals;   ///< the normals from the .obj
    std::vector<In::Tex_coord> _texCoords; ///< the tex coords from the .obj
    std::vector<In::Face>      _triangles; ///< the triangles in the .obj
    std::vector<In::Group>     _groups;    ///< the groups in the .obj
    std::vector<In::Material>  _materials; ///< the materials from the .mtl
};

// -----------------------------------------------------------------------------

/**
  @class Wavefront_data
  @brief Internal representation of various datas for the Obj_file class
  The class Obj_file in charge of loading '.obj' files use this structure
  to store bezier patches points etc. once the file is parsed
  @see Obj_file
*/
struct Wavefront_data {

    Wavefront_data() {}

    std::vector<In::Vertex_param> _vertexParams;
    std::vector<In::Bezier_patch> _patches;
    std::vector<unsigned>         _points;
    std::vector<In::Line>         _lines;
};

// CLASS Obj_File ==============================================================

std::string Obj_file::read_chunk(std::istream& ifs)
{
    std::string s;
    do {
        char c = ifs.get();
        if (c=='\\') {
            while (ifs.get()!='\n') { /*empty*/
                if (ifs.eof()) {
                    break;
                }
            }
        } else if (c != '\n') {
            break;
        } else
            s += c;

        if (ifs.eof()) {
            break;
        }
    } while (1);
    return s;
}

// -----------------------------------------------------------------------------

Obj_file::Obj_file() : _mesh(0)
{
    init();
}

// -----------------------------------------------------------------------------

Obj_file::Obj_file(const std::string& filename) :
    Loader::Base_loader(filename),
    _mesh(0)
{
    init();
    import_file(filename);
}

// -----------------------------------------------------------------------------

Obj_file::~Obj_file(){
    release();
}

// -----------------------------------------------------------------------------

void Obj_file::init()
{
    _mesh = new Wavefront_mesh();
    _data = new Wavefront_data();
}

// -----------------------------------------------------------------------------

/// releases all object data
void Obj_file::release()
{
    delete _mesh;
    delete _data;
}

//------------------------------------------------------------------------------

void Obj_file::read_points(std::istream& ifs)
{
    char c;
    std::vector<std::string> VertInfo;

    c = ifs.get();
    // store all strings
    do {
        // strip white spaces
        if (ifs.eof()) {
            goto vinf;
        }
        while(c==' ' || c=='\t') {
            c=ifs.get();
            if (c=='\\') {
                while (ifs.get()!='\n') {
                    if (ifs.eof()) {
                        goto vinf;
                    }
                }
                c=ifs.get();
            }
            if (ifs.eof()) {
                goto vinf;
            }
        }
        std::string s;

        // read vertex info
        while(c!=' '&&c!='\t'&&c!='\n') {
            s+=c;
            c=ifs.get();
            if (ifs.eof()) {
                goto vinf;
            }
        }

        // store string
        VertInfo.push_back(s);
    } while(c!='\n'); // loop till end of line
    vinf: ;
    std::vector<std::string>::iterator it= VertInfo.begin();
    for( ; it != VertInfo.end(); ++it ) {
        int i;
        sscanf(it->c_str(),"%d",&i);
        if (i<0) {
            i=static_cast<int>(_mesh->_vertices.size())+i;
        } else
            --i;
        _data->_points.push_back(i);
    }
}

//------------------------------------------------------------------------------

void Obj_file::read_line(std::istream& ifs)
{
    using namespace In;
    char c;
    std::vector<std::string> VertInfo;

    c = ifs.get();
    // store all strings
    do {
        // strip white spaces
        if (ifs.eof()) {
            goto vinf;
        }
        while(c==' ' || c=='\t') {
            c=ifs.get();
            if (c=='\\') {
                while (ifs.get()!='\n') {
                    if (ifs.eof()) {
                        goto vinf;
                    }
                }
                c=ifs.get();
            }
            if (ifs.eof()) {
                goto vinf;
            }
        }
        std::string s;

        // read vertex info
        while(c!=' '&&c!='\t'&&c!='\n') {
            s+=c;
            c=ifs.get();
            if (ifs.eof()) {
                goto vinf;
            }
        }

        // store string
        VertInfo.push_back(s);
    } while(c!='\n'); // loop till end of line
    vinf: ;
    In::Line l;

    l._vertices.resize(VertInfo.size());
    l._tex_coords.resize(_mesh->_texCoords.size());

    std::vector<std::string>::iterator it= VertInfo.begin();
    for( ; it != VertInfo.end(); ++it ) {
        if(has_only_vertex(*it))
        {
            int i;
            sscanf(it->c_str(),"%d",&i);
            if (i<0) {
                i=static_cast<int>(_mesh->_vertices.size())+i;
            } else
                --i;
            l._vertices.push_back(i);
        }
        else
        {
            int i,j;
            sscanf(it->c_str(),"%d/%d",&i,&j);
            if (i<0) {
                i=static_cast<int>(_mesh->_vertices.size())+i;
            } else
                --i;
            if (j<0) {
                j=static_cast<int>(_mesh->_texCoords.size())+j;
            } else
                --j;
            l._vertices.push_back(i);
            l._tex_coords.push_back(j);
        }
    }
    _data->_lines.push_back(l);
}

//------------------------------------------------------------------------------

void Obj_file::read_face(std::istream& ifs)
{
    using namespace In;
    char c;
    std::vector<std::string> VertInfo;

    // store all strings
    do {
        // strip white spaces
        c = ifs.get();
        if (ifs.eof()) {
            goto vinf;
        }
        while(c==' ' || c=='\t') {
            c=ifs.get();
            if (ifs.eof()) {
                goto vinf;
            }
        }
        std::string s;

        // read vertex info
        while(c!=' '&&c!='\t'&&c!='\n') {
            s+=c;
            c=ifs.get();
            if (ifs.eof()) {
                goto vinf;
            }
        }

        // store string
        VertInfo.push_back(s);
    } while(c!='\n'); // loop till end of line

    vinf: ;
    std::vector<int> verts;
    std::vector<int> norms;
    std::vector<int> uvs;
    // split strings into individual indices
    std::vector<std::string>::const_iterator it = VertInfo.begin();
    for( ; it != VertInfo.end(); ++it ) {
        int v, n=0, t=0;

        if(has_only_vertex(*it))
            sscanf(it->c_str(),"%d",&v);
        else
            if(missing_uv(*it))
                sscanf(it->c_str(),"%d//%d",&v,&n);
            else
                if(missing_normal(*it))
                    sscanf(it->c_str(),"%d/%d/",&v,&t);
                else
                    sscanf(it->c_str(),"%d/%d/%d",&v,&t,&n);

        if (v<0) {
            v=static_cast<int>(_mesh->_vertices.size())+v+1;
        }
        if (n<0) {
            n=static_cast<int>(_mesh->_normals.size())+n+1;
        }
        if (t<0) {
            t=static_cast<int>(_mesh->_texCoords.size())+t+1;
        }

        // obj indices are 1 based, change them to zero based indices
        --v; --n; --t;

        verts.push_back(v);
        norms.push_back(n);
        uvs.push_back(t);
    }

    // construct triangles from indices
    for(unsigned i = 2; i < verts.size(); ++i)
    {
        In::Face f;

        // construct triangle
        f.v[0] = verts[0];
        f.n[0] = norms[0];
        f.t[0] = uvs[0];
        f.v[1] = verts[i-1];
        f.n[1] = norms[i-1];
        f.t[1] = uvs[i-1];
        f.v[2] = verts[i];
        f.n[2] = norms[i];
        f.t[2] = uvs[i];

        // append to list
        _mesh->_triangles.push_back(f);
    }
}

//------------------------------------------------------------------------------

void Obj_file::read_group(std::istream& ifs)
{
    std::string s;
    ifs >> s;
    // ignore the default group, it just contains the verts, normals & uv's
    // for all surfaces. Might as well ignore it!
    if(s!="default")
    {
        if( _mesh->_groups.size() )
        {
            In::Group& gr = _mesh->_groups[_mesh->_groups.size()-1];
            gr._end_face = static_cast<unsigned int>(_mesh->_triangles.size());

            if(gr._assigned_materials.size())
                gr._assigned_materials[gr._assigned_materials.size()-1]._end_face
                        = static_cast<unsigned int>(_mesh->_triangles.size());
        }

        In::Group g;
        g._name = s;
        g._start_face = static_cast<unsigned int>(_mesh->_triangles.size());
        _mesh->_groups.push_back(g);
    }
}

// -----------------------------------------------------------------------------

void Obj_file::read_material_lib(std::istream& ifs)
{
    std::string file_name;
    ifs >> file_name;

    std::string mtl_file = _path+file_name;

    if( !load_mtl(mtl_file.c_str()) )
        std::cerr << "[WARNING] Unable to load material file: "+mtl_file+"\n";
}

//------------------------------------------------------------------------------

void Obj_file::read_use_material(std::istream& ifs)
{
    std::string mat_name;
    ifs >> mat_name;

    if(_mesh->_materials.size())
    {
        // find material index
        unsigned mat=0;
        for( ; mat != _mesh->_materials.size(); ++mat ) {
            if(_mesh->_materials[mat].name == mat_name) {
                break;
            }
        }

        // if found
        if(mat != _mesh->_materials.size())
        {

            // no groups ? add a default group
            if(_mesh->_groups.size() == 0)
            {
                In::Group g;
                g._name = "";      // no name for the default group
                g._start_face = 0u; // groups every lonely faces
                _mesh->_groups.push_back(g);
            }

            In::Group& gr = _mesh->_groups[_mesh->_groups.size()-1];
            if(gr._assigned_materials.size())
                gr._assigned_materials[gr._assigned_materials.size()-1]._end_face
                        = static_cast<unsigned int>(_mesh->_triangles.size());

            In::Material_group mg;
            mg._material_idx = mat;
            mg._start_face = static_cast<unsigned int>(_mesh->_triangles.size());
            gr._assigned_materials.push_back(mg);
        }
    }
}

//------------------------------------------------------------------------------

std::string Obj_file::eat_line(std::istream& ifs)
{
    char c = 'a';
    std::string line;
    while( !ifs.eof() )
    {
         c = ifs.get();
         if(c == '\n') break;

         line += c;
    }
    return line;
}

//------------------------------------------------------------------------------

bool Obj_file::import_file(const std::string& file_path)
{
    Loader::Base_loader::update_paths( file_path );

    // just in case a model is already loaded reset memory
    release();
    init();

    std::ifstream ifs( file_path.c_str() );
    if (!ifs) return false;

    // loop through the file to the end
    unsigned line = 0;
    while(!ifs.eof())
    {
        std::string s;

        ifs >> s;

        line++;
        if(s.size() == 0)
            continue;
        else if(s[0]=='#') // comment, skip line
            eat_line(ifs);
        else if(s=="deg")
            std::cerr << "[ERROR] Unable to handle deg yet. Sorry! RB.\n";
        else if(s=="cstype")  // a new group of faces, ie a seperate mesh
            std::cerr << "[ERROR] Unable to handle cstype yet. Sorry! RB.\n";
        else if(s=="bzp") // a new group of faces, ie a seperate mesh
        {
            In::Bezier_patch bzp;
            std::string text = read_chunk(ifs);

            sscanf(text.c_str(),"%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d",
                   &bzp._vertex_indices[0][0],
                   &bzp._vertex_indices[0][1],
                   &bzp._vertex_indices[0][2],
                   &bzp._vertex_indices[0][3],

                   &bzp._vertex_indices[1][0],
                   &bzp._vertex_indices[1][1],
                   &bzp._vertex_indices[1][2],
                   &bzp._vertex_indices[1][3],

                   &bzp._vertex_indices[2][0],
                   &bzp._vertex_indices[2][1],
                   &bzp._vertex_indices[2][2],
                   &bzp._vertex_indices[2][3],

                   &bzp._vertex_indices[3][0],
                   &bzp._vertex_indices[3][1],
                   &bzp._vertex_indices[3][2],
                   &bzp._vertex_indices[3][3]);

            // subtract 1 from all indices
            for(unsigned i=0;i!=4;++i)
                for(unsigned j=0;j!=4;++j)
                    --bzp._vertex_indices[i][j];
        }
        else if(s=="g") // a new group of faces, ie a seperate mesh
            read_group(ifs);
        else if(s=="f"||s=="fo") // face
            read_face(ifs);
        else if(s=="p") // points
            read_points(ifs);
        else if(s=="l") // lines
            read_line(ifs);
        else if(s=="vt") // texture coord
            _mesh->_texCoords.push_back( In::Tex_coord(ifs) );
        else if(s=="vn") // normal
            _mesh->_normals.push_back( In::Normal(ifs) );
        else if(s=="v") // vertex
            _mesh->_vertices.push_back( In::Vert(ifs) );
        else if(s=="vp") // vertex parameter
            _data->_vertexParams.push_back( In::Vertex_param(ifs) );
        else if(s=="mtllib") // material library
            read_material_lib(ifs);
        else if(s=="usemtl") // material to apply
            read_use_material(ifs);
        else if(s=="end"  || s=="parm"|| s=="stech"|| s=="ctech"|| s=="curv"||
                s=="curv2"|| s=="surf"|| s=="bmat" || s=="res"  || s=="sp"  ||
                s=="trim" || s=="hole")
        {

            std::cerr << "[ERROR] Unable to handle " << s << " outside of cstype/end pair\n";
            std::cerr << "[ERROR] Unable to handle cstype yet. Sorry! RB.\n";
            read_chunk(ifs);
        }
        else
        {
            std::string field = s + eat_line(ifs);

            std::cerr << "WARNING line " << line << ": the field '"<< field;
            std::cerr << "' could not be read in file ";
            std::cerr << file_path << std::endl;
        }
    }// END WHILE( END_OF_FILE )

    // if groups exist, terminate it.
    if(_mesh->_groups.size())
    {
        In::Group& gr = _mesh->_groups[_mesh->_groups.size()-1];

        gr._end_face = static_cast<unsigned int>(_mesh->_triangles.size());

        // terminate any assigned materials
        if(gr._assigned_materials.size())
            gr._assigned_materials[gr._assigned_materials.size()-1]._end_face
                    = static_cast<unsigned int>(_mesh->_triangles.size());
    }

    return true;
}

//------------------------------------------------------------------------------

bool Obj_file::export_file(const std::string& filename)
{
    using namespace In;
    Loader::Base_loader::update_paths( filename );

    // if we have materials in the model, save the mtl file
    if(_mesh->_materials.size()) {
        std::string file = filename;
        size_t len = file.size();
        // strip "obj" extension
        while(file[--len] != '.') /*empty*/ ;
        file.resize(len);
        file += ".mtl";
        save_mtl(file.c_str());
    }

    std::ofstream ofs(filename.c_str());
    if( !ofs.is_open() ) return false;

    if(_mesh->_groups.size())
    {
        std::vector<In::Group>::const_iterator itg = _mesh->_groups.begin();
        for( ; itg != _mesh->_groups.end(); ++itg )
        {
            // need to determine start and end vertex/normal and uv indices
            unsigned int s_vert,e_vert;
            int s_norm,s_uv,e_norm,e_uv;

            determine_index_range(s_vert,e_vert,s_norm,e_norm,s_uv,e_uv,
                                _mesh->_triangles.begin() + itg->_start_face,
                                _mesh->_triangles.begin() + itg->_end_face);

            // write default group
            ofs << "g default\n";

            // write groups vertices
            write_array_range<In::Vert>(ofs,_mesh->_vertices,s_vert,e_vert);

            // write groups normals (if present)
            if(e_norm != -1)
                write_array_range<In::Normal>(ofs,_mesh->_normals,s_norm,e_norm);

            // write groups uv coords (if present)
            if(e_uv != -1)
                write_array_range<In::Tex_coord>(ofs,_mesh->_texCoords,s_uv,e_uv);

            // write group name
            ofs << "g " << itg->_name << std::endl;

            // write triangles in group
            if( itg->_assigned_materials.size() )
            {
                // write out each material group
                std::vector<In::Material_group>::const_iterator itmg = itg->_assigned_materials.begin();
                for( ; itmg != itg->_assigned_materials.end(); ++itmg ) {
                    unsigned int mat = itmg->_material_idx;
                    // write use material flag
                    ofs << "usemtl " << _mesh->_materials[mat].name << "\n";

                    write_array_range<In::Face>(ofs,_mesh->_triangles,itmg->_start_face,itmg->_end_face-1);
                }
            }
            else
                write_array_range<In::Face>(ofs,_mesh->_triangles,itg->_start_face,itg->_end_face-1);
        }
    }
    else
    {
        // all part of default group
        ofs << "g default\n";
        write_array_range<In::Vert>(ofs,_mesh->_vertices,0,static_cast<unsigned>(_mesh->_vertices.size())-1);
        write_array_range<In::Tex_coord>(ofs,_mesh->_texCoords,0,static_cast<unsigned>(_mesh->_texCoords.size())-1);
        write_array_range<In::Normal>(ofs,_mesh->_normals,0,static_cast<unsigned>(_mesh->_normals.size())-1);
        write_array_range<In::Face>(ofs,_mesh->_triangles,0,static_cast<unsigned>(_mesh->_triangles.size())-1);
    }
    ofs.close();
    return true;


}

//------------------------------------------------------------------------------

// loads the specified material file
bool Obj_file::load_mtl(const char filename[])
{
    std::ifstream ifs(filename);

    if(!ifs.is_open()) return false;

    In::Material* pmat = 0;
    while(!ifs.eof())
    {
        std::string s;
        ifs >> s;

        // To many discrepancies with obj implems so I lower everything to
        // catch as much as possible symbols
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        if(s.size() == 0)
            continue;
        else if(s[0]=='#')
            eat_line(ifs);
        else if(s == "newmtl")
        {
            In::Material mat;
            ifs >> mat.name;
            _mesh->_materials.push_back(mat);
            pmat = &(_mesh->_materials[_mesh->_materials.size()-1]);
        }
        else if(s == "illum") ifs >> pmat->illum;
        else if(s == "kd") ifs >> pmat->Kd[0] >> pmat->Kd[1] >> pmat->Kd[2];
        else if(s == "ka") ifs >> pmat->Ka[0] >> pmat->Ka[1] >> pmat->Ka[2];
        else if(s == "ks") ifs >> pmat->Ks[0] >> pmat->Ks[1] >> pmat->Ks[2];
        else if(s == "tf") ifs >> pmat->Tf[0] >> pmat->Tf[1] >> pmat->Tf[2];
        else if(s == "d"){ ifs >> pmat->Tf[0]; pmat->Tf[1] = pmat->Tf[2] = pmat->Tf[0]; }
        else if(s == "ni") ifs >> pmat->Ni;
        else if(s == "ns") ifs >> pmat->Ns;
        else if(s == "map_ka") ifs >> pmat->map_Ka;
        else if(s == "map_kd") ifs >> pmat->map_Kd;
        else if(s == "map_ks") ifs >> pmat->map_Ks;
        else if(s == "bump" || s == "map_bump")
            ifs >> pmat->map_Bump;
        else // unknown entry, we skip it
        {
            std::string field = s + eat_line(ifs);

            std::cerr << "WARNING: the material field '"<< field << "' could not be read in file ";
            std::cerr << filename << std::endl;
        }
    }// END WHILE( END_OF_FILE )
    ifs.close();
    return true;
}

//------------------------------------------------------------------------------

// loads the specified material file
bool Obj_file::save_mtl(const char filename[]) const
{
    using namespace In;
    std::ofstream ofs(filename);
    if(ofs)
    {
        std::vector<In::Material>::const_iterator it = _mesh->_materials.begin();
        for( ; it != _mesh->_materials.end(); ++it)
            ofs << *it;
        ofs.close();
        return true;
    }
    return false;
}

//------------------------------------------------------------------------------

void Obj_file::compute_normals()
{
    if(!_mesh->_triangles.size()) return;

    if(_mesh->_triangles[0].n[0] == -1)
    {
        _mesh->_normals.resize(_mesh->_vertices.size());


        std::vector<In::Face>::iterator it = _mesh->_triangles.begin();
        for( ; it != _mesh->_triangles.end(); ++it ) {

            // use vertex indices for normal indices
            it->n[0] = it->v[0];
            it->n[1] = it->v[1];
            it->n[2] = it->v[2];
        }
    }

    // resize normal array if not present
    zero_normals( _mesh->_normals );

    // loop through each triangle in face
    std::vector<In::Face>::const_iterator it = _mesh->_triangles.begin();
    for( ; it != _mesh->_triangles.end(); ++it ) {
        do_face_calc(_mesh->_vertices[ it->v[0] ],_mesh->_vertices[ it->v[1] ],_mesh->_vertices[ it->v[2] ],
                   _mesh->_normals[ it->n[0] ],_mesh->_normals[ it->n[1] ],_mesh->_normals[ it->n[2] ]);
    }
    normalize_normals( _mesh->_normals );
}

//------------------------------------------------------------------------------

void Obj_file::groups_to_surfaces(std::vector<In::Surface>& surface_list)
{
    using namespace In;

    if(_mesh->_groups.size())
    {
        surface_list.resize(_mesh->_groups.size());

        std::vector<In::Surface>::iterator its = surface_list.begin();

        std::vector<In::Group>::const_iterator itg = _mesh->_groups.begin();
        for( ; itg != _mesh->_groups.end(); ++itg, ++its ) {

            // need to determine start and end vertex/normal and uv indices
            unsigned int s_vert,e_vert;
            int s_norm,s_uv,e_norm,e_uv;

            determine_index_range(s_vert,e_vert,s_norm,e_norm,s_uv,e_uv,
                                _mesh->_triangles.begin() + itg->_start_face,
                                _mesh->_triangles.begin() + itg->_end_face);

            // set file pointer
            its->_pfile = this;

            // set name
            its->_name = itg->_name;

            // copy material groups for surface
            its->_assigned_materials = itg->_assigned_materials;

            // make material groups relative to material start
            std::vector<In::Material_group>::iterator itmg = its->_assigned_materials.begin();
            for( ;itmg != its->_assigned_materials.end(); ++itmg ) {
                itmg->_start_face -= itg->_start_face;
                itmg->_end_face   -= itg->_start_face;
            }

            // resize triangles
            its->_triangles.resize( itg->_end_face - itg->_start_face );

            std::vector<In::Face>::iterator ito = its->_triangles.begin();
            std::vector<In::Face>::const_iterator it = _mesh->_triangles.begin() + itg->_start_face;
            std::vector<In::Face>::const_iterator end = _mesh->_triangles.begin() + itg->_end_face;

            for( ; it != end; ++it, ++ito ) {
                for(int i=0;i!=3;++i) {
                    ito->v[i] = it->v[i]-s_vert;
                    ito->n[i] = (e_norm==-1) ? -1 : (it->n[i]-s_norm);
                    ito->t[i] = (e_uv==-1) ? -1 : (it->t[i]-s_uv);
                }
            }

            // copy over vertices
            copy_array< In::Vert >( its->_vertices, _mesh->_vertices, s_vert, e_vert );

            // copy over normals
            if(e_norm!=-1)
                copy_array< In::Normal >( its->_normals, _mesh->_normals, s_norm, e_norm );

            // copy over tex coords
            if(e_uv!=-1)
                copy_array< In::Tex_coord >( its->_tex_coords, _mesh->_texCoords, s_uv, e_uv );
        }
    }
    else {
        surface_list.resize(1);
        surface_list[0]._vertices   = _mesh->_vertices;
        surface_list[0]._normals    = _mesh->_normals;
        surface_list[0]._tex_coords = _mesh->_texCoords;
        surface_list[0]._triangles  = _mesh->_triangles;
        surface_list[0]._pfile      = this;
        surface_list[0]._name       = "default";
    }
}

//------------------------------------------------------------------------------

void Obj_file::groups_to_vertex_arrays(std::vector<In::Vertex_buffer>& surface_list)
{
    // first split into surfaces
    std::vector<In::Surface> surfaces;
    groups_to_surfaces(surfaces);

    // now convert each surface into a vertex array
    surface_list.resize(surfaces.size());

    std::vector<In::Vertex_buffer>::iterator itb = surface_list.begin();
    std::vector<In::Surface>::iterator its = surfaces.begin();

    for( ; itb != surface_list.end(); ++itb, ++its )
    {
        // set name
        itb->_name = its->_name;

        // set file
        itb->_pfile = this;

        // copy material assignments
        itb->_assigned_materials = its->_assigned_materials;

        // determine new vertex and index arrays.
        std::vector<In::Face>::iterator itf = its->_triangles.begin();
        for( ; itf != its->_triangles.end(); ++itf )
        {
            for(int i=0; i != 3; ++i)
            {
                const In::Vert* v = &its->_vertices[ itf->v[i] ];
                const In::Normal* n = 0;
                if(itf->n[i] != -1)
                    n = &its->_normals[ itf->n[i] ];

                const In::Tex_coord* t = 0;
                if(itf->t[i] != -1)
                    t = &its->_tex_coords[ itf->t[i] ];

                unsigned int idx = 0;
                if( n && t)
                {
                    std::vector<In::Vert>::const_iterator itv = itb->_vertices.begin();
                    std::vector<In::Normal>::const_iterator itn = itb->_normals.begin();
                    std::vector<In::Tex_coord>::const_iterator itt = itb->_tex_coords.begin();
                    for( ; itv != itb->_vertices.end(); ++itv,++itn,++itt,++idx )
                        if( *v == *itv && *n == *itn && *t == *itt )
                            break;

                    if( itv == itb->_vertices.end() )
                    {
                        itb->_vertices.push_back(*v);
                        itb->_normals.push_back(*n);
                        itb->_tex_coords.push_back(*t);
                    }
                    itb->_indices.push_back(idx);
                }
                else if(n)
                {
                    std::vector<In::Vert>::const_iterator itv = itb->_vertices.begin();
                    std::vector<In::Normal>::const_iterator itn = itb->_normals.begin();
                    for( ; itv != itb->_vertices.end(); ++itv,++itn,++idx )
                        if( *v == *itv && *n == *itn )
                            break;

                    if( itv == itb->_vertices.end() )
                    {
                        itb->_vertices.push_back(*v);
                        itb->_normals.push_back(*n);
                    }
                    itb->_indices.push_back(idx);
                }
                else if(t)
                {
                    std::vector<In::Vert   >::const_iterator itv = itb->_vertices.begin();
                    std::vector<In::Tex_coord>::const_iterator itt = itb->_tex_coords.begin();
                    for( ; itv != itb->_vertices.end(); ++itv,++itt,++idx )
                        if( *v == *itv && *t == *itt )
                            break;

                    if( itv == itb->_vertices.end() )
                    {
                        itb->_vertices.push_back(*v);
                        itb->_tex_coords.push_back(*t);
                    }
                    itb->_indices.push_back(idx);
                }
                else
                {
                    std::vector<In::Vert>::const_iterator itv = itb->_vertices.begin();
                    for( ; itv != itb->_vertices.end(); ++itv,++idx )
                        if( *v == *itv )
                            break;

                    itb->_indices.push_back(idx);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------

void Obj_file::get_mesh(Loader::Abs_mesh& mesh)
{
    mesh._mesh_path = _path;

    mesh._vertices.clear();
    mesh._vertices.resize( _mesh->_vertices.size() );
    for (unsigned i=0; i<_mesh->_vertices.size(); ++i)
    {
        In::Vert v = _mesh->_vertices[i];
        mesh._vertices[i] = Loader::Vertex( v.x, v.y, v.z );
    }

    mesh._normals.clear();
    mesh._normals.resize( _mesh->_normals.size() );
    for (unsigned i=0; i<_mesh->_normals.size(); ++i)
    {
        In::Normal n = _mesh->_normals[i];
        mesh._normals[i] = Loader::Normal( n.x, n.y, n.z );
    }

    mesh._texCoords.clear();
    mesh._texCoords.resize( _mesh->_texCoords.size() );
    for (unsigned i=0; i<_mesh->_texCoords.size(); ++i)
    {
        In::Tex_coord t = _mesh->_texCoords[i];
        mesh._texCoords[i] = Loader::Tex_coord( t.u, t.v );
    }

    mesh._triangles.clear();
    mesh._triangles.resize( _mesh->_triangles.size() );
    for (unsigned i=0; i<_mesh->_triangles.size(); ++i)
    {
        In::Face f = _mesh->_triangles[i];
        Loader::Tri_face F;
        F.v[0] = f.v[0]; F.v[1] = f.v[1]; F.v[2] = f.v[2];
        F.n[0] = f.n[0]; F.n[1] = f.n[1]; F.n[2] = f.n[2];
        F.t[0] = f.t[0]; F.t[1] = f.t[1]; F.t[2] = f.t[2];
        mesh._triangles[i] = F;
    }

    mesh._materials.clear();
    mesh._materials.resize( _mesh->_materials.size() );
    for (unsigned i=0; i<_mesh->_materials.size(); ++i)
    {
        In::Material m = _mesh->_materials[i];
        Loader::Material M;
        M._name = m.name;
        M._illum = m.illum;
        M._Ka[0] = m.Ka[0]; M._Ka[1] = m.Ka[1]; M._Ka[2] = m.Ka[2]; M._Ka[3] = m.Ka[3];
        M._Kd[0] = m.Kd[0]; M._Kd[1] = m.Kd[1]; M._Kd[2] = m.Kd[2]; M._Kd[3] = m.Kd[3];
        M._Ks[0] = m.Ks[0]; M._Ks[1] = m.Ks[1]; M._Ks[2] = m.Ks[2]; M._Ks[3] = m.Ks[3];
        M._Tf[0] = m.Tf[0]; M._Tf[1] = m.Tf[1]; M._Tf[2] = m.Tf[2];
        M._Ni = m.Ni;
        M._Ns = m.Ns;
        M._map_Ka = m.map_Ka;
        M._map_Kd = m.map_Kd;
        M._map_Ks = m.map_Ks;
        M._map_Bump = m.map_Bump;
        M._Bm = m.Bm;
        mesh._materials[i] = M;
    }

    mesh._groups.clear();
    mesh._groups.resize( _mesh->_groups.size() );
    for (unsigned i=0; i<_mesh->_groups.size(); ++i)
    {
        In::Group g = _mesh->_groups[i];
        Loader::Group G;
        G._start_face = g._start_face;
        G._end_face = g._end_face;
        //G.start_point = g.StartPoint;
        //G._end_point = g.EndPoint;
        G._name = g._name;
        G._assigned_mats.resize( g._assigned_materials.size() );
        for (unsigned j=0; j<g._assigned_materials.size(); ++j){
            In::Material_group mg = g._assigned_materials[j];
            Loader::Material_group MG;
            MG._start_face = mg._start_face;
            MG._end_face = mg ._end_face;
            //MG._start_point = mg.StartPoint;
            //MG._end_point = mg.EndPoint;
            MG._material_idx = mg._material_idx;
            G._assigned_mats[j] = MG;
        }
        mesh._groups[i] = G;
    }
}

//------------------------------------------------------------------------------

void Obj_file::set_mesh(const Loader::Abs_mesh& mesh)
{
    _mesh->_vertices.clear();
    _mesh->_vertices.resize( mesh._vertices.size() );
    for (unsigned i=0; i<mesh._vertices.size(); ++i)
    {
        Loader::Vertex v = mesh._vertices[i];
        _mesh->_vertices[i] = In::Vert( v.x, v.y, v.z );
    }

    _mesh->_normals.clear();
    _mesh->_normals.resize( mesh._normals.size() );
    for (unsigned i=0; i<mesh._normals.size(); ++i)
    {
        Loader::Normal n = mesh._normals[i];
        _mesh->_normals[i] = In::Normal( n.x, n.y, n.z );
    }

    _mesh->_texCoords.clear();
    _mesh->_texCoords.resize( mesh._texCoords.size() );
    for (unsigned i=0; i<mesh._texCoords.size(); ++i)
    {
        Loader::Tex_coord t = mesh._texCoords[i];
        _mesh->_texCoords[i] = In::Tex_coord( t.u, t.v );
    }

    _mesh->_triangles.clear();
    _mesh->_triangles.resize( mesh._triangles.size() );
    for (unsigned i=0; i<mesh._triangles.size(); ++i)
    {
        Loader::Tri_face f = mesh._triangles[i];
        In::Face F;
        F.v[0] = f.v[0]; F.v[1] = f.v[1]; F.v[2] = f.v[2];
        F.n[0] = f.n[0]; F.n[1] = f.n[1]; F.n[2] = f.n[2];
        F.t[0] = f.t[0]; F.t[1] = f.t[1]; F.t[2] = f.t[2];
        _mesh->_triangles[i] = F;
    }

    _mesh->_materials.clear();
    _mesh->_materials.resize( mesh._materials.size() );
    for (unsigned i=0; i<mesh._materials.size(); ++i)
    {
        Loader::Material m = mesh._materials[i];
        In::Material M;
        M.name = m._name;
        M.illum = m._illum;
        M.Ka[0] = m._Ka[0]; M.Ka[1] = m._Ka[1]; M.Ka[2] = m._Ka[2]; M.Ka[3] = m._Ka[3];
        M.Kd[0] = m._Kd[0]; M.Kd[1] = m._Kd[1]; M.Kd[2] = m._Kd[2]; M.Kd[3] = m._Kd[3];
        M.Ks[0] = m._Ks[0]; M.Ks[1] = m._Ks[1]; M.Ks[2] = m._Ks[2]; M.Ks[3] = m._Ks[3];
        M.Tf[0] = m._Tf[0]; M.Tf[1] = m._Tf[1]; M.Tf[2] = m._Tf[2];
        M.Ni = m._Ni;
        M.Ns = m._Ns;
        M.map_Ka = m._map_Ka;
        M.map_Kd = m._map_Kd;
        M.map_Ks = m._map_Ks;
        M.map_Bump = m._map_Bump;
        M.Bm = m._Bm;
        _mesh->_materials[i] = M;
    }

    _mesh->_groups.clear();
    _mesh->_groups.resize( mesh._groups.size() );
    for (unsigned i=0; i<mesh._groups.size(); ++i)
    {
        Loader::Group g = mesh._groups[i];
        In::Group G;
        G._start_face = g._start_face;
        G._end_face = g._end_face;
        //G.StartPoint = g.start_point;
        //G.EndPoint = g._end_point;
        G._name = g._name;
        G._assigned_materials.resize( g._assigned_mats.size() );
        for (unsigned j = 0; j < g._assigned_mats.size(); ++j){
            Loader::Material_group mg = g._assigned_mats[j];
            In::Material_group MG;
            MG._start_face = mg._start_face;
            MG._end_face = mg ._end_face;
            //MG.StartPoint = mg._start_point;
            //MG.EndPoint = mg._end_point;
            MG._material_idx = mg._material_idx;
            G._assigned_materials[j] = MG;
        }
        _mesh->_groups[i] = G;
    }
}

// END CLASS Obj_File ==========================================================


// =============================================================================
} // namespace Obj_Loader
// =============================================================================

