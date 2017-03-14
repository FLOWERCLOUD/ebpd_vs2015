#ifndef MESH_MATERIALS_HPP__
#define MESH_MATERIALS_HPP__

#include <algorithm>
#include <string>

#include "mesh_types.hpp"
#include "toolbox/gl_utils/glassert.h"
#include "toolbox/gl_utils/gltex2D.hpp"

// =============================================================================
namespace EMesh {
	// =============================================================================

	struct Mat_grp{
		EMesh::Face_idx starting_idx; ///< starting face index to apply the material
		int nb_face;      ///< (starting_idx+nb_face) is the ending face index
		int mat_idx;      ///< material index to apply to the set of faces
	};

	// -------------------------------------------------------------------------

	struct Material{

		Material();

		void setup_opengl_materials();

		void set_ns(float v){ _ns = std::max( 0.f, std::min(v, 128.f)); }
		float get_ns() const { return _ns; }

		void set_ka(float r, float g, float b, float a) { _ka[0] = r; _ka[1] = g; _ka[2] = b; _ka[3] = a; }
		void set_kd(float r, float g, float b, float a) { _kd[0] = r; _kd[1] = g; _kd[2] = b; _kd[3] = a; }
		void set_ks(float r, float g, float b, float a) { _ks[0] = r; _ks[1] = g; _ks[2] = b; _ks[3] = a; }

		void set_tf(float t) { _tf[0] = _tf[1] = _tf[2] = t; }

		// TODO: use vec3 and vec4
		float _ka[4]; ///< ambient
		float _kd[4]; ///< diffuse
		float _ks[4]; ///< specular
		float _tf[3]; ///< transparency
		float _ni;    ///< intensity
	private:
		float _ns;    ///< specular power
	public:

		Tbx::GlTex2D* _map_ka;   ///< ambient texture map
		Tbx::GlTex2D* _map_kd;   ///< diffuse texture map
		Tbx::GlTex2D* _map_ks;   ///< specular texture map
		Tbx::GlTex2D* _map_bump; ///< bump texture map

		std::string _file_path_ka;
		std::string _file_path_kd;
		std::string _file_path_ks;
		std::string _file_path_bump;

		/// bump map depth. Only used if bump is relevent.
		float _bump_strength;
		std::string _name;
	};

}// End Namespace EMesh ========================================================

#endif // MESH_MATERIALS_HPP__
