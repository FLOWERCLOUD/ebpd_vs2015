#include "mesh_static.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

// -----------------------------------------------------------------------------

void Mesh_static::clear_data()
{
	_tris.clear();
	_quads.clear();
	_verts.clear();
}

// -----------------------------------------------------------------------------

void Mesh_static::import_off(const char* filename)
{
	//!!!!!!!!!!
	// TODO: off loader to be deleted -> use the general interface #Loader with abstract files
	//!!!!!!!!!!
	using namespace std;
	int nil;
	ifstream file(filename);
	int nb_faces;

	if(!file.is_open()){
		cout << "error loading file : " << filename << endl;
		exit(1);
	}
	string line;
	file >> line;
	if(line.compare("OFF\n") == 0){
		cerr << "this is not an OFF file\n";
		return;
	}

	int nb_verts;
	file >> nb_verts >> nb_faces >> nil;

	_verts.resize( nb_verts );

	std::vector<int> tri_index(nb_faces * 3 * 2);
	std::vector<int> quad_index(nb_faces * 4);

	for(int i = 0; i < nb_verts;i++)
	{
		Tbx::Vec3 p;
		file >> p.x >> p.y >> p.z;
		set_vertex(i, p);
	}

	int nb_tri = 0;
	int nb_quad = 0;
	int k = 0, k_quad = 0, max_edges_per_face = 8;
	for(int i = 0; i < nb_faces;i++)
	{
		int face_edges;
		file >> face_edges;
		if(face_edges == 3)
		{
			for(int j = 0; j < 3; j++){
				int idx;
				file >> idx;
				tri_index[k++] = idx;
			}
			nb_tri++;
		}
		if(face_edges == 4)
		{
			for(int j = 0; j < 4; j++){
				int idx;
				file >> idx;
				quad_index[k_quad++] =idx;
			}
			nb_quad++;
		}
		if(face_edges > 4)
		{
			int* v_face = new int[max_edges_per_face];
			cout << "large face: " << face_edges << "at " << nb_tri << "\n";
			if(face_edges > max_edges_per_face){
				delete[] v_face;
				max_edges_per_face = face_edges;
				v_face = new int[max_edges_per_face];
			}
			for(int i = 0; i < face_edges; i++){
				file >> v_face[i];
			}
			int a = 0;
			int b = 1;
			int c = face_edges - 1;
			int d = face_edges - 2;
			for(int i = 0; i < face_edges - 2; i += 2){
				int v0 = v_face[a], v1 = v_face[b];
				int v2 = v_face[c], v3 = v_face[d];
				tri_index[k++] = v0; tri_index[k++] = v1; tri_index[k++] = v2;
				nb_tri++;
				if(i < face_edges - 3)
				{
					tri_index[k++] = v3; tri_index[k++] = v2; tri_index[k++] = v1;
					nb_tri++;
				}
				a++; b++;
				c--; d--;
			}
			delete[] v_face;
		}
	}
	file.close();


	if( nb_tri > 0)
	{
		_tris.resize( nb_tri );
		for(int i = 0; i < nb_tri; i++)
		{
			for(int u = 0; u < 3; u++)
				_tris[i][u] = tri_index[i*3 + u];
		}
	}

	if( nb_quad > 0)
	{
		_quads.resize( nb_quad );
		for(int i = 0; i < nb_quad; i++)
			for(int u = 0; u < 3; u++)
				_quads[i][u] = quad_index[i*3 + u];
	}
}

// -----------------------------------------------------------------------------

void Mesh_static::export_off(const char* filename,
							 bool invert_index,
							 float scale,
							 const Tbx::Vec3& offset) const
{
	// TODO: to be deleted use the general interface with abstract files
	using namespace std;
	ofstream file(filename);
	if( !file.is_open() ){
		cerr << "Error creating file: " << filename << endl;
		exit(1);
	}
	file << "OFF" << endl;
	file << get_nb_vertices() << ' ' << get_nb_faces() << " 0" << endl;

	for(int i = 0; i < get_nb_vertices(); i++)
	{
		file << (get_vertex(i).x * 1.f / scale - offset.x) << ' '
			<< (get_vertex(i).y * 1.f / scale - offset.y) << ' '
			<< (get_vertex(i).z * 1.f / scale - offset.z) << ' ' << endl;
	}

	for(int i = 0; i < get_nb_tris(); i++)
	{
		EMesh::Tri_face tri = _tris[i];
		if(invert_index)
			file << "3 " << tri.c << ' ' << tri.b << ' ' << tri.a << endl;
		else
			file << "3 " << tri.a << ' ' << tri.b << ' ' << tri.c << endl;
	}

	for(int i = 0; i < get_nb_quads(); i++)
	{
		EMesh::Quad_face quad = _quads[i];
		if(invert_index)
			file << "4 " << quad.d << ' ' << quad.c << ' ' << quad.b << ' ' << quad.a << endl;
		else
			file << "4 " << quad.a << ' ' << quad.b << ' ' << quad.c << ' ' << quad.d << endl;
	}
	file.close();
}


