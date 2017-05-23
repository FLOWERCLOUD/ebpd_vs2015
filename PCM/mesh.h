#pragma once
#include "basic_types.h"
#include "rendering/render_types.h"
#include "QGLViewer/frame.h"
#include <vector>
#include <string>
class Vertex;
class TriangleType;
class Shader;
class PaintCanvas;
namespace MyOpengl
{
	class MeshOpengl;
}
struct Texture {
	GLuint id;
	std::string type;
	std::string path;
};
namespace pcm
{
	class Mesh
	{
	public:
		Mesh() :opengl_mesh_(NULL)
		{

		}
		Mesh(std::vector<Vertex*>&	_vertices, std::vector<TriangleType*>&  _triangle_array, std::vector<Texture*>& _textures)
		{
			vertices_ = _vertices;
			triangle_array_ = _triangle_array;
			textures_ = _textures;
			opengl_mesh_ = NULL;
		}
		~Mesh()
		{
			clear();
		}
		void clear()
		{
			for (int i = 0; i < vertices_.size(); ++i)
			{
				delete vertices_[i];
			}
			vertices_.clear();
			for (int i = 0; i < triangle_array_.size(); ++i)
			{
				delete triangle_array_[i];
			}
			triangle_array_.clear();
			for (int i = 0; i < textures_.size(); ++i)
			{
				delete textures_[i];
			}
			textures_.clear();
			if (opengl_mesh_)
				delete opengl_mesh_;
			opengl_mesh_ = NULL;
		}

		inline Vertex& operator[](IndexType i) const
		{
			return *vertices_[i];
		}
		inline TriangleType& getTriangle(IndexType i) const
		{
			return *triangle_array_[i];
		}


		void setupOpenglMesh();
		void draw(RenderMode::WhichColorMode mode, RenderMode::RenderType& r, Shader* openglShader,PaintCanvas* canvas);
		size_t num_vertices() const { return vertices_.size(); }
		size_t num_triangles() const { return triangle_array_.size(); }
		size_t num_textures() const { return textures_.size(); }
		inline qglviewer::Frame& getFrame()
		{
			return m_frame;
		}
		std::vector<pcm::ColorType> colors_;
	
	private:
		qglviewer::Frame m_frame;
		std::vector<Vertex*>	vertices_;
		std::vector<TriangleType*>  triangle_array_;
		std::vector<Texture*> textures_;
		MyOpengl::MeshOpengl* opengl_mesh_;
	};
}
