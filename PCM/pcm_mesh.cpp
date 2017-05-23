#include "mesh.h"
#include "MeshOpenGL.h"
#include "shader.h"
#include "QGLViewer\vec.h"
#include "QGLViewer\camera.h"
#include <sstream>
using namespace std;
namespace pcm
{
	void Mesh::setupOpenglMesh()
	{
		opengl_mesh_->updateMesh();
	}
	void Mesh::draw(RenderMode::WhichColorMode mode, RenderMode::RenderType& r , Shader* openglShader, PaintCanvas* canvas)
	{
		if (opengl_mesh_)
			opengl_mesh_->draw(mode, r, openglShader, canvas);
	}

}
