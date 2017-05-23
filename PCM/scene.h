#pragma once
#include "basic_types.h"
#include "rendering/render_types.h"
#include <string>
#include <vector>


class aiNode;
class aiScene;
class aiMesh;
class Shader;
class Sample;
class PaintCanvas;
namespace pcm
{
	class Model;

	class Scene
	{
	public:
		static Shader* openglShader;
		static Shader* normal_edge_Shader;
		static int reference_count;
 		static void loalshader(Shader*& shader, const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath = std::string());
		Scene(Sample& _sample, std::string path);
		~Scene()
		{
			clear();
		}
		void clear();
		void draw(RenderMode::WhichColorMode mode, RenderMode::RenderType& r);
	private:
		void updateView();
		void loadScene(std::string path);
		void addToScene(std::string path);
		Model* processNode(aiNode* node, const aiScene* scene);
		GLfloat p_viewmatrix_[16];
		GLfloat p_projmatrix_[16];
		GLfloat p_modelmatrix_[16];
		std::string directory;
		std::vector<Model*> models_;
		Sample& sample_;
		PaintCanvas* canvas_;

	};
}

