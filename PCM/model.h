#pragma once

#include "basic_types.h"
#include "rendering/render_types.h"
#include <string>
#include <vector>
class aiNode;
class aiScene;
class aiMesh;
class aiMaterial;
class Sample;
class Shader;
class PaintCanvas;
namespace pcm
{
	class Mesh;
}
class Texture;

namespace pcm
{
	class Model
	{
	public:
		Model(Sample& _sample) :sample_(_sample)
		{

		}
		Model(Sample& _sample, std::string _directory) :sample_(_sample), directory(_directory)
		{

		}
		~Model()
		{
			clear();
		}
		void clear()
		{
			for (int i = 0; i < meshes.size(); ++i)
			{
				delete meshes[i];
			}
			meshes.clear();
			for (int i = 0; i < textures_loaded.size(); ++i)
			{
				delete textures_loaded[i];
			}
			textures_loaded.clear();
		}
		void processNode(aiNode* node, const aiScene* scene);
		Mesh* processMesh(aiMesh* mesh, const aiScene* scene);
		void draw(RenderMode::WhichColorMode mode, RenderMode::RenderType& r, Shader* openglShader, PaintCanvas* canvas);
	private:


		std::vector<Mesh*> meshes;
		std::vector<Texture*> textures_loaded;
		Sample& sample_;
		std::string directory;
	};

}
