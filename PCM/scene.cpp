#include "scene.h"
#include "model.h"
#include "shader.h"
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "GlobalObject.h"
#include "paint_canvas.h"

namespace pcm
{
	Shader*  Scene::openglShader = NULL;
	Shader*  Scene::normal_edge_Shader = NULL;
	int Scene::reference_count = 0;
	void Scene::loalshader(Shader*& shader, const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath /*= std::string()*/)
	{
		//string shaderDir("./rendering/myshaders/");
		//string vertex_shader_path = shaderDir + "shader.vs";
		//string frag_shader_path = shaderDir + "shader.frag";
		if (!shader)
		{
			if (geometryPath.size())
			{
				shader = new Shader(vertexPath.c_str(), fragmentPath.c_str(), geometryPath.c_str());
			}
			else
			{
				shader = new Shader(vertexPath.c_str(), fragmentPath.c_str(), nullptr);
			}


		}

		else
		{

		}
	}

	Scene::Scene(Sample& _sample, std::string path) :sample_(_sample)
	{
		this->loadScene(path);
	}
	void Scene::clear()
	{
		for (int i = 0; i < models_.size(); ++i)
		{
			delete models_[i];
		}
		models_.clear();
	}

	void Scene::draw(RenderMode::WhichColorMode mode, RenderMode::RenderType& r)
	{
		for (size_t i = 0; i < models_.size(); i++)
		{
			models_[i]->draw(mode, r, openglShader , canvas_);
		}

	}

	/*
	the code is referenced to https://learnopengl.com/#!Model-Loading/Model
	*/
	void  Scene::loadScene(std::string path)
	{
		clear();
		addToScene(path);
	}

	void Scene::addToScene(std::string path)
	{

//	Assimp::Importer importer;  //
//		const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);//using this has link error ,don't kown why
		const aiScene* scene = aiImportFile(path.c_str(), aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);

		this->directory = path.substr(0, path.find_last_of('/'));

		Model* model = this->processNode(scene->mRootNode, scene);
		models_.push_back(model);
	}


	Model*  Scene::processNode(aiNode* node, const aiScene* scene)
	{
		Model* model = new Model(sample_, directory);
		model->processNode(node, scene);
		return model;
	}
	void Scene::updateView()
	{
		canvas_->camera()->getProjectionMatrix(p_projmatrix_);
		canvas_->camera()->getModelViewMatrix(p_viewmatrix_);
		qglviewer::Frame& frame = sample_.getFrame();
		double local_matrix[16];
		frame.getMatrix(local_matrix);
		pcm::Matrix44 matirx44;
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				matirx44(i, j) = local_matrix[i + 4 * j];
			}

		}
		pcm::Matrix44& mat = sample_.matrix_to_scene_coord();
		matirx44 = matirx44*mat;
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				p_modelmatrix_[i + 4 * j] = matirx44(i, j);
			}
		}
	}
}

