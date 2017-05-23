#pragma once
#include <CustomGL\glew.h>
#include <string>
#include "basic_types.h"
#include "rendering/render_types.h"
#include "shader.h"

class Sample;
class PaintCanvas;
namespace pcm
{
	class Mesh;
}
namespace MyOpengl
{
	struct OpenglVertex {
		// Position
		pcm::PointType Position;
		// Normal
		pcm::NormalType Normal;
		// TexCoords
		pcm::TextureType TexCoords;
		// Tangent
		pcm::NormalType Tangent;
		// Bitangent
		pcm::NormalType Bitangent;
	};
	struct OpenglColor
	{
		pcm::ColorType color;

	};

	struct OpenglTexture {
		GLuint id;
		std::string type;
		std::string path;
	};

	/*
	this class is used for wrap the opengl buffer for the mesh data,
	make the mesh drawing faster
	*/


	class MeshOpengl
	{
	public:
		static Shader* openglShader;
		static Shader* normal_edge_Shader;
		static int reference_count;
		static void loalshader(Shader*& shader, const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath = std::string());
		MeshOpengl(Sample& _smp, const pcm::Mesh* _mesh = NULL);
		~MeshOpengl()
		{
			if (reference_count == 1)
			{
				delete openglShader;
				openglShader = NULL;
				delete normal_edge_Shader;
				normal_edge_Shader = NULL;
				reference_count--;
			}
			glDeleteBuffers(1, &VBO);
			glDeleteBuffers(1, &EBO);
			glDeleteBuffers(1, &VAO);
			glDeleteBuffers(1, &VAO2);
			glDeleteBuffers(1, &CBO);
		}
	private:
		void setViewMatrix(GLfloat _viewmatrix[16]);
		void setProjMatrix(GLfloat _projmatrix[16]);
		void setModelMatrix(GLfloat _modelmatrix[16]);
		void draw(Shader& _shader);
		void drawNormal(Shader& _shader);
	public:
		void draw(RenderMode::WhichColorMode mode, RenderMode::RenderType& r ,Shader* openglShader = NULL, PaintCanvas* canvas =NULL);
		void drawNormal(pcm::ColorType normalColor = pcm::ColorType(0.0f, 0.0f, 1.0f, 1.0f) , Shader* _shader =NULL);
		void updateMesh();
		void updateViewOfMesh();
		void updateColor();
	private:
		void updateBuffer();
		void setup();
		void setupMesh();
		void setupBuffer();
		void loadMeshFromSample();
		void loadMeshFrompcmMesh();

		bool isBufferSetup;
		bool isMeshSetup;
		Sample& smp_;
		const pcm::Mesh* mesh_;
		PaintCanvas* canvas_;
		GLfloat p_viewmatrix_[16];
		GLfloat p_projmatrix_[16];
		GLfloat p_modelmatrix_[16];
		/*  Render data  */
		GLuint VBO, EBO ,CBO;
		GLuint VAO;
		GLuint VAO2;
		/*  Mesh Data  */
		std::vector<OpenglVertex> vertices;
		std::vector<GLuint> indices;
		std::vector<OpenglTexture> textures;
		std::vector<OpenglColor> colors;
	};


	class MeshOpengl2
	{
	public:
		MeshOpengl2(pcm::Mesh& mesh);
		~MeshOpengl2()
		{

			glDeleteBuffers(1, &VBO);
			glDeleteBuffers(1, &EBO);
			glDeleteBuffers(1, &VAO);
			glDeleteBuffers(1, &VAO2);
			glDeleteBuffers(1, &CBO);
		}
		void updateMesh();
		void draw(RenderMode::WhichColorMode mode, RenderMode::RenderType& r, Shader* openglShader = NULL, PaintCanvas* canvas = NULL);
	private:
		void draw(Shader& _shader);
		void updateBuffer();
		void setup();
		void setupMesh();
		void setupBuffer();

		bool isBufferSetup;
		bool isMeshSetup;
		/*  Render data  */
		GLuint VBO, EBO, CBO;
		GLuint VAO;
		GLuint VAO2;
		/*  Mesh Data  */
		std::vector<OpenglVertex> vertices;
		std::vector<GLuint> indices;
		std::vector<OpenglTexture> textures;
		std::vector<OpenglColor> colors;


	};

}
