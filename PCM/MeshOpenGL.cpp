#include "MeshOpenGL.h"
#include "sample.h"
#include "vertex.h"
#include "triangle.h"
#include "GlobalObject.h"
#include "paint_canvas.h"
#include "mesh.h"
#include <QGLViewer\manipulatedFrame.h>
#include <sstream>
#include <string>
using namespace std;
static bool isDebug = true;
namespace MyOpengl
{
	Shader*  MeshOpengl::openglShader = NULL;
	Shader*  MeshOpengl::normal_edge_Shader = NULL;
	int MeshOpengl::reference_count = 0;

	void MeshOpengl::loalshader(Shader*& shader ,const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath /*= std::string()*/)
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

	MeshOpengl::MeshOpengl(Sample& _smp ,const pcm::Mesh* _mesh):smp_(_smp),mesh_(_mesh)
	{
		reference_count++;
		isBufferSetup = false;
		isMeshSetup = false;
		if (!openglShader)
		{
			string shaderDir("./rendering/myshaders/");
			string vertex_shader_path("./rendering/myshaders/shader.vs");
			string frag_shader_path("./rendering/myshaders/shader.frag");
			loalshader(openglShader,vertex_shader_path, frag_shader_path);
		}else if (!normal_edge_Shader)
		{
			string shaderDir("./rendering/myshaders/");
			string vertex_shader_path("./rendering/myshaders/normal_edge_shader.vs");
			string frag_shader_path("./rendering/myshaders/normal_edge_shader.frag");
			string geo_shader_path("./rendering/myshaders/normal_edge_shader.gs");
			loalshader(normal_edge_Shader, vertex_shader_path, frag_shader_path, geo_shader_path);
		}
		canvas_ = Global_Canvas;

	}
	void MeshOpengl::draw(RenderMode::WhichColorMode mode , RenderMode::RenderType& r , Shader* _shader, PaintCanvas* _canvas)
	{
		switch (r)
		{
		case RenderMode::FlatMode:
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			break;
		}
		case RenderMode::WireMode:
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glLineWidth(0.01f);
			break;
		}
		case RenderMode::PointMode:
		{
			glEnable(GL_POINT_SMOOTH);
			glPointSize(20.0f);
			glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
			break;
		}
		defalut:
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}
		updateViewOfMesh();
		Shader* _curshader = NULL;
		if (_shader)
		{
			_curshader = _shader;
		}
		if (openglShader)
		{
			_curshader = _shader;
		}
		if (_curshader)
		{
			_curshader->Use();
			glUniformMatrix4fv(glGetUniformLocation(_curshader->Program, "projection"), 1, GL_FALSE, p_projmatrix_);
			glUniformMatrix4fv(glGetUniformLocation(_curshader->Program, "view"), 1, GL_FALSE, p_viewmatrix_);
			glUniformMatrix4fv(glGetUniformLocation(_curshader->Program, "model"), 1, GL_FALSE, p_modelmatrix_);

			GLint lightColorLoc = glGetUniformLocation(_curshader->Program, "lightColor");
			GLint lightPosLoc = glGetUniformLocation(_curshader->Program, "lightPos");
			GLint viewPosLoc = glGetUniformLocation(_curshader->Program, "viewPos");
			glUniform3f(lightColorLoc, 1.0f, 1.0f, 1.0f); //light color
			qglviewer::Vec camera_pos = canvas_->camera()->position();
			glUniform3f(lightPosLoc, camera_pos.x, camera_pos.y, camera_pos.z);   //light position
			glUniform3f(viewPosLoc, camera_pos.x, camera_pos.y, camera_pos.z); //camera position
			draw(*_curshader);
			_curshader->UnUse();
		}
	}
	void MeshOpengl::draw(Shader& _shader)
	{
		// Bind appropriate textures
		GLuint diffuseNr = 1;
		GLuint specularNr = 1;
		GLuint normalNr = 1;
		GLuint heightNr = 1;
		for (GLuint i = 0; i < this->textures.size(); i++)
		{
			glActiveTexture(GL_TEXTURE0 + i); // Active proper texture unit before binding
											  // Retrieve texture number (the N in diffuse_textureN)
			stringstream ss;
			string number;
			string name = this->textures[i].type;
			if (name == "texture_diffuse")
				ss << diffuseNr++; // Transfer GLuint to stream
			else if (name == "texture_specular")
				ss << specularNr++; // Transfer GLuint to stream
			else if (name == "texture_normal")
				ss << normalNr++; // Transfer GLuint to stream
			else if (name == "texture_ambient")
				ss << heightNr++; // Transfer GLuint to stream
			number = ss.str();
			//  using glUniform1i to set the location or texture-unit of the uniform samplers. 
			//By setting them via glUniform1i we make sure each uniform sampler corresponds to the proper texture unit.
			glUniform1i(glGetUniformLocation(_shader.Program, (name + number).c_str()), i);
			glUniform1i(glGetUniformLocation(_shader.Program, ("use_"+name + number).c_str()), 1);
			// And finally bind the texture to current texture unit
			glBindTexture(GL_TEXTURE_2D, this->textures[i].id);
		}

		// Draw mesh
		glBindVertexArray(this->VAO);
		if (this->indices.size())
			glDrawElements(GL_TRIANGLES, this->indices.size(), GL_UNSIGNED_INT, 0);
		else
			glDrawArrays(GL_POINTS, 0, this->vertices.size());
//		glDrawElements(GL_LINES, this->indices.size(), GL_UNSIGNED_INT, 0); 用线，点感觉都比面慢
//		glDrawElements(GL_POINTS, this->indices.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		// Always good practice to set everything back to defaults once configured.
		for (GLuint i = 0; i < this->textures.size(); i++)
		{
			glActiveTexture(GL_TEXTURE0 + i);
			glBindTexture(GL_TEXTURE_2D, 0);
		}


	}
	void MeshOpengl::drawNormal(Shader& _shader)
	{
		//draw normal 
		glBindVertexArray(this->VAO2);
		glDrawElements(GL_TRIANGLES, this->indices.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

	}
	void MeshOpengl::drawNormal(pcm::ColorType normalColor , Shader* _shader)
	{
		Shader* _curshader = NULL;
		if (_shader)
		{
			_curshader = _shader;
		}
		else if (_curshader)
		{
			_curshader = _shader;
		}
		updateViewOfMesh();
		if (_curshader)
		{
			_curshader->Use();
			glLineWidth(0.01f);
			glUniformMatrix4fv(glGetUniformLocation(_curshader->Program, "u_proj_mat"), 1, GL_FALSE, p_projmatrix_);
			glUniformMatrix4fv(glGetUniformLocation(_curshader->Program, "u_view_mat"), 1, GL_FALSE, p_viewmatrix_);
			glUniformMatrix4fv(glGetUniformLocation(_curshader->Program, "u_model_mat"), 1, GL_FALSE, p_modelmatrix_);

			GLint color_loc1 = glGetUniformLocation(_curshader->Program, "u_edgeColor");
			GLint color_loc2 = glGetUniformLocation(_curshader->Program, "u_faceNormalColor");
			GLint color_loc3 = glGetUniformLocation(_curshader->Program, "u_normalColor");
			GLint color_loc4 = glGetUniformLocation(_curshader->Program, "u_tangentColor");
			GLint color_loc5 = glGetUniformLocation(_curshader->Program, "u_bitangentColor");

			glUniform4f(color_loc1, 0.2f, 0.2f, 0.2f,1.0f); 
			glUniform4f(color_loc2, 0.0f, 1.0f, 0.2f, 1.0f);
			glUniform4f(color_loc3, 1.0f, 1.0f, 0.0f, 1.0f);
			glUniform4f(color_loc4, 0.05f, 0.15f, 0.646f, 1.0f);
			glUniform4f(color_loc5, 0.035f, 0.9f, 0.89f, 1.0f);
			// modifier for size of the normals, tangents and bitangents
			glUniform1f(glGetUniformLocation(_curshader->Program, "u_normalScale"), 0.025f);
			drawNormal(*_curshader);
			_curshader->UnUse();
		}


	}
	void MeshOpengl::setViewMatrix(GLfloat _viewmatrix[16])
	{
		for (size_t i = 0; i < 16; i++)
		{
			p_viewmatrix_[i] = _viewmatrix[i];
		}
	}
	void MeshOpengl::setProjMatrix(GLfloat _projmatrix[16])
	{
		for (size_t i = 0; i < 16; i++)
		{
			p_viewmatrix_[i] = _projmatrix[i];
		}
	}
	void MeshOpengl::setModelMatrix(GLfloat _modelmatrix[16])
	{
		for (size_t i = 0; i < 16; i++)
		{
			p_modelmatrix_[i] = _modelmatrix[i];
		}
	}

	void MeshOpengl::updateViewOfMesh()
	{
		canvas_->camera()->getProjectionMatrix(p_projmatrix_);
		canvas_->camera()->getModelViewMatrix(p_viewmatrix_);
		if (smp_.num_vertices())
		{
			qglviewer::Frame& frame = smp_.getFrame();
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
			pcm::Matrix44& mat = smp_.matrix_to_scene_coord();
			matirx44 = matirx44*mat;
			for (int i = 0; i < 4; ++i)
			{
				for (int j = 0; j < 4; ++j)
				{
					p_modelmatrix_[i + 4 * j] = matirx44(i, j);
				}
			}

		}
		else
		{


		}

	}
	void MeshOpengl::updateMesh()
	{
		canvas_->makeCurrent();
		if (!isMeshSetup)
		{
			setup();
			isMeshSetup = true;
		}
		else
		{
			loadMeshFromSample();
			if(isBufferSetup)
				updateBuffer();
			else
			{
				Logger << "buffer not setup error" << endl;
			}
		}
			


	}
	void MeshOpengl::updateBuffer()
	{
		if (!isBufferSetup)
			return;
		// Create buffers/arrays

//		glBindVertexArray(this->VAO);
		// Load data into vertex buffers
		glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
		// A great thing about structs is that their memory layout is sequential for all its items.
		// The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
		// again translates to 3/2 floats which translates to a byte array.
		if(this->vertices.size())
			glBufferData(GL_ARRAY_BUFFER, this->vertices.size() * sizeof(OpenglVertex), &this->vertices[0], GL_STREAM_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 1 * sizeof(OpenglVertex), 0, GL_STREAM_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
		if(this->indices.size())
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->indices.size() * sizeof(GLuint), &this->indices[0], GL_STREAM_DRAW);
		else
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, 1 * sizeof(GLuint), 0, GL_STREAM_DRAW);

		//// Set the vertex attribute pointers
		//// Vertex Positions
		//glEnableVertexAttribArray(0);
		//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)0);
		//// Vertex Normals
		//glEnableVertexAttribArray(1);
		//glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)offsetof(OpenglVertex, Normal));
		//// Vertex Texture Coords
		//glEnableVertexAttribArray(2);
		//glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)offsetof(OpenglVertex, TexCoords));
		//// Vertex Tangent
		//glEnableVertexAttribArray(3);
		//glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)offsetof(OpenglVertex, Tangent));
		//// Vertex Bitangent
		//glEnableVertexAttribArray(4);
		//glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)offsetof(OpenglVertex, Bitangent));

		// Load data into color buffers
		glBindBuffer(GL_ARRAY_BUFFER, this->CBO);
		glBufferData(GL_ARRAY_BUFFER, this->colors.size() * sizeof(OpenglColor), &this->colors[0], GL_STREAM_DRAW);
		//glEnableVertexAttribArray(5);
		//glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(OpenglColor), (GLvoid*)0);

		//glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		if (isDebug)
		{
			Logger << "updateBuffer" << endl;
		}

	}
	void MeshOpengl::setup()
	{
		setupMesh();
		setupBuffer();
	}
	void MeshOpengl::setupMesh()
	{
		if (!isMeshSetup)
		{
			if (smp_.num_vertices())
			{
				loadMeshFromSample();
				isMeshSetup = true;
			}			
			else if (mesh_)
			{
				loadMeshFrompcmMesh();
				isMeshSetup = true;
			}
				
			isMeshSetup = false;
		}


	}
	void MeshOpengl::setupBuffer()
	{
		if (!isBufferSetup)
		{
			// Create buffers/arrays
			glGenVertexArrays(1, &this->VAO);
			glGenBuffers(1, &this->VBO);
			glGenBuffers(1, &this->EBO);
			glGenBuffers(1, &this->CBO);//color buffer

			glBindVertexArray(this->VAO);
			// Load data into vertex buffers
			glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
			// A great thing about structs is that their memory layout is sequential for all its items.
			// The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
			// again translates to 3/2 floats which translates to a byte array.
			if(this->vertices.size())
				glBufferData(GL_ARRAY_BUFFER, this->vertices.size() * sizeof(OpenglVertex), &this->vertices[0], GL_STREAM_DRAW);
			else
				glBufferData(GL_ARRAY_BUFFER, 1 * sizeof(OpenglVertex), 0, GL_STREAM_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
			if(this->indices.size())
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->indices.size() * sizeof(GLuint), &this->indices[0], GL_STREAM_DRAW);
			else
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, 1 * sizeof(GLuint), 0, GL_STREAM_DRAW);
			// Set the vertex attribute pointers
			// Vertex Positions
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)0);
			// Vertex Normals
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)offsetof(OpenglVertex, Normal));
			// Vertex Texture Coords
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)offsetof(OpenglVertex, TexCoords));
			// Vertex Tangent
			glEnableVertexAttribArray(3);
			glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)offsetof(OpenglVertex, Tangent));
			// Vertex Bitangent
			glEnableVertexAttribArray(4);
			glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)offsetof(OpenglVertex, Bitangent));

			// Load data into color buffers
			glBindBuffer(GL_ARRAY_BUFFER, this->CBO);
			glBufferData(GL_ARRAY_BUFFER, this->colors.size() * sizeof(OpenglColor), &this->colors[0], GL_STREAM_DRAW);
			glEnableVertexAttribArray(5);
			glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(OpenglColor), (GLvoid*)0);

			glBindVertexArray(0);

//   VAO2 ,THIS IS FOR DRAW NORMAL OR EDGE
			glGenVertexArrays(1, &this->VAO2);
			glBindVertexArray(this->VAO2);
			glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

			// Set the vertex attribute pointers
			// Vertex Positions
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)0);
			// Vertex Normals
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)offsetof(OpenglVertex, Normal));

			// Vertex Tangent
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)offsetof(OpenglVertex, Tangent));
			// Vertex Bitangent
			glEnableVertexAttribArray(3);
			glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(OpenglVertex), (GLvoid*)offsetof(OpenglVertex, Bitangent));
			
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
			//// Load data into color buffers
			//glBindBuffer(GL_ARRAY_BUFFER, this->CBO);
			//glBufferData(GL_ARRAY_BUFFER, this->colors.size() * sizeof(OpenglColor), &this->colors[0], GL_STREAM_DRAW);
			//glEnableVertexAttribArray(4);
			//glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(OpenglColor), (GLvoid*)0);

			glBindVertexArray(0);


			isBufferSetup = true;
		}

	}
	void MeshOpengl::loadMeshFromSample()
	{
		
		vertices.clear();
		indices.clear();
		textures.clear();
		colors.clear();
		//vertices.resize(smp_.num_vertices());
		//indices.resize(smp_.num_triangles());
		for (int i = 0; i < smp_.num_vertices(); i++)
		{
			OpenglVertex vertex;
			vertex.Position  =  smp_[i].get_position();
			vertex.Normal    =  smp_[i].get_normal();
			vertex.TexCoords =  smp_[i].get_texture();
			vertex.Tangent   =  smp_[i].get_tangent();
			vertex.Bitangent =  smp_[i].get_bi_tangent();
			Vertex& v = smp_[i];
			vertices.push_back(vertex);

		}
		for (size_t i = 0; i < smp_.num_triangles(); i++)
		{
			TriangleType& triangle = smp_.getTriangle(i);
			indices.push_back(triangle.get_i_vertex(0));
			indices.push_back(triangle.get_i_vertex(1));
			indices.push_back(triangle.get_i_vertex(2));

		}
		for (size_t i = 0; i < smp_.num_vertices(); i++)
		{
			OpenglColor openglColor;
			openglColor.color = pcm::ColorType(smp_[i].r(), smp_[i].g(), smp_[i].b(), smp_[i].alpha());
			colors.push_back(openglColor);
		}
		if (isDebug)
		{
			Logger << "loadMeshFromSample" << endl;
		}
	}

	void  MeshOpengl::loadMeshFrompcmMesh()
	{
		vertices.clear();
		indices.clear();
		textures.clear();
		colors.clear();

		if (!mesh_)
			return;
		for (int i = 0; i < mesh_->num_vertices(); i++)
		{
			OpenglVertex vertex;
			vertex.Position = (*mesh_)[i].get_position();
			vertex.Normal = (*mesh_)[i].get_normal();
			vertex.TexCoords = (*mesh_)[i].get_texture();
			vertex.Tangent = (*mesh_)[i].get_tangent();
			vertex.Bitangent = (*mesh_)[i].get_bi_tangent();
			Vertex& v = (*mesh_)[i];
			vertices.push_back(vertex);

		}
		for (size_t i = 0; i < mesh_->num_triangles(); i++)
		{
			TriangleType& triangle = mesh_->getTriangle(i);
			indices.push_back(triangle.get_i_vertex(0));
			indices.push_back(triangle.get_i_vertex(1));
			indices.push_back(triangle.get_i_vertex(2));

		}
		for (size_t i = 0; i < mesh_->num_vertices(); i++)
		{
			OpenglColor openglColor;
			openglColor.color = pcm::ColorType((*mesh_)[i].r(), (*mesh_)[i].g(), (*mesh_)[i].b(), (*mesh_)[i].alpha());
			colors.push_back(openglColor);
		}
		if (isDebug)
		{
			Logger << "loadMeshFromSample" << endl;
		}

	}
	void MeshOpengl::updateColor()
	{
		canvas_->makeCurrent();
		colors.clear();
		if (smp_.num_vertices())
		{
			if (smp_.colors_.size() == smp_.num_vertices())
			{
				for (size_t i = 0; i < smp_.num_vertices(); i++)
				{
					OpenglColor openglColor;
					openglColor.color = smp_.colors_[i];
					colors.push_back(openglColor);
				}
			}
			else
			{
				for (size_t i = 0; i < smp_.num_vertices(); i++)
				{
					OpenglColor openglColor;
					openglColor.color = pcm::ColorType(smp_[i].r(), smp_[i].g(), smp_[i].b(), smp_[i].alpha());
					colors.push_back(openglColor);
				}

			}
		}
		else
		{
			if (mesh_->colors_.size() == mesh_->num_vertices())
			{
				for (size_t i = 0; i < mesh_ -> num_vertices(); i++)
				{
					OpenglColor openglColor;
					openglColor.color = mesh_ ->colors_[i];
					colors.push_back(openglColor);
				}
			}
			else
			{
				for (size_t i = 0; i < mesh_->num_vertices(); i++)
				{
					OpenglColor openglColor;
					openglColor.color = pcm::ColorType( (*mesh_)[i].r(), (*mesh_)[i].g(), (*mesh_)[i].b(), (*mesh_)[i].alpha());
					colors.push_back(openglColor);
				}

			}
		}


		glBindBuffer(GL_ARRAY_BUFFER, this->CBO);
		glBufferData(GL_ARRAY_BUFFER, this->colors.size() * sizeof(OpenglColor), &this->colors[0], GL_STREAM_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}
