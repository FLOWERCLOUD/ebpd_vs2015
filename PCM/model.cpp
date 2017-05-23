#include "model.h"
#include "mesh.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "FreeImage.h"
#include "vertex.h"
#include "triangle.h"
using namespace  std;


namespace pcm
{
	void Model::processNode(aiNode* node, const aiScene* scene)
	{
		// Process each mesh located at the current node
		for (GLuint i = 0; i < node->mNumMeshes; i++)
		{
			// The node object only contains indices to index the actual objects in the scene. 
			// The scene contains all the data, node is just to keep stuff organized (like relations between nodes).
			aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
			this->meshes.push_back(this->processMesh(mesh, scene));
		}
		// After we've processed all of the meshes (if any) we then recursively process each of the children nodes
		for (GLuint i = 0; i < node->mNumChildren; i++)
		{
			this->processNode(node->mChildren[i], scene);
		}

	}

	std::vector<Texture*> loadMaterialTextures(std::vector<Texture*> textures_loaded, string directory, aiMaterial* mat, aiTextureType type, std::string typeName);
	GLint TextureFromFile(std::string path, std::string directory, bool gamma = false);
	Mesh* Model::processMesh(aiMesh* _mesh, const aiScene* scene)
	{
		vector<Vertex*> vertices;
		vector<TriangleType*> indices;
		vector<Texture*> textures;

		// Walk through each of the mesh's vertices
		for (GLuint i = 0; i < _mesh->mNumVertices; i++)
		{
			Vertex* vertex = new Vertex(i);
			pcm::PointType vector; // We declare a placeholder vector since assimp uses its own vector class that doesn't directly convert to glm's vec3 class so we transfer the data to this placeholder glm::vec3 first.
								   // Positions
			vector(0) = _mesh->mVertices[i].x;
			vector(1) = _mesh->mVertices[i].y;
			vector(2) = _mesh->mVertices[i].z;
			vertex->set_position(vector);
			// Normals
			vector(0) = _mesh->mNormals[i].x;
			vector(1) = _mesh->mNormals[i].y;
			vector(2) = _mesh->mNormals[i].z;
			vertex->set_normal(vector);
			// Texture Coordinates
			if (_mesh->mTextureCoords[0]) // Does the mesh contain texture coordinates?
			{
				pcm::TextureType vec;
				// A vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't 
				// use models where a vertex can have multiple texture coordinates so we always take the first set (0).
				vec(0) = _mesh->mTextureCoords[0][i].x;
				vec(1) = _mesh->mTextureCoords[0][i].y;
				vertex->set_texture(vec);
			}
			else
				vertex->set_texture(pcm::TextureType(0.0f, 0.0f));
			// Tangent
			vector(0) = _mesh->mTangents[i].x;
			vector(1) = _mesh->mTangents[i].y;
			vector(2) = _mesh->mTangents[i].z;
			vertex->set_tangent(vector);
			// Bitangent
			vector(0) = _mesh->mBitangents[i].x;
			vector(1) = _mesh->mBitangents[i].y;
			vector(2) = _mesh->mBitangents[i].z;
			vertex->set_bi_tangent(vector);
			vertices.push_back(vertex);
		}
		// Now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
		for (GLuint i = 0; i < _mesh->mNumFaces; i++)
		{
			aiFace face = _mesh->mFaces[i];
			// Retrieve all indices of the face and store them in the indices vector
			if (3 == face.mNumIndices)
			{
				TriangleType* triangle = new TriangleType(sample_, i);

				for (GLuint j = 0; j < face.mNumIndices; j++)
					triangle->set_i_vetex(j, face.mIndices[j]);
				indices.push_back(triangle);
			}

		}
		// Process materials
		if (_mesh->mMaterialIndex >= 0)
		{
			aiMaterial* material = scene->mMaterials[_mesh->mMaterialIndex];
			// We assume a convention for sampler names in the shaders. Each diffuse texture should be named
			// as 'texture_diffuseN' where N is a sequential number ranging from 1 to MAX_SAMPLER_NUMBER. 
			// Same applies to other texture as the following list summarizes:
			// Diffuse: texture_diffuseN
			// Specular: texture_specularN
			// Normal: texture_normalN

			// 1. Diffuse maps
			vector<Texture*> diffuseMaps = loadMaterialTextures(textures_loaded, directory, material, aiTextureType_DIFFUSE, "texture_diffuse");
			textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
			// 2. Specular maps
			vector<Texture*> specularMaps = loadMaterialTextures(textures_loaded, directory, material, aiTextureType_SPECULAR, "texture_specular");
			textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
			// 3. Normal maps
			std::vector<Texture*> normalMaps = loadMaterialTextures(textures_loaded, directory, material, aiTextureType_NORMALS, "texture_normal");
			textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());
			// 4. ambient maps
			std::vector<Texture*> ambientMaps = loadMaterialTextures(textures_loaded, directory, material, aiTextureType_AMBIENT, "texture_ambient");
			textures.insert(textures.end(), ambientMaps.begin(), ambientMaps.end());
	
		}


		Mesh* mesh = new Mesh(vertices, indices, textures);
		return mesh;

	}

	void Model::draw(RenderMode::WhichColorMode mode, RenderMode::RenderType& r, Shader* openglShader, PaintCanvas* canvas)
	{
		for (size_t i = 0; i < meshes.size(); i++)
		{
			meshes[i]->draw(mode, r, openglShader, canvas);
		}

	}

	// Checks all material textures of a given type and loads the textures if they're not loaded yet.
	// The required info is returned as a Texture struct.
	vector<Texture*> loadMaterialTextures(std::vector<Texture*> textures_loaded, string directory, aiMaterial* mat, aiTextureType type, std::string typeName)
	{
		vector<Texture*> textures;
		for (GLuint i = 0; i < mat->GetTextureCount(type); i++)
		{
			aiString str;
			mat->GetTexture(type, i, &str);
			// Check if texture was loaded before and if so, continue to next iteration: skip loading a new texture
			GLboolean skip = false;
			for (GLuint j = 0; j < textures_loaded.size(); j++)
			{
				if (std::strcmp(textures_loaded[j]->path.c_str(), str.C_Str()) == 0)
				{
					textures.push_back(textures_loaded[j]);
					skip = true; // A texture with the same filepath has already been loaded, continue to next one. (optimization)
					break;
				}
			}
			if (!skip)
			{   // If texture hasn't been loaded already, load it
				Texture*  texture = new Texture();
				texture->id = TextureFromFile(string(str.C_Str()), directory);
				texture->type = typeName;
				texture->path = string(str.C_Str());
				textures.push_back(texture);
				textures_loaded.push_back(texture);  // Store it as texture loaded for entire model, to ensure we won't unnecesery load duplicate textures.
			}
		}
		return textures;
	}

	GLint TextureFromFile(string filename, string directory, bool gamma)
	{
		//Generate texture ID and load texture data 
		filename = directory + '/' + filename;
		GLuint textureID;
		glGenTextures(1, &textureID);

		//image format
		FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
		//pointer to the image, once loaded
		FIBITMAP *dib(0);
		//pointer to the image data
		BYTE* bits(0);
		//image width and height
		unsigned int width(0), height(0);

		//check the file signature and deduce its format
		fif = FreeImage_GetFileType(filename.c_str(), 0);
		//if still unknown, try to guess the file format from the file extension
		if (fif == FIF_UNKNOWN)
			fif = FreeImage_GetFIFFromFilename(filename.c_str());
		//if still unkown, return failure
		if (fif == FIF_UNKNOWN)
			return false;
		//check that the plugin has reading capabilities and load the file
		if (FreeImage_FIFSupportsReading(fif))
			dib = FreeImage_Load(fif, filename.c_str());
		//if the image failed to load, return failure
		if (!dib)
			return false;
		//retrieve the image data
		bits = FreeImage_GetBits(dib);
		//get the image width and height
		width = FreeImage_GetWidth(dib);
		height = FreeImage_GetHeight(dib);

		//if this somehow one of these failed (they shouldn't), return failure
		if ((bits == 0) || (width == 0) || (height == 0))
			return false;

		//	unsigned char* image;// = SOIL_load_image(filename.c_str(), &width, &height, 0, SOIL_LOAD_RGB);
		// Assign texture to ID
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, gamma ? GL_SRGB : GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, bits);
		glGenerateMipmap(GL_TEXTURE_2D);

		// Parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
		//	SOIL_free_image_data(image);
		FreeImage_Unload(dib);
		return textureID;
	}

}
