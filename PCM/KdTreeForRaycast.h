#pragma once
#include "basic_types.h"
#include "box.h"
#include "ray.h"
#include "shader.h"
#include <vector>
class TriangleType;
class Sample;
class PaintCanvas;
class KDNode
{
public:
	static int max_depth;
	KDNode(Sample& _smp);
	KDNode* build(std::vector<TriangleType*>& tris,int depth);
	Box& get_bounding_box();
	bool hit(KDNode* node, const Ray& ray, float& t, float& min, HitResult& hitResult);
	Box bbox;
	KDNode* left;
	KDNode* right;
	std::vector<TriangleType*> triangles_;
	Sample& smp_;

};
class KDTree
{
public:
	static Shader* openglShader;
	static int reference_count;
	static void loalshader(Shader*& shader, const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath = std::string());
	KDTree(Sample& _smp);
	~KDTree()
	{
		freeNode(root);
	}
	void freeNode(KDNode* node)
	{
		if (node)
		{
			KDNode* left = node->left;
			KDNode* right = node->right;
			delete node;
			freeNode(left);
			freeNode(right);
		}

	}
	void build();
	void updateViewOfMesh();
	void setupBuffer();
	void updateBuffer();
	void updateHitrayBuffer();
	void drawKdTree();
	bool hit(const Ray& ray, float& t, float& min, HitResult& hitResult);
	void clearDebugCube()
	{
		cubic_position_.clear();
		cubic_idx_.clear();
		isBufferSetup = false;
		element_size = 0;
	}
	void clearHitray()
	{
		hitray_position_.clear();
		hitray_idx_.clear();
		isHitrayBufferSetup = false;
	}
private:
	bool max_depth;
	bool isBuild;
	bool isBufferSetup;
	bool isHitrayBufferSetup;
	KDNode* root;
	Sample& smp_;
	PaintCanvas* canvas_;
	GLfloat p_viewmatrix_[16];
	GLfloat p_projmatrix_[16];
	GLfloat p_modelmatrix_[16];

	GLuint VBO;
	GLuint VBO2; //render hit ray
	GLuint VAO;
	GLuint VAO2;
	GLuint EBO;
	GLuint EBO2;
	std::vector<pcm::PointType> cubic_position_;
	std::vector<int>			cubic_idx_;
	std::vector<pcm::PointType> hitray_position_;
	std::vector<int>			hitray_idx_;
	GLsizei element_size;
	GLsizei* count;
	GLvoid** indices;

};
