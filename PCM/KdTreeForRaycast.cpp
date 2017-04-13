#include "KdTreeForRaycast.h"
#include "sample.h"
#include "vertex.h"
#include "triangle.h"
#include "GlobalObject.h"
#include "paint_canvas.h"
#include <fstream>
using namespace std;
int KDNode::max_depth = 5 ;
KDNode::KDNode(Sample& _smp):smp_(_smp)
{



}

KDNode* KDNode::build(std::vector<TriangleType*>& tris, int depth)
{
	KDNode* node = new KDNode(smp_);
	node->triangles_ = tris;
	node->left = NULL;
	node->right = NULL;
	node->bbox = Box();
	if (tris.size()==0)
		return node;
	if (tris.size() == 1)
	{
		node->bbox = tris[0]->get_bounding_box();
		node->left = new KDNode(smp_);
		node->right = new KDNode(smp_);
		node->left->triangles_ = vector<TriangleType*>();
		node->right->triangles_ = vector<TriangleType*>();
		return node;
	}
	pcm::Vec3 midpt = pcm::Vec3(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < tris.size(); ++i)
	{
		TriangleType* p_tri = tris[i];
		int i0 = p_tri->get_i_vertex(0);
		int i1 = p_tri->get_i_vertex(1);
		int i2 = p_tri->get_i_vertex(2);
		pcm::PointType& p1 = smp_[i0].get_position();
		pcm::PointType& p2 = smp_[i1].get_position();
		pcm::PointType& p3 = smp_[i2].get_position();
		node->bbox.expand(p1);
		node->bbox.expand(p2);
		node->bbox.expand(p3);
		midpt += p_tri->get_midpoint();
	}
	if (depth >= max_depth)
		return node;
	midpt /= tris.size();
	vector<TriangleType*> left_tris;
	vector<TriangleType*> right_tris;
	int axis = node->bbox.longest_axis();
	for (int i = 0; i < tris.size(); ++i)
	{
		switch (axis)
		{
		case 0:
			midpt.x() >= tris[i]->get_midpoint().x() ? right_tris.push_back(tris[i]) :
				left_tris.push_back(tris[i]);
			break;
		case 1:
			midpt.y() >= tris[i]->get_midpoint().y() ? right_tris.push_back(tris[i]) :
				left_tris.push_back(tris[i]);
			break;
		case 2:
			midpt.z() >= tris[i]->get_midpoint().z() ? right_tris.push_back(tris[i]) :
				left_tris.push_back(tris[i]);
			break;
		}
	}
	if (left_tris.size() == 0 && right_tris.size() > 0) left_tris = right_tris;
	if (right_tris.size() == 0 && left_tris.size() > 0) right_tris = left_tris ;
	int matches = 0;
	for (int i = 0; i < left_tris.size(); ++i)
	{
		for (int j = 0; j < right_tris.size(); ++j)
		{
			if (left_tris[i] == right_tris[j])
				matches++;

		}
	}
	if ((float)matches / left_tris.size() < 0.5 && (float)matches / right_tris.size() < 0.5)
	{
		node->left = build(left_tris, depth + 1);
		node->right = build(right_tris, depth + 1);
	}
	else
	{//stop subdivision
		node->left = new KDNode(smp_);
		node->right = new KDNode(smp_);
		node->left->triangles_ = vector<TriangleType*>();
		node->right->triangles_ = vector<TriangleType*>();

	}
	return node;

}
Box& KDNode::get_bounding_box()
{

	return bbox;
}

bool KDNode::hit(KDNode* node, const Ray& ray, float& t, float& min, HitResult& hitResult)
{
	if (node->bbox.hit(ray))
	{
		pcm::NormalType normal;
		bool hit_tris = false;

		//if either child still has triangles ,recurse down both sides and check for intersection
		bool isheat = false;
		if (node->left &&
			node->left->triangles_.size() > 0 || node->right &&node->right->triangles_.size() > 0)
		{

				bool hitleft = hit(node->left, ray, t, min, hitResult);
				bool hitright = hit(node->right, ray, t, min, hitResult);
				return hitleft || hitright;
		}
		else
		{
			//we have reched a leaf
			float min_t = 10000.0f;
			pcm::PointType hit_pt, local_hit_pt;
			HitResult besthit;
			for (int i = 0; i < node->triangles_.size(); ++i)
			{
				HitResult triangle_hitresult;
				if (node->triangles_[i]->hit(ray, t, min, triangle_hitresult))
				{
					hit_tris = true;
					if( t <min_t)
						besthit = triangle_hitresult;

				}
			}
			if (hit_tris)
			{
				hitResult = besthit;
				hitResult.hit_obj = true;
				return true;
			}
			return false;


		}

	}
	return false;
}



Shader*  KDTree::openglShader = NULL;
int		 KDTree::reference_count = 0;
void KDTree::loalshader(Shader*& shader, const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath /*= std::string()*/)
{
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
KDTree::KDTree(Sample& _smp):smp_(_smp), isBuild(false),isBufferSetup(false),root(NULL),
 count(NULL),indices(NULL), isHitrayBufferSetup(false)
{
	reference_count++;
	if (!openglShader)
	{
		string shaderDir("./rendering/myshaders/");
		string vertex_shader_path("./rendering/myshaders/basic_shader.vs");
		string frag_shader_path("./rendering/myshaders/basic_shader.frag");
		loalshader(openglShader, vertex_shader_path, frag_shader_path);
	}
	canvas_ = Global_Canvas;
}

void buildDebugCub(KDNode* node, std::vector<pcm::PointType>& cubic_position_,
std::vector<int>&	cubic_idx_)
{
	if (!node)
		return;
	if (!node->triangles_.size())
		return;
	Box& box = node->get_bounding_box();
	int vtx_size = cubic_position_.size();
	const pcm::PointType& high = box.high_corner();
	const pcm::PointType& low = box.low_corner();
	cubic_position_.push_back(low);
	cubic_position_.push_back(pcm::PointType(high(0),low(1),low(2)));
	cubic_position_.push_back(pcm::PointType(high(0), high(1), low(2)));
	cubic_position_.push_back(pcm::PointType(low(0), high(1), low(2)));
	
	cubic_position_.push_back(pcm::PointType(low(0), low(1), high(2)));
	cubic_position_.push_back(pcm::PointType(high(0), low(1), high(2)));
	cubic_position_.push_back(high);
	cubic_position_.push_back(pcm::PointType(low(0), high(1), high(2)));

	cubic_idx_.push_back(vtx_size + 0);
	cubic_idx_.push_back(vtx_size + 1);
	cubic_idx_.push_back(vtx_size + 2);
	cubic_idx_.push_back(vtx_size + 3);
	cubic_idx_.push_back(vtx_size + 4);
	cubic_idx_.push_back(vtx_size + 5);
	cubic_idx_.push_back(vtx_size + 6);
	cubic_idx_.push_back(vtx_size + 7);
	cubic_idx_.push_back(vtx_size + 0);
	cubic_idx_.push_back(vtx_size + 4);
	cubic_idx_.push_back(vtx_size + 5);
	cubic_idx_.push_back(vtx_size + 1);
	cubic_idx_.push_back(vtx_size + 2);
	cubic_idx_.push_back(vtx_size + 3);
	cubic_idx_.push_back(vtx_size + 7);
	cubic_idx_.push_back(vtx_size + 6);
	buildDebugCub(node->left, cubic_position_, cubic_idx_);
	buildDebugCub(node->right, cubic_position_, cubic_idx_);

}
void KDTree::build()
{
	KDNode::max_depth = 7;
	vector<TriangleType*>& triangles = smp_.getTriangleArray();
	KDNode node(smp_);
	root = node.build(triangles, 0);
	buildDebugCub(root, cubic_position_, cubic_idx_);
	element_size = this->cubic_idx_.size() / 4;
	if (count)
		delete[] count;
	count = new GLsizei[element_size];
	for (size_t i = 0; i < element_size; i++)
	{
		count[i] = 4;
	}
	if (indices)
		delete[] indices;
	indices = new GLvoid*[element_size];

	for (size_t i = 0; i < element_size; i++)
	{
		indices[i] = (GLuint*)(0)+4*i;
	}
	
}
void KDTree::updateViewOfMesh()
{
	canvas_->camera()->getProjectionMatrix(p_projmatrix_);
	canvas_->camera()->getModelViewMatrix(p_viewmatrix_);
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

void KDTree::setupBuffer()
{
	if (!isBuild)
	{
		build();
		isBuild = true;
	}
	if (!isBufferSetup)
	{
		// Create buffers/arrays
		glGenVertexArrays(1, &this->VAO);
		glGenBuffers(1, &this->VBO);
		glGenBuffers(1, &this->EBO);
		glBindVertexArray(this->VAO);
		glBindBuffer(GL_ARRAY_BUFFER, this->VBO);       
		if(this->cubic_position_.size())
			glBufferData(GL_ARRAY_BUFFER, this->cubic_position_ .size()* sizeof(pcm::PointType), &this->cubic_position_[0], GL_STREAM_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 1 * sizeof(pcm::PointType), 0, GL_STREAM_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
		if(this->cubic_idx_.size())
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->cubic_idx_.size() * sizeof(GLuint), &this->cubic_idx_[0], GL_STREAM_DRAW);
		else
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, 1 * sizeof(GLuint), 0, GL_STREAM_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(pcm::PointType), (GLvoid*)0);
		glBindVertexArray(0);
		isBufferSetup = true; 
		//ofstream writer1("./vbo.txt");
		//writer1 << cubic_position_.size() << std::endl;
		//for (size_t i = 0; i < cubic_position_.size(); i++)
		//{
		//	writer1 << cubic_position_[i](0) <<" "<< cubic_position_[i](1)<<" "<< cubic_position_[i](2)<< std::endl;
		//}
		//
		//writer1.close();
		//ofstream writer2("./idx.txt");
		//writer2 << cubic_idx_.size() << std::endl;
		//for (size_t i = 0; i < cubic_idx_.size(); i+=4)
		//{
		//	writer2 << cubic_idx_[i] << " " << cubic_idx_[i+1] << " " << cubic_idx_[i+2] <<" "<< cubic_idx_[i + 3]<< std::endl;
		//}
		//writer2.close();
	}
	if (!isHitrayBufferSetup)
	{

			glGenVertexArrays(1, &this->VAO2);
			glGenBuffers(1, &this->VBO2);
			glGenBuffers(1, &this->EBO2);
			glBindVertexArray(this->VAO2);
			glBindBuffer(GL_ARRAY_BUFFER, this->VBO2);
			if (this->hitray_position_.size())
				glBufferData(GL_ARRAY_BUFFER, this->hitray_position_.size() * sizeof(pcm::PointType), &this->hitray_position_[0], GL_STREAM_DRAW);
			else
				glBufferData(GL_ARRAY_BUFFER, 1*sizeof(pcm::PointType), 0, GL_STREAM_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO2);
			if (this->hitray_idx_.size())
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->hitray_idx_.size() * sizeof(GLuint), &this->hitray_idx_[0], GL_STREAM_DRAW);
			else
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, 1* sizeof(GLuint), 0, GL_STREAM_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(pcm::PointType), (GLvoid*)0);
			glBindVertexArray(0);
			isHitrayBufferSetup = true;


	}

}

void KDTree::updateBuffer()
{

	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	if (this->cubic_position_.size())
		glBufferData(GL_ARRAY_BUFFER, this->cubic_position_.size() * sizeof(pcm::PointType), &this->cubic_position_[0], GL_STREAM_DRAW);
	else
		glBufferData(GL_ARRAY_BUFFER, 1* sizeof(pcm::PointType), 0, GL_STREAM_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
	if (this->cubic_idx_.size())
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->cubic_idx_.size() * sizeof(GLuint), &this->cubic_idx_[0], GL_STREAM_DRAW);
	else
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 1* sizeof(GLuint), 0, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
void KDTree::updateHitrayBuffer()
{
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO2);
	if (this->hitray_position_.size())
		glBufferData(GL_ARRAY_BUFFER, this->hitray_position_.size() * sizeof(pcm::PointType), &this->hitray_position_[0], GL_STREAM_DRAW);
	else
		glBufferData(GL_ARRAY_BUFFER, 1* sizeof(pcm::PointType), 0, GL_STREAM_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO2);
	if (this->hitray_idx_.size())
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->hitray_idx_.size() * sizeof(GLuint), &this->hitray_idx_[0], GL_STREAM_DRAW);
	else
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 1* sizeof(GLuint), 0, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
void KDTree::drawKdTree()
{
	canvas_->makeCurrent();
	if (!isBufferSetup)
		setupBuffer();
	if (!isHitrayBufferSetup)
		updateHitrayBuffer();
	updateViewOfMesh();
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glLineWidth(0.01f);
	//glEnable(GL_POINT_SMOOTH);
	//glPointSize(20.0f);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
	if (openglShader)
	{
		openglShader->Use();
		//glUniformMatrix4fv(glGetUniformLocation(openglShader->Program, "projection"), 1, GL_FALSE, p_projmatrix_);
		//glUniformMatrix4fv(glGetUniformLocation(openglShader->Program, "view"), 1, GL_FALSE, p_viewmatrix_);
		//glUniformMatrix4fv(glGetUniformLocation(openglShader->Program, "model"), 1, GL_FALSE, p_modelmatrix_);
		pcm::Matrix44 projmatrix, viewmatrix, modelmatrix;
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				projmatrix(i, j) = p_projmatrix_[i + 4 * j];
				viewmatrix(i, j) = p_viewmatrix_[i + 4 * j];
				modelmatrix(i, j) = p_modelmatrix_[i + 4 * j];
			}
		}
		modelmatrix = projmatrix*viewmatrix*modelmatrix;
		float allmatrix[16];
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				allmatrix[i + 4 * j] = modelmatrix(i,j);

			}
		}

		glUniformMatrix4fv(glGetUniformLocation(openglShader->Program, "allmatrix"), 1, GL_FALSE, allmatrix);
		GLint lightColorLoc = glGetUniformLocation(openglShader->Program, "lightColor");
		GLint lightPosLoc = glGetUniformLocation(openglShader->Program, "lightPos");
		GLint viewPosLoc = glGetUniformLocation(openglShader->Program, "viewPos");
		glUniform3f(lightColorLoc, 1.0f, 1.0f, 1.0f); //light color
		qglviewer::Vec camera_pos = canvas_->camera()->position();
		glUniform3f(lightPosLoc, camera_pos.x, camera_pos.y, camera_pos.z);   //light position
		glUniform3f(viewPosLoc, camera_pos.x, camera_pos.y, camera_pos.z); //camera position		
		
		
		glBindVertexArray(this->VAO);
		glMultiDrawElements(GL_LINE_LOOP, (const GLsizei*)count, GL_UNSIGNED_INT, (const GLvoid**)indices, element_size);
		//for (GLsizei i = 0; i < element_size; ++i)
		//{

		//	glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_INT,(GLuint*) (0)+4*i);
		//}
//		glDrawElements(GL_LINE_LOOP, cubic_idx_.size()/3*3, GL_UNSIGNED_INT, 0);

//		glDrawArrays(GL_LINE_LOOP, 0, 4);
		glBindVertexArray(0);

		glBindVertexArray(this->VAO2);

		//		glMultiDrawElements(GL_LINE_LOOP, (const GLsizei*)count, GL_UNSIGNED_INT, (const GLvoid**)indices, element_size);
		int ray_size = hitray_idx_.size() / 2;
		for (GLsizei i = 0; i < ray_size; ++i)
		{
			glDrawElements(GL_LINE_LOOP, 2, GL_UNSIGNED_INT, (GLuint*)(0) + 2 * i);
		}

		glBindVertexArray(0);


		openglShader->UnUse();

	}

}

bool KDTree::hit(const Ray& ray, float& t, float& min, HitResult& hitResult)
{
	if (root->hit(root, ray, t, min, hitResult))
	{
		//add hit result to render;
		int vtx_size = hitray_position_.size();
		hitray_position_.push_back(ray.origin);
		hitray_position_.push_back(hitResult.target_ph);
		//hitray_position_.push_back(hitResult.ph+pcm::PointType(0.01f,0.01f,0.01f));
		//hitray_position_.push_back(ray.origin+pcm::Vec3(0.01f, 0.01f, 0.01f));
		hitray_idx_.push_back(vtx_size + 0);
		hitray_idx_.push_back(vtx_size + 1);
		//hitray_idx_.push_back(vtx_size + 2);
		//hitray_idx_.push_back(vtx_size + 3);
		isHitrayBufferSetup = false;
		return true;
	}
	else
	{
		int vtx_size = hitray_position_.size();
		hitray_position_.push_back(ray.origin);
		hitray_position_.push_back(ray.origin+ray.max_length*ray.dir);
		//hitray_position_.push_back(hitResult.ph+pcm::PointType(0.01f,0.01f,0.01f));
		//hitray_position_.push_back(ray.origin+pcm::Vec3(0.01f, 0.01f, 0.01f));
		hitray_idx_.push_back(vtx_size + 0);
		hitray_idx_.push_back(vtx_size + 1);
		//hitray_idx_.push_back(vtx_size + 2);
		//hitray_idx_.push_back(vtx_size + 3);
		isHitrayBufferSetup = false;
	}
	return false;
}
