#include "toolbox/gl_utils/glsave.hpp"
#include "LBS_Control.h"
using namespace pcm;

void Handle::draw(const Matrix44& adjust_matrix, bool isWithName /*= false*/)
{
	ColorType color = Color_Utility::span_color_from_table(handle_idx_);
	glEnable(GL_POINT_SMOOTH);
	if (isSelected)
	{
		glPointSize(20.0f);
		glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
	}
	else
	{
		glPointSize(10.0f);
		glColor4f(color(0), color(1), color(2), color(3));
	}
	glBegin(GL_POINTS);
	Vec4	tmp(frame_.translation().x, frame_.translation().y, frame_.translation().z, 1.);
	Vec4	point_to_show = adjust_matrix * tmp;
	glVertex3f(point_to_show(0), point_to_show(1), point_to_show(2));
	glEnd();
	glPushMatrix();

	GLdouble scale[16];
	for (int j = 0; j < 4; ++j)
	{
		scale[4 * j + 0] = adjust_matrix(0, j);
		scale[4 * j + 1] = adjust_matrix(1, j);
		scale[4 * j + 2] = adjust_matrix(2, j);
		scale[4 * j + 3] = adjust_matrix(3, j);
	}
	glTranslated(adjust_matrix(0, 3), adjust_matrix(1, 3), adjust_matrix(2, 3));
	glScaled(adjust_matrix(0, 0), adjust_matrix(1, 1), adjust_matrix(2, 2));
	glMultMatrixd(frame_.matrix());  //frame matrix reference in sample frame
	LBS_Control::drawFrame(1.5f);
	glPopMatrix();
}

void MeshControl::updateSample()
{
	Q_ASSERT(handles_.size() == bind_frame_.size());

	int num_vtx = smp_.num_vertices();
	for (int i_vertex = 0; i_vertex < num_vtx; ++i_vertex)
	{
		float x = inputVertices_[3 * i_vertex];
		float y = inputVertices_[3 * i_vertex + 1];
		float z = inputVertices_[3 * i_vertex + 2];
		qglviewer::Vec cur_point(x, y, z);
		qglviewer::Vec acc_point;


		for (int i_indice = 0; i_indice < numIndices_; ++i_indice)
		{
			int i_bone = boneWightIdx_[i_vertex*numIndices_ + i_indice];

			qglviewer::Frame bind_frame = bind_frame_[i_bone];
			qglviewer::Frame cur_frame = handles_[i_bone]->frame_;
/*    trans     Ori
	[R2 T2] * [R1 T1] = [R2*R1 R2*T1+T2]
	[0   1]	  [0   1]   [  0       1   ]
	accord it to caculate R2, T2
			*/
			qglviewer::Quaternion rotate_quat = cur_frame.rotation()*bind_frame.rotation().inverse();
			qglviewer::Vec translate = cur_frame.translation() - rotate_quat.rotate(bind_frame.translation());
			
			qglviewer::Vec acc_translate;
			qglviewer::Quaternion acc_rotate;
			qglviewer::Quaternion acc_rotate_quat;

			acc_translate += translate;
			//let it identity
			acc_rotate_quat = rotate_quat;
			if (isQlerp_)
				acc_rotate_quat.normalized();
			acc_rotate = acc_rotate_quat;
			TransAndRotation final_transfo;
			final_transfo.rotation_ = acc_rotate;
			final_transfo.translation_ = acc_translate;
			float bone_weight = boneWeights_[i_vertex*numIndices_ + i_indice];
			acc_point = acc_point + bone_weight * (final_transfo.rotation_*cur_point + final_transfo.translation_);
		}

		smp_.setLocalPosition(i_vertex, acc_point);

	}
}

void MeshControl::draw(bool isWithName /*= false*/)
{
	Tbx::GLEnabledSave save_light(GL_LIGHTING, true, false);
//	Tbx::GLEnabledSave save_depth(GL_DEPTH_TEST, true, false);
	Tbx::GLEnabledSave save_blend(GL_BLEND, true, true);
	Tbx::GLBlendSave save_blend_eq;
	glAssert(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
	Tbx::GLEnabledSave save_alpha(GL_ALPHA_TEST, true, true);
	Tbx::GLEnabledSave save_textu(GL_TEXTURE_2D, true, false);
	Tbx::GLLineWidthSave save_line;

	glPushMatrix();
	glMultMatrixd(smp_.getFrame().matrix());
	Matrix44 mat = smp_.matrix_to_scene_coord();

	for (unsigned int idx = 0; idx < handles_.size(); idx++)
	{

		handles_[idx]->draw(mat, isWithName);

		//vertices_[idx]->drawNormal(mat, bias);

	}
	glPopMatrix();
}

void MeshControl::bindControl(std::vector<float>& _inputVertices,/*ori points*/ /*std::vector<qglviewer::Frame>& _bind_frame,*/ std::vector<TransAndRotation>& _transfo, std::vector<float>& _boneWeights, std::vector<int>& _boneWightIdx, int numIndices, int numBone, int numVertices, bool _isQlerp)
{
	for (size_t i = 0; i < handles_.size(); i++)
	{
		delete handles_[i];
	}
	handles_.clear();
	bind_frame_.clear();
	transfo_.clear();
	inputVertices_ = _inputVertices;
	//		bind_frame_		= _bind_frame;
	transfo_ = _transfo;
	boneWeights_ = _boneWeights;
	boneWightIdx_ = _boneWightIdx;
	numIndices_ = numIndices;
	numBone_ = numBone;
	numVertices_ = numVertices;
	isQlerp_ = _isQlerp;

	float(*centerOfBone)[3] = new float[numBone_][3]; //to caculate the center of bone
	for (int i = 0; i < numBone_; i++)
	{
		centerOfBone[i][0] = 0.f;
		centerOfBone[i][1] = 0.f;
		centerOfBone[i][2] = 0.f;
	}
	std::vector<int> numOfVertexInBone(numBone_, 0);
	std::vector<float> maxWeight(numBone_, 0);
	std::vector<int> maxWeightOfVetex(numBone_, 0);
	for (int v = 0; v < numVertices; ++v)
	{
		//int maxweigntbone = -1;
		//int max_weight = 0.1;
		for (int b = 0; b < numIndices; ++b)
		{

			if (boneWeights_[v * numIndices + b] > maxWeight[boneWightIdx_[v * numIndices + b]])
			{
				maxWeightOfVetex[boneWightIdx_[v * numIndices + b]] = v;
				maxWeight[boneWightIdx_[v * numIndices + b]] = boneWeights_[v * numIndices + b];
			}

		}

	}
	for (int i = 0; i < numBone_; i++)
	{
		//centerOfBone[i][0] /= numOfVertexInBone[i];
		//centerOfBone[i][1] /= numOfVertexInBone[i];
		//centerOfBone[i][2] /= numOfVertexInBone[i];
		centerOfBone[i][0] = inputVertices_[3 * maxWeightOfVetex[i] + 0];
		centerOfBone[i][1] = inputVertices_[3 * maxWeightOfVetex[i] + 1];
		centerOfBone[i][2] = inputVertices_[3 * maxWeightOfVetex[i] + 2];
		qglviewer::Frame f;
//		f.setReferenceFrame(&smp_.getFrame());  //set sample frame reference frame
		f.setTranslation(centerOfBone[i][0], centerOfBone[i][1], centerOfBone[i][2]);
		bind_frame_.push_back(f);
	}
	delete[] centerOfBone;
	for (int i = 0; i < bind_frame_.size(); ++i)
	{
		TransAndRotation tr = transfo_[i];
		Handle* handle = new Handle(smp_,i);
//		handle->frame_.setReferenceFrame(&smp_.getFrame()); //set sample frame reference frame
//		we need to accord tr to caculate the local position of handle

/*    trans     Ori
	[R2 T2] * [R1 T1] = [R2*R1 R2*T1+T2]
	[0   1]	  [0   1]   [  0       1   ]
*/
		qglviewer::Vec trans = bind_frame_[i].translation();
		tr.rotation_.rotate(trans);
		handle->frame_.setTranslationAndRotation(
			tr.rotation_.rotate(trans) + tr.translation_,
			bind_frame_[i].rotation()*tr.rotation_
		);

		//handle->frame_.setTranslationAndRotation(
		//	bind_frame_[i].translation() + tr.translation_,
		//	bind_frame_[i].rotation()*tr.rotation_);
		handles_.push_back(handle);
	}
}

void LBS_Control::drawFrame(int length)
{

	glLineWidth(2.0f);
	double color[4];
	//z
	color[0] = 0.0f;  color[1] = 0.0f;  color[2] = 1.0f;  color[3] = 1.0f;
	glColor4dv(color);
	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, length);
	glEnd();
	//x
	color[0] = 1.0f;  color[1] = 0.0f;  color[2] = 0.0f;  color[3] = 1.0f;
	glColor4dv(color);
	glPushMatrix();
	glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, length);
	glEnd();
	glPopMatrix();
	//y
	color[0] = 0.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
	glColor4dv(color);
	glPushMatrix();
	glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, length);
	glEnd();
	glPopMatrix();
}
