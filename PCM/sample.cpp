#include "sample.h"
#include "windows.h"
#include <gl/gl.h>
#include <gl/glu.h>
#include <fstream>
#include "globals.h"
#include <assert.h>
#include "rendering/render_types.h"
#include "vertex.h"
#include "triangle.h"
#include "file_io.h"
#include <set>
#include <algorithm>
#include "MeshOpenGL.h"
using namespace pcm;
Sample::Sample() :vertices_(),allocator_(),kd_tree_(nullptr),
	kd_tree_should_rebuild_(true),
	mutex_(QMutex::NonRecursive),clayerDepth_(0)
{
	file_type = FileIO::NONE;
	isload_ = false;
	color_mode = RenderMode::OBJECT;
	isOpenglMeshUpdated = false;
	isOpenglMeshColorUpdated = false;
	isUsingProgramablePipeLine = true;
	opengl_mesh_ = new MyOpengl::MeshOpengl(*this);
}

Sample::~Sample()
{ 
	clear();
	if(opengl_mesh_)
		delete opengl_mesh_;
	opengl_mesh_ = NULL;
}
void Sample::clear()
{
	isload_ = false;
	vertices_.clear();
	triangle_array.clear();
	allocator_.free_all(); 
	if(kd_tree_)
		delete	kd_tree_;
	kd_tree_ = NULL;
	lb_wrapbox_.clear();
	wrap_box_link_.clear();
	
}
Vertex* Sample::add_vertex(const PointType& pos = NULL_POINT,
						const NormalType& n = NULL_NORMAL, 
						const ColorType& c = NULL_COLOR)
{
	Vertex*	new_space = allocator_.allocate<Vertex>();
	Vertex* new_vtx = new(new_space)Vertex;
	if ( !new_vtx )
	{
		return nullptr;
	}
	int a = sizeof(Vertex);
	int b = sizeof(SelectableItem);
	new_vtx->set_position(pos);
	new_vtx->set_normal(n);
	new_vtx->set_color(c);
	new_vtx->set_idx(vertices_.size());
	vertices_.push_back(new_vtx);

	box_.expand( pos );
	kd_tree_should_rebuild_ = true;

	return new_vtx;
}

TriangleType* Sample::add_triangle(const TriangleType& tt)
{
	TriangleType*	new_triangle_space = allocator_.allocate<TriangleType>();
	TriangleType* new_triangle = new(new_triangle_space)TriangleType(tt);
	if ( !new_triangle )
	{
		return nullptr;
	}
	triangle_array.push_back(new_triangle);
	return new_triangle;
}
void Sample::draw(ColorMode::ObjectColorMode&, const Vec3& bias )
{
	if (!visible_||!isload_)
	{
		return;
	}
	if (isUsingProgramablePipeLine)
	{

	}
	else
	{
		glPointSize(Paint_Param::g_point_size);
		//glEnable(GL_POINT_SMOOTH);

		glEnable(GL_POINT_SMOOTH);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_DEPTH_TEST);


		ColorType c;

		if (selected_)
		{
			c = HIGHTLIGHTED_COLOR;
			//glColor4f(c(0), c(1), c(2),c(3));
			glColor4f(c(0), c(1), c(2), 0.1);
		}
		else {
			glColor4f(c(0), c(1), c(2), 0.1);
		}
		//glColor4f( color_(0), color_(1), color_(2), color_(3) );
		glPushMatrix();
		glMultMatrixd(m_frame.matrix());
		glBegin(GL_POINTS);

		Matrix44 mat = matrix_to_scene_coord();
		for (IndexType i = 0; i < vertices_.size(); i++)
		{
			vertices_[i]->draw_without_color(mat, bias);
		}
		glEnd();
		glDisable(GL_DEPTH_TEST);

		glPopMatrix();
	}

}

void Sample::draw(ColorMode::VertexColorMode&, const Vec3& bias)
{
	if (!visible_||!isload_)
	{
		return;
	}
	if (isUsingProgramablePipeLine)
	{

	}
	else
	{
		glEnable(GL_DEPTH_TEST);
		glPointSize(Paint_Param::g_point_size);
		glEnable(GL_POINT_SMOOTH);

		glPushMatrix();
		glMultMatrixd(m_frame.matrix());

		glBegin(GL_POINTS);

		Matrix44 mat = matrix_to_scene_coord();
		for (IndexType i = 0; i < vertices_.size(); i++)
		{
			vertices_[i]->draw(mat, bias);
		}
		glEnd();

		glPopMatrix();
		glDisable(GL_DEPTH_TEST);
	}

}

void Sample::draw(ColorMode::LabelColorMode&, const Vec3& bias)
{
	if (!visible_||!isload_)
	{
		return;
	}
	if (isUsingProgramablePipeLine)
	{

	}
	else
	{
		//if ( selected_ )
		{
			//glDisable(GL_LIGHTING);
			glEnable(GL_DEPTH_TEST);
			glPointSize(Paint_Param::g_point_size);
			glEnable(GL_POINT_SMOOTH);
			glPushMatrix();
			glMultMatrixd(m_frame.matrix());
			glBegin(GL_POINTS);
			Matrix44 mat = matrix_to_scene_coord();
			for (IndexType i = 0; i < vertices_.size(); i++)
			{
				vertices_[i]->draw_with_label(mat, bias);
			}
			glEnd();
			glPopMatrix();
			glDisable(GL_POINT_SMOOTH);
			glDisable(GL_DEPTH_TEST);
			//glEnable(GL_LIGHTING);
		}
	}




}

void Sample::draw(ColorMode::WrapBoxColorMode&,const Vec3& bias)
{
	if (!visible_||!isload_)
	{
		return;
	}
	if (isUsingProgramablePipeLine)
	{

	}
	else
	{
		//if ( selected_ )
		{
			//glDisable(GL_LIGHTING);
			glPointSize(Paint_Param::g_point_size);
			glEnable(GL_POINT_SMOOTH);
			glPushMatrix();
			glMultMatrixd(m_frame.matrix());
			glBegin(GL_POINTS);
			Matrix44 mat = matrix_to_scene_coord();
			for (IndexType i = 0; i < vertices_.size(); i++)
			{
				vertices_[i]->draw_with_Graph_wrapbox(mat, bias);
			}
			glPopMatrix();
			glEnd();
			//glEnable(GL_LIGHTING);
		}
	}


}
void Sample::draw(ColorMode::EdgePointColorMode&,const Vec3& bias)
{
	if (!visible_||!isload_)
	{
		return;
	}
	if (isUsingProgramablePipeLine)
	{

	}
	else
	{
		//if ( selected_ )
		{
			//glDisable(GL_LIGHTING);
			glPointSize(Paint_Param::g_point_size);
			glEnable(GL_POINT_SMOOTH);
			glPushMatrix();
			glMultMatrixd(m_frame.matrix());
			glBegin(GL_POINTS);

			Matrix44 mat = matrix_to_scene_coord();
			for (IndexType i = 0; i < vertices_.size(); i++)
			{
				vertices_[i]->draw_with_edgepoints(mat, bias);
			}
			glPopMatrix();
			glEnd();
			//glEnable(GL_LIGHTING);
		}
	}


}
//glLoadIdentity();  
//glColor3f( 1.0f, 1.0f, 1.0f );
//glutSolidSphere( 50.f, 15, 15 ); 
//glutSwapBuffers();



void Sample::draw(ColorMode::SphereMode&,const Vec3& bias)
{
	if (!visible_||!isload_)
	{
		return;
	}
	if (isUsingProgramablePipeLine)
	{

	}
	else
	{
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		//if ( selected_ )
		//glClearColor(0.0, 0.0, 0.0, 0.0);
		//glViewport(0, 0, (GLsizei)1200, (GLsizei)600);
		//glMatrixMode(GL_PROJECTION);
		/*glPushMatrix();
		glTranslatef(point_to_show(0)+bias(0),point_to_show(1)+bias(1), point_to_show(2)+bias(2));

		glColor3f( 1.0  ,0.0 ,0.0);
		glutSolidSphere(0.001* Paint_Param::g_point_size, 10, 10);
		glPopMatrix();
		*/
		//glLoadIdentity();
		//glOrtho(-1.0, 1.0, -1.0, 1.0, -30.0, 30.0);

		////glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		//glEnable(GL_LIGHTING);
		// glEnable(GL_LIGHT0);

		//// Set lighting intensity and color
		//GLfloat qaAmbientLight[]	= {0.2, 0.2, 0.2, 1.0};
		//GLfloat qaDiffuseLight[]	= {0.8, 0.8, 0.8, 1.0};
		//GLfloat qaSpecularLight[]	= {1.0, 1.0, 1.0, 1.0};
		//glLightfv(GL_LIGHT0, GL_AMBIENT, qaAmbientLight);
		//glLightfv(GL_LIGHT0, GL_DIFFUSE, qaDiffuseLight);
		//glLightfv(GL_LIGHT0, GL_SPECULAR, qaSpecularLight);

		//// Set the light position
		//GLfloat qaLightPosition[]	= {0.0, 1.0, -.5, 0.0};
		//glLightfv(GL_LIGHT0, GL_POSITION, qaLightPosition);

		////glDisable(GL_LIGHTING);
		////glPointSize(Paint_Param::g_point_size);
		////glEnable(GL_POINT_SMOOTH);
		////glBegin(GL_POINTS);
		//GLfloat global_ambient[] = { 0.1 ,0.1 ,0.1 ,1.0};
		//glLightModelfv( GL_LIGHT_MODEL_AMBIENT, global_ambient);
		glPushMatrix();
		glMultMatrixd(m_frame.matrix());
		Matrix44 mat = matrix_to_scene_coord();
		for (IndexType i = 0; i < vertices_.size()/* vertices_.size()*/; i++)
		{
			vertices_[i]->draw_with_sphere(mat, bias);
			//glEnd();

		}
		glPopMatrix();
		//glDisable(GL_LIGHTING);
		//glDisable(GL_LIGHT0);
		glDisable(GL_DEPTH_TEST);
		glPopAttrib();
	}

}

void Sample::draw( RenderMode::WhichColorMode& wcm ,RenderMode::RenderType& r,const Vec3& bias)
{
	if (!visible_||!isload_)
	{
		return;
	}

	if (isUsingProgramablePipeLine)
	{
		if(!isOpenglMeshUpdated)
			update_openglMesh();
		if (!isOpenglMeshColorUpdated)
			update_openglMeshColor();
		opengl_mesh_->draw(wcm,r);
	}
	else
	{
		glPushMatrix();
		glMultMatrixd(m_frame.matrix());
		Matrix44 mat = matrix_to_scene_coord();
		switch (r)
		{
		case RenderMode::PointMode: {
			IndexType i_triangle;
			IndexType n_triangel = this->num_triangles();
			glEnable(GL_DEPTH_TEST);
			for (i_triangle = 0; i_triangle < n_triangel; ++i_triangle)
			{
				/*			const std::vector<NormalType>& m_norms = _model->normal_array;
				const std::vector<VertexType>& m_vtxs = _model->vertex_array;*/
				TriangleType* m_triangle = this->triangle_array[i_triangle];
				RenderMode::RenderType rt = RenderMode::PointMode;
				m_triangle->draw(wcm, rt, mat, bias);
			}
			glDisable(GL_DEPTH_TEST);
			break;
		};
		case RenderMode::FlatMode: {

			IndexType i_triangle;
			IndexType n_triangel = this->num_triangles();
			glEnable(GL_DEPTH_TEST);
			//glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
			//glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
			//glEnable(GL_COLOR_MATERIAL);
			//glEnable(GL_LIGHTING);
			//glEnable(GL_LIGHT0);
			//glShadeModel(GL_SMOOTH);
			//SetMaterial(&material);
			for (i_triangle = 0; i_triangle < n_triangel; ++i_triangle)
			{
				/*			const std::vector<NormalType>& m_norms = _model->normal_array;
				const std::vector<VertexType>& m_vtxs = _model->vertex_array;*/
				TriangleType* m_triangle = this->triangle_array[i_triangle];
				RenderMode::RenderType rt = RenderMode::FlatMode;
				m_triangle->draw(wcm, rt, mat, bias);
			}
			//glDisable(GL_COLOR_MATERIAL);
			//glDisable(GL_LIGHTING);
			//glDisable(GL_LIGHT0);

			glDisable(GL_DEPTH_TEST);
			break;
		};
		case RenderMode::WireMode: {

			IndexType i_triangle;
			IndexType n_triangel = this->num_triangles();
			glEnable(GL_DEPTH_TEST);
			glEnable(GL_CULL_FACE);
			for (i_triangle = 0; i_triangle < n_triangel; ++i_triangle)
			{
				TriangleType* m_triangle = this->triangle_array[i_triangle];
				RenderMode::RenderType rt = RenderMode::WireMode;
				m_triangle->draw(wcm, rt, mat, bias);
			}
			glDisable(GL_DEPTH_TEST);
			glDisable(GL_CULL_FACE);
			break;
		};
		case RenderMode::FlatWireMode: {

			IndexType i_triangle;
			IndexType n_triangel = this->num_triangles();
			glEnable(GL_DEPTH_TEST);

			for (i_triangle = 0; i_triangle < n_triangel; ++i_triangle)
			{
				TriangleType* m_triangle = this->triangle_array[i_triangle];
				RenderMode::RenderType rt = RenderMode::FlatWireMode;
				m_triangle->draw(wcm, rt, mat, bias);
			}
			glDisable(GL_DEPTH_TEST);

			break;
		};
		case RenderMode::SmoothMode: {break; };
		case RenderMode::TextureMode: {break; };
		case RenderMode::SelectMode: {break; };


		}
		glPopMatrix();
	}

	
}

void Sample::drawNormal(const Vec3& bias)
{
	if (!visible_||!isload_)
	{
		return;
	}
	if (isUsingProgramablePipeLine)
	{
		if (!isOpenglMeshUpdated)
			update_openglMesh();
		if (!isOpenglMeshColorUpdated)
			update_openglMeshColor();
		opengl_mesh_->drawNormal();
	}
	else
	{
		glEnable(GL_DEPTH_TEST);
		glPushMatrix();
		glMultMatrixd(m_frame.matrix());
		Matrix44 mat = matrix_to_scene_coord();
		for (unsigned int idx = 0; idx < vertices_.size(); idx++)
		{
			vertices_[idx]->drawNormal(mat, bias);
		}
		glPopMatrix();
		glDisable(GL_DEPTH_TEST);
	}

}

void Sample::draw_with_name()
{
	if (!visible_||!isload_)
	{
		return;
	}
	if (isUsingProgramablePipeLine)
	{

	}
	else
	{
		glPushMatrix();
		glMultMatrixd(m_frame.matrix());
		Matrix44 mat = matrix_to_scene_coord();
		//	Matrix44 mat = Matrix44::Identity();
		for (unsigned int idx = 0; idx < vertices_.size(); idx++)
		{
			vertices_[idx]->draw_with_name(idx, mat);
		}
		glPopMatrix();
	}

}
void Sample::caculateNorm(NormalType& baseline  )
{
	if (!this->num_triangles())  //only have points
	{
		const IndexType k = 36;
		for (IndexType i = 0; i < this->num_vertices(); i++)
		{

			MatrixX3	k_nearest_verts(k, 3);
			IndexType		neighbours[k];
			ScalarType dist[k];
			this->neighbours(i, k, neighbours, dist);
			for (IndexType j = 0; j<k; j++)
			{
				IndexType neighbour_idx = neighbours[j];

				k_nearest_verts.row(j) << (*this)[neighbour_idx].x(), (*this)[neighbour_idx].y(), (*this)[neighbour_idx].z();
			}

			MatrixX3 vert_mean = k_nearest_verts.colwise().mean();
			MatrixX3 Q(k, 3);
			for (IndexType j = 0; j<k; j++)
			{
				Q.row(j) = k_nearest_verts.row(j) - vert_mean;
			}

			Matrix33 sigma = Q.transpose() * Q;

			Eigen::EigenSolver<Matrix33> eigen_solver(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);

			auto ev = eigen_solver.eigenvectors();
			auto eval = eigen_solver.eigenvalues();
			ScalarType tmp[3] = { eval(0).real(),eval(1).real(),eval(2).real() };
			IndexType min_idx = std::min_element(tmp, tmp + 3) - tmp;
			NormalType nv;
			nv(0) = (ev.col(min_idx))(0).real();
			nv(1) = (ev.col(min_idx))(1).real();
			nv(2) = (ev.col(min_idx))(2).real();

			nv.normalize();
			if ((baseline).dot(nv) < 0)
			{
				nv = -nv;
			}

			(*this)[i].set_normal(nv);

		}

	}
	else
	{ //has face
	  //auto& m_triangles  = smp.triangle_array;
		Sample& smp = *this;
		for (IndexType i = 0; i < this->num_triangles(); ++i)
		{
			IndexType i_vetex1 = smp.getTriangle(i).get_i_vertex(0);
			IndexType i_vetex2 = smp.getTriangle(i).get_i_vertex(1);
			IndexType i_vetex3 = smp.getTriangle(i).get_i_vertex(2);
			PointType vtx1(smp[i_vetex1].x(), smp[i_vetex1].y(), smp[i_vetex1].z());
			PointType vtx2(smp[i_vetex2].x(), smp[i_vetex2].y(), smp[i_vetex2].z());
			PointType vtx3(smp[i_vetex3].x(), smp[i_vetex3].y(), smp[i_vetex3].z());
			PointType vector1 = vtx2 - vtx1;
			PointType vector2 = vtx3 - vtx1;
			vector1.normalize();
			vector2.normalize();
			PointType vector3 = vector1.cross(vector2); //get the normal of the triangle
			vector3.normalize();
			//Logger<<"vector1: "<<vector1(0)<<" "<<vector1(1)<<" "<<vector1(2)<<std::endl;
			//Logger<<"vector2: "<<vector2(0)<<" "<<vector2(1)<<" "<<vector2(2)<<std::endl;
			//Logger<<"vector3: "<<vector3(0)<<" "<<vector3(1)<<" "<<vector3(2)<<std::endl;
			//assign the normal to all the vertex of the triangle
			for (int x = 0; x<3; ++x)
			{
				IndexType i_normal = smp.getTriangle(i).get_i_normal(x);
				//Logger<<"norm: "<<smp[i_normal].nx()<<" "<<smp[i_normal].ny()<<" "<<smp[i_normal].nz()<<std::endl;
				smp[i_normal].set_normal(
					NormalType(smp[i_normal].nx() + vector3(0), smp[i_normal].ny() + vector3(1), smp[i_normal].nz() + vector3(2)));
			}


		}
		for (IndexType i = 0; i < smp.num_vertices(); i++)
		{
			NormalType norm(smp[i].nx(), smp[i].ny(), smp[i].nz());
			norm.normalize();
			//			Logger<<"norm: "<<norm(0)<<" "<<norm(1)<<" "<<norm(2)<<std::endl;
			smp[i].set_normal(norm);
		}

	}
	caculateTangent();
	this->update_openglMesh();


}
void Sample::caculateTangent()
{
	Sample& smp = *this;
	for (IndexType i = 0; i < this->num_vertices(); i++)
	{

		Vec3 tangent;
		Vec3 normal(smp[i].nx(), smp[i].ny(), smp[i].nz());
		Vec3 c1 = normal.cross(Vec3(0.0, 0.0, 1.0));
		Vec3 c2 = 	normal.cross(Vec3(0.0, 1.0, 0.0));


		if (c1.squaredNorm() > c2.squaredNorm())
		{
			tangent = c1;
		}
		else
		{
			tangent = c2;
		}
		tangent.normalize();
		
		Vec3 bitangent = tangent.cross(normal);
		smp[i].set_tangent(tangent);
		smp[i].set_bi_tangent(bitangent);

	}




}

void Sample::build_kdtree()
{
	if( !kd_tree_should_rebuild_  || vertices_.size() == 0 )
	{
		return;
	}


	if (kd_tree_)
	{
		delete kd_tree_;
	}

	size_t n_vtx = vertices_.size();
	vtx_matrix_ = Matrix3X( 3, n_vtx );

	//reconstruct vtx_matrix
	for ( IndexType v_idx = 0; v_idx < n_vtx; v_idx++ )
	{
		Vertex*	pv = vertices_[v_idx];
		vtx_matrix_.col(v_idx) << pv->x() , pv->y() , pv->z();
	}

	kd_tree_ = new nanoflann::KDTreeAdaptor<Matrix3X, 3>(vtx_matrix_);

	kd_tree_should_rebuild_ = false;


}

IndexType Sample::closest_vtx( const PointType& query_point ) 
{
	qglviewer::Vec query_vec(query_point(0), query_point(1), query_point(2));

	float min_distance = 10000.0f;
	IndexType short_idx = 0;
	for (int i = 0; i < vertices_.size(); ++i)
	{
//		qreal x, y, z;
		//m_input_objs[i]->frame.getPosition(x, y, z);
		
		Matrix44 mat = matrix_to_scene_coord();
		Vec4	tmp(vertices_[i]->x(), vertices_[i]->y(), vertices_[i]->z(), 1.);
		Vec4	point_to_show = mat * tmp;
		qglviewer::Vec cur_point(point_to_show(0), point_to_show(1), point_to_show(2));
		qglviewer::Vec world_pos = m_frame.inverseCoordinatesOf(cur_point);
		//qglviewer::Vec cur_point(x, y, z);
		float distance = (query_vec - world_pos).squaredNorm();
		if (distance < min_distance)
		{
			min_distance = distance;
			short_idx = i;

		}

	}
	return short_idx;

	//if (kd_tree_should_rebuild_)
	//{
	//	return -1;
	//}

	//return kd_tree_->closest( qp );
	//

}

Matrix44 Sample::matrix_to_scene_coord()
{
	Matrix44 mat;

	const ScalarType scale_factor = 1./box_.diag();
	const PointType	 box_center = box_.center();

	mat << scale_factor, 0, 0, -box_center(0)*scale_factor,
			0, scale_factor, 0, -box_center(1)*scale_factor,
			0, 0, scale_factor, -box_center(2)*scale_factor,
			0, 0, 0, 1;

	return mat;
						
}
Matrix44 Sample::inverse_matrix_to_scene_coord()
{
	Matrix44 mat;

	const ScalarType scale_factor = box_.diag();
	const PointType	 box_center = box_.center();

	mat << scale_factor, 0, 0, box_center(0),
		0, scale_factor, 0, box_center(1),
		0, 0, scale_factor, box_center(2),
		0, 0, 0, 1;

	return mat;

}

bool Sample::neighbours(const IndexType query_point_idx, const IndexType num_closet, IndexType* out_indices)
{
	if (kd_tree_should_rebuild_)
	{
		return false;
	}

	ScalarType* out_distances = new ScalarType[num_closet];
	ScalarType	qp[3] = {vertices_[query_point_idx]->x(), 
						vertices_[query_point_idx]->y(), 
						vertices_[query_point_idx]->z() };
	kd_tree_->query( qp, num_closet, out_indices, out_distances);

	delete out_distances;

	return true;
}

bool Sample::neighbours(const IndexType query_point_idx, const IndexType num_closet,
						IndexType* out_indices,ScalarType* out_distances)
{
	if (kd_tree_should_rebuild_)
	{
		return false;
	}

	ScalarType	qp[3] = {vertices_[query_point_idx]->x(), 
		vertices_[query_point_idx]->y(), 
		vertices_[query_point_idx]->z() };
	kd_tree_->query( qp, num_closet, out_indices, out_distances);

	return true;
}

void Sample::update()
{
	if (!isload_)
	{
		return;
	}
	assert( vtx_matrix_.cols() == vertices_.size() );
	IndexType v_idx = 0;
	box_ = Box();
	for ( vtx_iterator v_iter = begin();
			v_iter != end(); v_iter++,v_idx++ )
	{
		PointType p( vtx_matrix_(0, v_idx),
					vtx_matrix_(1, v_idx),
					vtx_matrix_(2, v_idx));
		(*v_iter)->set_position( p );
		box_.expand( p );
	}
	kd_tree_should_rebuild_ = true;
	build_kdtree();
	update_openglMesh();
}

void Sample::delete_vertex_group(const std::vector<IndexType>& idx_grp )
{
	IndexType  i=0, j=0;
	IndexType size = idx_grp.size();
	if (size==0)
	{
		return;
	}
	for ( std::vector<Vertex*>::iterator iter = vertices_.begin();
		iter != vertices_.end(); i++)
	{
		if ( i == idx_grp[j] )
		{
			//This is the node to delete
			iter = vertices_.erase( iter );
			j++;
			if ( j>=size )
			{
				break;
			}
		}
		else
		{
			iter++;
		}
	}

	//kdtree dirty
	kd_tree_should_rebuild_ = true;
	build_kdtree();
	update_openglMesh();
}

void Sample::set_vertex_label(const std::vector<IndexType>& idx_grp ,IndexType label)
{
	if (!isload_)
	{
		return;
	}
	IndexType  i=0, j=0;
	IndexType size = idx_grp.size();
	if (size==0)
	{
		return;
	}
	for ( std::vector<Vertex*>::iterator iter = vertices_.begin();
		iter != vertices_.end(); i++,++iter)
	{
		if ( i == idx_grp[j] )
		{
			//This is the node to label
			(*iter)->set_label(label);
			j++;
			if ( j>=size )
			{
				break;
			}
		}
	}
	Logger<<"set vertx label groupsize "<<idx_grp.size()<<" label : "<<label<<"\n";

}

bool Sample::load()
{
	 bool isload = FileIO::load_point_cloud_file(this);
	 isload_ =isload;
	 return isload_;

}

bool Sample::unload()
{
	return false;
}

void Sample::set_visble(const bool v)
{
	if(v)
	{		if(isload_)
			{
				visible_ = true;
			}
			else
			{
				if(load())visible_ = true;
				else visible_ = false;
			}
				
	}else
	{
		visible_ = v;
	}

}
//rearrange the label in order
void Sample::smoothLabel()
{
	std::set<int> labels; //to sort the lable in ascend order
	for( auto iter = begin() ; iter!=end(); ++iter)
	{
		labels.insert( (*iter)->label());
	}
	Logger<<"smooth label:\n ";

	std::vector<int> old_label( labels.size()); 
	std::copy(labels.begin(),labels.end(), old_label.begin());
	std::vector<int> labelmap( *std::max_element(old_label.begin(),old_label.end()) + 1); //we must use the max label +1as the size
	int count = 0;
	for(int i = 0;i<old_label.size();++i)
	{
		labelmap[ old_label[i]] = count ;		
		Logger<<"old :"<<old_label[i]<<"new label "<<count<<std::endl;
		++count;
	}
	for( auto iter = begin() ;iter!= end();++iter)
	{
		(*iter)->set_label( labelmap[(*iter)->label()]);
	}

}

void Sample::update_openglMesh()
{
	if (!visible_)
		return;
	if (opengl_mesh_)
		opengl_mesh_->updateMesh();
	isOpenglMeshUpdated = true;

}
void Sample::update_openglMeshColor()
{
	if (!visible_)
		return;
	if (opengl_mesh_)
		opengl_mesh_->updateColor();
	isOpenglMeshColorUpdated = true;
}






