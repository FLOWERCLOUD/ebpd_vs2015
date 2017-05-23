#include "sample.h"
#include "triangle.h"
#include "vertex.h"
#include "ray.h"
#include "rendering/render_types.h"

using namespace pcm;
TriangleType::TriangleType(Sample& s ,int idx):sample_(s), triangle_idx_(idx)
{
	i_vertex[0] =i_vertex[1] = i_vertex[2] = 0;
	i_norm[0]= i_norm[1] = i_norm[2] = 0;
}
TriangleType::TriangleType(const TriangleType& s):sample_(s.sample_), triangle_idx_(s.triangle_idx_)
{
	for(int i = 0 ;i<3;++i)
	{
		this->i_vertex[i] = s.i_vertex[i];
		this->i_norm[i]= s.i_norm[i];
	}
	//Logger<<"TriangleType copy"<<std::endl;
	
}
TriangleType& TriangleType::operator=(const TriangleType& s)
{
	for(int i = 0 ;i<3;++i)
	{
		this->i_vertex[i] = s.i_vertex[i];
		this->i_norm[i]= s.i_norm[i];
	}
	triangle_idx_ = s.triangle_idx_;
	//Logger<<"TriangleType assignment"<<std::endl;
	return *this;
}
/*
ray: x = o + d * t (t > 0)
triangle : x = (1 - u - v) * p + u * p1 + v * p2 (0 <= u <= 1, 0 <= v <= 1)
(1 - u - v) * p + u * p1 + v * p2 = o + d * t
let e1 = p1 - p, e2 = p2 - p
=> e1 * u + e2 * v - d * t = o - p
let K = -d, L = o - p, => e1 * u + e2 * v + K * t = L
So we solve A X = B
*/
bool TriangleType::hit(const Ray& ray, float& _t, float& min, HitResult& hitResult)
{
	using namespace pcm;
	int i0 = get_i_vertex(0);
	int i1 = get_i_vertex(1);
	int i2 = get_i_vertex(2);
	PointType& p0 = sample_[i0].get_position();
	PointType& p1 = sample_[i1].get_position();
	PointType& p2 = sample_[i2].get_position();
	Vec3 e1(p1 - p0);
	Vec3 e2(p2 - p0);
	Vec3 k = -(ray.dir);
	Vec3 B = ray.origin - p0;
	Matrix33 A;
	A << e1(0), e2(0), k(0),
		e1(1), e2(1), k(1),
		e1(2), e2(2), k(2);
	Vec3 x = A.colPivHouseholderQr().solve(B);
	float u = x(0);
	float v = x(1);
	float t = x(2);
	if ( u >=0 && u <= 1.0)
	{
		if (v >= 0 && v <= 1.0)
		{
			if (u + v >= 0 && u + v <= 1.0)
			{
				if (t > 0 && t < ray.max_length)
				{
					hitResult.target_ph = ray.origin + t*ray.dir;
					hitResult.target_ph2 = (1-u-v)*p0 +u*p1 +v*p2;
					hitResult.hit_obj = true;
					hitResult.target_sample_idx = sample_.smpId;
					hitResult.target_triangle_idx = this->get_idx();
					hitResult.u = u;
					hitResult.v = v;
					hitResult.t = t;
					hitResult.p0 = p0;
					hitResult.p1 = p1;
					hitResult.p2 = p2;
					_t = t;
					return true;
				}
				else
				{
					return false;
				}

			}
			else
			{
				return false;
			}
		}
		else
		{
			return false;
		}
	}


	return false;
}

pcm::PointType TriangleType::get_midpoint()
{
	int i0 = get_i_vertex(0);
	int i1 = get_i_vertex(1);
	int i2 = get_i_vertex(2);
	pcm::PointType& p0 = sample_[i0].get_position();
	pcm::PointType& p1 = sample_[i1].get_position();
	pcm::PointType& p2 = sample_[i2].get_position();
	pcm::PointType midp(0.0f, 0.0f, 0.0f);
	midp += p0;
	midp += p1;
	midp += p2;
	midp /= 3;
	return midp;
}

Box TriangleType::get_bounding_box()
{
	int i0 = get_i_vertex(0);
	int i1 = get_i_vertex(1);
	int i2 = get_i_vertex(2);
	pcm::PointType& p1 = sample_[i0].get_position();
	pcm::PointType& p2 = sample_[i1].get_position();
	pcm::PointType& p3 = sample_[i2].get_position();
	Box bbox;
	bbox.expand(p1);
	bbox.expand(p2);
	bbox.expand(p3);
	return bbox;
}

void TriangleType::draw(RenderMode::WhichColorMode& wcm, RenderMode::RenderType& r,const Matrix44& adjust_matrix, const Vec3& bias)
{
	if (!visible_)
	{
		return;
	}
//	std::vector<Vertex*>& vs = sample_.vertices_;
	//r = RenderMode::PointMode;
	switch(r)
	{
		case RenderMode::PointMode:{
			
			break;
		}
	case RenderMode::FlatMode:{

		glBegin(GL_TRIANGLES);
		for( int i = 0 ;i <3 ;++i)
		{
			if(this->i_norm[i]!=-1)
			{
				Vec4	tmpn( sample_[this->i_norm[i]].nx()  ,  sample_[this->i_norm[i]].ny() ,sample_[this->i_norm[i]].nz() ,1.); 
				Vec4	normal_to_show = tmpn;//adjust_matrix * tmpn;


				glNormal3f( 
					normal_to_show(0),
					normal_to_show(1),
					normal_to_show(2));
				//Logger<<"NORMAL: "<<(float)vs[this->i_norm[i]]->nx()<<" "<<
				//	(float)vs[this->i_norm[i]]->ny()<<" "<<
				//	(float)vs[this->i_norm[i]]->nz()<<std::endl;

			}
			switch (sample_.color_mode)
			{
			case RenderMode::VERTEX:
			{
				ColorType color2 = Color_Utility::span_color_from_table(sample_.smpId);
				glColor3f(color2(0), color2(1), color2(2));
				break;
			}
			case RenderMode::HANDLE:
			{
				if (i < sample_.colors_.size())
				{
					if (this->i_norm[i] != -1)
					{
						ColorType color2 = sample_.colors_[i_norm[i]];
						glColor3f(color2(0), color2(1), color2(2));
					}
				}
				else
				{
					ColorType color2 = Color_Utility::span_color_from_table(sample_.smpId);
					glColor3f(color2(0), color2(1), color2(2));
				}

				break;
			}
			case RenderMode::OBJECT:
			{
				ColorType color2 = Color_Utility::span_color_from_table(sample_.smpId);
				glColor3f(color2(0), color2(1), color2(2)
					/*				(GLfloat) vs[this->i_vertex[i]]->r(),
					(GLfloat) vs[this->i_vertex[i]]->g(),
					(GLfloat) vs[this->i_vertex[i]]->b()*/);
				break;
			}


			}

			//Logger<<"COLOR: "<<vs[this->i_vertex[i]]->r()<<" "<<
			//	vs[this->i_vertex[i]]->g()<<" "<<
			//	vs[this->i_vertex[i]]->b()<<std::endl;
			Vec4	tmpv( sample_[this->i_vertex[i]].x()  ,  sample_[this->i_vertex[i]].y() ,sample_[this->i_vertex[i]].z() ,1.);
			Vec4	point_to_show = adjust_matrix * tmpv;
			glVertex3f(	
				point_to_show(0)+ bias(0),
				point_to_show(1)+ bias(1),
				point_to_show(2)+ bias(2) );
			//Logger<<"VERTEX: "<<vs[this->i_vertex[i]]->x()<<" "<<
			//	vs[this->i_vertex[i]]->y()<<" "<<
			//	vs[this->i_vertex[i]]->z()<<std::endl;

		}
		glEnd();
		break;
							   }
	case RenderMode::WireMode:{
		glLineWidth(2.0f);
		glBegin(GL_LINE_LOOP);
		for( int i = 0 ;i <3 ;++i)
		{
			if(this->i_norm[i]!=-1)
			{
				Vec4	tmpn( sample_[this->i_norm[i]].nx()  ,  sample_[this->i_norm[i]].ny() ,sample_[this->i_norm[i]].nz() ,1.);
				Vec4	normal_to_show = adjust_matrix * tmpn;


				glNormal3f( 
				normal_to_show(0),
				normal_to_show(1),
				normal_to_show(2));
			}
			ColorType color2 = Color_Utility::span_color_from_table(sample_[this->i_vertex[i]].label()); 
			glColor3f( color2(0) ,color2(1) ,color2(2)
				/*(GLfloat) vs[this->i_vertex[i]]->r(),
				(GLfloat) vs[this->i_vertex[i]]->g(),
				(GLfloat) vs[this->i_vertex[i]]->b() */);
			Vec4	tmpv( sample_[this->i_vertex[i]].x()  ,  sample_[this->i_vertex[i]].y() ,sample_[this->i_vertex[i]].z() ,1.);
			Vec4	point_to_show = adjust_matrix * tmpv;
			glVertex3f(	
				point_to_show(0)+ bias(0),
				point_to_show(1)+ bias(1),
				point_to_show(2)+ bias(2) );

		}
		glEnd();

		break;
							  }
	case RenderMode::FlatWireMode:{

			glBegin(GL_TRIANGLES);
			for( int i = 0 ;i <3 ;++i)
			{
				if(this->i_norm[i]!=-1)
				{
					Vec4	tmpn( sample_[this->i_norm[i]].nx()  ,  sample_[this->i_norm[i]].ny() ,sample_[this->i_norm[i]].nz() ,1.); 
					Vec4	normal_to_show = tmpn;//adjust_matrix * tmpn;


					glNormal3f( 
						normal_to_show(0),
						normal_to_show(1),
						normal_to_show(2));
					//Logger<<"NORMAL: "<<(float)vs[this->i_norm[i]]->nx()<<" "<<
					//	(float)vs[this->i_norm[i]]->ny()<<" "<<
					//	(float)vs[this->i_norm[i]]->nz()<<std::endl;

				}
				ColorType color2 = Color_Utility::span_color_from_table(sample_[this->i_vertex[i]].label()); 
				glColor3f( color2(0) ,color2(1) ,color2(2)
					/*				(GLfloat) vs[this->i_vertex[i]]->r(),
					(GLfloat) vs[this->i_vertex[i]]->g(),
					(GLfloat) vs[this->i_vertex[i]]->b()*/ );
				//Logger<<"COLOR: "<<vs[this->i_vertex[i]]->r()<<" "<<
				//	vs[this->i_vertex[i]]->g()<<" "<<
				//	vs[this->i_vertex[i]]->b()<<std::endl;
				Vec4	tmpv( sample_[this->i_vertex[i]].x()  ,  sample_[this->i_vertex[i]].y() ,sample_[this->i_vertex[i]].z() ,1.);
				Vec4	point_to_show = adjust_matrix * tmpv;
				glVertex3f(	
					point_to_show(0)+ bias(0),
					point_to_show(1)+ bias(1),
					point_to_show(2)+ bias(2) );
				//Logger<<"VERTEX: "<<vs[this->i_vertex[i]]->x()<<" "<<
				//	vs[this->i_vertex[i]]->y()<<" "<<
				//	vs[this->i_vertex[i]]->z()<<std::endl;

			}
			glEnd();

			glLineWidth(2.0f);
			glBegin(GL_LINE_LOOP);
			for( int i = 0 ;i <3 ;++i)
			{
				if(this->i_norm[i]!=-1)
				{
					Vec4	tmpn( sample_[this->i_norm[i]].nx()  ,  sample_[this->i_norm[i]].ny() ,sample_[this->i_norm[i]].nz() ,1.);
					Vec4	normal_to_show = adjust_matrix * tmpn;


					glNormal3f( 
					normal_to_show(0),
					normal_to_show(1),
					normal_to_show(2));
				}
				glColor3f(	0.5f,0.5f,0.5f
					/*(GLfloat) vs[this->i_vertex[i]]->r(),
					(GLfloat) vs[this->i_vertex[i]]->g(),
					(GLfloat) vs[this->i_vertex[i]]->b() */);
				Vec4	tmpv( sample_[this->i_vertex[i]].x()  ,  sample_[this->i_vertex[i]].y() ,sample_[this->i_vertex[i]].z() ,1.);
				Vec4	point_to_show = adjust_matrix * tmpv;
				glVertex3f(	
					point_to_show(0)+ bias(0),
					point_to_show(1)+ bias(1),
					point_to_show(2)+ bias(2) );

			}
			glEnd();

			break;
									}
	case RenderMode::SmoothMode:{			break;}
	case RenderMode::TextureMode:{			break;}
	case RenderMode::SelectMode:{			break;}	
	default:{}
	}
	
}




