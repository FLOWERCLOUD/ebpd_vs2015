#ifndef _TRIANGLE_
#define _TRIANGLE_

#include "basic_types.h"
#include "selectable_item.h"
namespace RenderMode
{
	enum WhichColorMode;
	enum RenderType;
}
class Sample;
class Ray;
class HitResult;
class TriangleType :public SelectableItem
{
public:
	TriangleType(Sample& s ,int idx);
	TriangleType(const TriangleType& s);
	TriangleType& operator = (const TriangleType& s);
	inline void set_i_vetex(IndexType which_vertex, IndexType index)
	{
		i_vertex[which_vertex] = index;
	}
	inline void set_i_normal(IndexType which_normal, IndexType index)
	{
		i_norm[which_normal] = index;
	}
	inline IndexType get_i_vertex(IndexType which_vertex)
	{
		return i_vertex[which_vertex];
	}
	inline IndexType get_i_normal(IndexType which_normal)
	{
		return i_norm[which_normal];
	}
	bool hit(const Ray& ray, float& t, float& min, HitResult& hitResult);
	pcm::PointType get_midpoint()
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
	Box get_bounding_box()
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
	void draw(RenderMode::WhichColorMode& wcm, RenderMode::RenderType& r,
		const Matrix44& adjust_matrix, const Vec3& bias);
	int get_idx()
	{
		return triangle_idx_;
	}
	void set_triangle_idx(int idx)
	{
		triangle_idx_ = idx;
	}
private:
	int triangle_idx_;
	Sample& sample_;

	IndexType i_vertex[3];
	IndexType i_norm[3];
};
#endif

