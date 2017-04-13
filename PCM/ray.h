#pragma once
#include "basic_types.h"
class Ray
{
public:
	Ray() :dir(0.0f, 0.0f, 0.0f),origin(0.0f,0.0f,0.0f),
		t(0),max_length(100)
	{
	}
	pcm::Vec3 dir;
	pcm::Vec3 origin;
	int t;
	int max_length;
};
class HitResult
{
public:
	bool hit_obj;
	pcm::PointType target_ph;
	pcm::PointType target_ph2;
	int target_sample_idx;
	int target_triangle_idx;
	float u;
	float v;
	float t;
	pcm::PointType p0;
	pcm::PointType p1;
	pcm::PointType p2;
	int source_sample_idx;
	int source_vtx_idx;

};