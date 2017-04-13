#include "box.h"
#include "ray.h"
static float offset = 0.05f;
static pcm::PointType offset_vec(0.05f, 0.05f, 0.05f);
void Box::expand(const pcm::PointType& p)
{
	diag_dirty_ = true;
	if (first_expand_)
	{
		low_corner_ = p- offset_vec;
		high_corner_ = p+ offset_vec;
		first_expand_ = false;
		return;
	}

	for (int dim = 0; dim < 3; dim++)
	{
		if (p(dim) < low_corner_(dim))
			low_corner_(dim) = p(dim) - offset;
		else if (p(dim) > high_corner_(dim))
			high_corner_(dim) = p(dim) + offset;
	}
}

bool Box::hit(const Ray& ray)
{
	float x0 = low_corner()(0);
	float y0 = low_corner()(1);
	float z0 = low_corner()(2);
	float x1 = high_corner()(0);
	float y1 = high_corner()(1);
	float z1 = high_corner()(2);
	float ox = ray.origin(0);
	float oy = ray.origin(1);
	float oz = ray.origin(2);
	float dx = ray.dir(0);
	float dy = ray.dir(1);
	float dz = ray.dir(2);
	float tx_min, ty_min, tz_min;
	float tx_max, ty_max, tz_max;
	float a = 1.0f / dx;
	if (a >= 0)
	{
		tx_min = (x0 - ox)*a;
		tx_max = (x1 - ox)*a;
	}
	else
	{
		tx_min = (x1 - ox)*a;
		tx_max = (x0 - ox)*a;
	}
	float b = 1.0f / dy;
	if (b >= 0)
	{
		ty_min = (y0 - oy)*b;
		ty_max = (y1 - oy)*b;
	}
	else
	{
		ty_min = (y1 - oy)*b;
		ty_max = (y0 - oy)*b;
	}
	float c = 1.0f / dz;
	if (c >= 0)
	{
		tz_min = (z0 - oz)*c;
		tz_max = (z1 - oz)*c;
	}
	else {
		tz_min = (z1 - oz)*c;
		tz_max = (z0 - oz)*c;
	}
	float t0, t1;
	//find largest entering t value
	if (tx_min > ty_min)
		t0 = tx_min;
	else
		t0 = ty_min;
	if (tz_min > t0)
		t0 = tz_min;
	//find smallest exiting t value
	if (tx_max < ty_max)
		t1 = tx_max;
	else
		t1 = ty_max;
	if (tz_max < t1)
		t1 = tz_max;

	return (t0 < t1 && t1 > 0.000001);
}

