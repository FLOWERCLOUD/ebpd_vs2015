#ifndef _BOX_H
#define _BOX_H
#include "basic_types.h"
class Ray;

class Box
{
public:
	Box():first_expand_(true){}
	~Box(){}

	void expand( const pcm::PointType&  p );
	int longest_axis()
	{
		if (first_expand_)
		{
			return 0;
		}

		if (!diag_dirty_)
		{
			return 0;
		}

		ScalarType d[3];
		for (int dim = 0; dim < 3; dim++)
		{
			d[dim] = fabs(low_corner_(dim) - high_corner_(dim));
		}
		return d[0] > d[1] ? (d[0] > d[2] ? 0 : 2) : d[1] > d[2] ? 1 : 2;

	}
	bool hit(const Ray& ray);
	void reset(){ first_expand_ = true;diag_dirty_=true; }

	const ScalarType diag()
	{
		if (first_expand_)
		{
			return 0.;
		}

		if (!diag_dirty_)
		{
			return diag_;
		}

		ScalarType d[3];
		for (int dim = 0; dim < 3; dim++ )
		{
			d[dim] = fabs( low_corner_(dim) - high_corner_(dim) );
		}

		diag_dirty_ = false;
		diag_ = sqrt( d[0]*d[0] + d[1]*d[1] + d[2]*d[2] );

		return diag_;

	}

	const pcm::PointType center() const
	{
		return ( low_corner_ + high_corner_ )/2;
	}
	const pcm::PointType low_corner() const{return low_corner_;}
	const pcm::PointType high_corner()const{return high_corner_;}

private:
	pcm::PointType low_corner_;
	pcm::PointType high_corner_;

	//Because sqrt is expensive, we record dirty flag to update diag
	bool diag_dirty_;
	ScalarType diag_;
	bool first_expand_;
};

#endif