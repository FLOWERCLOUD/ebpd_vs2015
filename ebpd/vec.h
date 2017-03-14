#ifndef TDVIEWER_VEC_H
#define TDVIEWER_VEC_H
#include <math.h>
#include <iostream>

#include "config.h"

namespace tdviewer
{

class  Vec
{
public:
	union
	{
		struct { qreal x, y, z; };
		qreal v_[3];
	};
	Vec() : x(0.0), y(0.0), z(0.0) {}
	Vec(qreal X, qreal Y, qreal Z) : x(X), y(Y), z(Z) {}
	template <class C>
	explicit Vec(const C& c) : x(c[0]), y(c[1]), z(c[2]) {}
	Vec& operator=(const Vec& v)
	{
		x = v.x;   y = v.y;   z = v.z;
		return *this;
	}
	void setValue(qreal X, qreal Y, qreal Z)
	{ x=X; y=Y; z=Z; }

	qreal operator[](int i) const {
		return v_[i];
	}

	/*! Bracket operator returning an l-value. \p i must range in [0..2]. */
	qreal& operator[](int i) {
		return v_[i];

	}
	operator const qreal*() const {
		return v_;

	}
	operator qreal*() {
		return v_;
	}
	operator const float*() const {
		static float* const result = new float[3];
		result[0] = (float)x;
		result[1] = (float)y;
		result[2] = (float)z;
		return result;
	}
	friend Vec operator+(const Vec &a, const Vec &b)
	{
		return Vec(a.x+b.x, a.y+b.y, a.z+b.z);
	}
	/*! Returns the difference of the two vectors. */
	friend Vec operator-(const Vec &a, const Vec &b)
	{
		return Vec(a.x-b.x, a.y-b.y, a.z-b.z);
	}

	/*! Unary minus operator. */
	friend Vec operator-(const Vec &a)
	{
		return Vec(-a.x, -a.y, -a.z);
	}
	/*! Returns the product of the vector with a scalar. */
	friend Vec operator*(const Vec &a, qreal k)
	{
		return Vec(a.x*k, a.y*k, a.z*k);
	}

	/*! Returns the product of a scalar with the vector. */
	friend Vec operator*(qreal k, const Vec &a)
	{
		return a*k;
	}
	friend Vec operator/(const Vec &a, qreal k)
	{
#ifndef QT_NO_DEBUG
		if (fabs(k) < 1.0E-10)
			qWarning("Vec::operator / : dividing by a null value (%f)", k);
#endif
		return Vec(a.x/k, a.y/k, a.z/k);
	}
	friend bool operator!=(const Vec &a, const Vec &b)
	{
		return !(a==b);
	}
	friend bool operator==(const Vec &a, const Vec &b)
	{
		const qreal epsilon = 1.0E-10;
		return (a-b).squaredNorm() < epsilon;
	}
	Vec& operator+=(const Vec &a)
	{
		x += a.x; y += a.y; z += a.z;
		return *this;
	}
	Vec& operator-=(const Vec &a)
	{
		x -= a.x; y -= a.y; z -= a.z;
		return *this;
	}
	Vec& operator*=(qreal k)
	{
		x *= k; y *= k; z *= k;
		return *this;
	}
	Vec& operator/=(qreal k)
	{
#ifndef QT_NO_DEBUG
		if (fabs(k)<1.0E-10)
			qWarning("Vec::operator /= : dividing by a null value (%f)", k);
#endif
		x /= k; y /= k; z /= k;
		return *this;
	}
	/*! Dot product of the two Vec. */
	friend qreal operator*(const Vec &a, const Vec &b)
	{
		return a.x*b.x + a.y*b.y + a.z*b.z;
	}

	/*! Cross product of the two vectors. Same as cross(). */
	friend Vec operator^(const Vec &a, const Vec &b)
	{
		return cross(a,b);
	}
	/*! Cross product of the two Vec. Mind the order ! */
	friend Vec cross(const Vec &a, const Vec &b)
	{
		return Vec(a.y*b.z - a.z*b.y,
			a.z*b.x - a.x*b.z,
			a.x*b.y - a.y*b.x);
	}

	/*! Returns the \e squared norm of the Vec. */
	qreal squaredNorm() const { return x*x + y*y + z*z; }

	/*! Returns the norm of the vector. */
	qreal norm() const { return sqrt(x*x + y*y + z*z); }

	/*! Normalizes the Vec and returns its original norm.

  Normalizing a null vector will result in \c NaN values. */
		qreal normalize()
	{
		const qreal n = norm();
#ifndef QT_NO_DEBUG
		if (n < 1.0E-10)
			qWarning("Vec::normalize: normalizing a null vector (norm=%f)", n);
#endif
		*this /= n;
		return n;
	}
	/*! Returns a unitary (normalized) \e representation of the vector. The original Vec is not modified. */
	Vec unit() const
	{
		Vec v = *this;
		v.normalize();
		return v;
	}
	Vec orthogonalVec() const;
	void projectOnAxis(const Vec& direction);
	void projectOnPlane(const Vec& normal);
};

}
std::ostream& operator<<(std::ostream& o, const tdviewer::Vec&);
#endif