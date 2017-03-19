#pragma once

#include "QGLViewer/frame.h"
class ManipulatedObject
{
public:
	int idx;
	void draw() const;

	ManipulatedObject() :idx(-1)
	{}
	inline qglviewer::Vec getWorldPosition()
	{
		return frame.inverseCoordinatesOf(position);

	}
	inline qglviewer::Frame& getFrame()
	{
		return frame;
	}
	inline qglviewer::Vec getPosition()
	{
		return position;
	}
	inline qglviewer::Vec getNormal()
	{
		return normal;
	}
	inline qglviewer::Vec getColor()
	{
		return color;
	}

	inline void setFrame(qglviewer::Frame frame)
	{
		this->frame = frame;
	}
	inline void setPosition(qglviewer::Vec position)
	{
		this->position = position;
	}
	inline void setNormal(qglviewer::Vec normal)
	{
		this->normal = normal;
	}
	inline void setColor(qglviewer::Vec color)
	{
		this->color = color;
	}

protected:
	qglviewer::Frame frame;
	qglviewer::Vec position;
	qglviewer::Vec normal;
	qglviewer::Vec color;
};