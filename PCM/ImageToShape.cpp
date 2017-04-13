#include "ImageToShape.h"

DepthImage::DepthImage() :cols_(0), rows_(0),min_z(1000.0f),max_z(-1000.0f), sampleRatio_(5)
{

}

void DepthImage::clear()
{
	depth_image_.clear();
	faces_.clear();
	cols_ = 0;
	rows_ = 0;
}

void DepthImage::loadImage(std::vector<float>& _in, int rows, int cols)
{
	depth_image_ = _in;
}

void DepthImage::loadImageFromFile(std::string& _path)
{
	clear();
	std::ifstream reader(_path);
	reader >> rows_ >> cols_;
	float data;
	while (reader >> data)
	{
		if (data > max_z) max_z = data;
		if (data < min_z) min_z = data;
		depth_image_.push_back(data);
	}
}

float DepthImage::operator()(int _x, int _y)
{
	if (_x >= 0 && _x < cols_ && _y >= 0 && _y < rows_)
	{
		return depth_image_[_x  + _y * cols_];
	}
	else
	{
		Logger << "out of range" << std::endl;
		return  - 1;
	}
}

int DepthImage::getIdx(int _x, int _y)
{
	return _x  + _y*((cols() -1)/ sampleRatio_+1);
}
void DepthImage::toFloatArray(std::vector<float>& _in)
{
	_in.clear();
	float size = cols_ > rows_ ? cols_ : rows_;
	float height = (max_z - min_z) * 5;
	for (size_t y = 0; y < rows(); y+= sampleRatio_)
	{
		for (size_t x = 0; x < cols(); x += sampleRatio_)
		{
			_in.push_back(x/ size);
			_in.push_back(y/ size);
			_in.push_back( -(*this)(x, y)/ height);
		}
	}
}
void DepthImage::toPointTypeArray(std::vector<pcm::PointType>& _in)
{
	_in.clear();
	float size = cols_ > rows_ ? cols_ : rows_;
	float height = (max_z - min_z)*5;
	for (size_t y = 0; y < rows(); y += sampleRatio_)
	{
		for (size_t x = 0; x < cols(); x += sampleRatio_)
		{
			_in.push_back(pcm::PointType(y/ size, x/ size, - (*this)(x,y)/ height));
		}
	}
}

void DepthImage::generateFaces(std::vector <int>& _in)
{
	if (faces_.size())
	{
		_in = faces_;
	}
	else
	{
		_in.clear();
		for (size_t y1 = 0,y=0; y1 < rows() - sampleRatio_; y1+=sampleRatio_,y++)
		{
			for (size_t x1 = 0,x=0; x1 < cols() - sampleRatio_; x1 += sampleRatio_,x++)
			{
				_in.push_back(getIdx(x, y));
				_in.push_back(getIdx(x , y+1));
				_in.push_back(getIdx(x + 1 ,y));
				_in.push_back(getIdx(x + 1, y));
				_in.push_back(getIdx(x, y+1));
				_in.push_back(getIdx(x + 1, y + 1));
			}
		}
		faces_ = _in;
	}
}
