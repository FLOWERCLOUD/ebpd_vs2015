#pragma once
#include "basic_types.h"
#include <vector>
#include <string>
#include <fstream>

class DepthImage
{
public:
	DepthImage();
	void clear();
	void loadImage(std::vector<float>& _in, int rows, int cols);
	void loadImageFromFile(std::string& _path);
	float operator()(int _x, int _y);
	int getIdx(int _x, int _y);
	int cols()
	{
		return cols_;
	}
	int rows()
	{

		return rows_;
	}
	void  toPointTypeArray(std::vector<pcm::PointType>& _in);
	void toFloatArray(std::vector<float>& _in);
	void horizontal_reverse()
	{

	}
	void vertical_reverse()
	{

	}
	void generateFaces(std::vector <int>& _in);
	void setSampleRatio(int ratio)
	{
		sampleRatio_ = ratio;
	}
private:
	//row first 
	std::vector<float> depth_image_;
	std::vector<int> faces_;
	int cols_; //col size
	int rows_; //row size
	float max_z;
	float min_z;
	int sampleRatio_;


};