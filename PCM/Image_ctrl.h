#pragma once
#include <vector>
#include <sstream>
#include <fstream>
class Image_ctrl
{
public:
	void add_depthImage(std::string _file_paths, std::string name);
	void add_example_image(std::string _file_paths, std::string name);

};