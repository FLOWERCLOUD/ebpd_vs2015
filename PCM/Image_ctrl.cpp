#include "Image_ctrl.h"
#include "GlobalObject.h"
#include "ImageToShape.h"
#include "sample_set.h"
using namespace std;
void Image_ctrl::add_depthImage(std::string _file_paths, std::string name)
{
	string input_depthfile_path = _file_paths + name + ".depth";
	DepthImage image;
	image.loadImageFromFile(input_depthfile_path);
	SampleSet& smpset = (*Global_SampleSet);
	vector<float> vertice_array;
	image.toFloatArray(vertice_array);
	vector<int> faces;
	image.generateFaces(faces);
	smpset.add_sample_FromArray(vertice_array, faces);

}

void Image_ctrl::add_example_image(std::string _file_paths, std::string name)
{


}