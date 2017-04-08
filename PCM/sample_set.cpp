#include "sample_set.h"
#include "triangle.h"
qglviewer::Vec translate_interval(0.0f, 0.0f, 1.0f);
qglviewer::Vec translate(0.0f, 0.0f, -1.4f);
Sample* SampleSet::add_sample_Fromfile(std::string input_mesh_path, FileIO::FILE_TYPE type)
{

	Sample* new_sample = FileIO::load_point_cloud_file(input_mesh_path, type);
	int sample_idx =  this->size() -1;
	++sample_idx;
	new_sample->smpId = sample_idx;
	new_sample->setLoaded(true);
	new_sample->set_color(Color_Utility::span_color_from_table(sample_idx));
	new_sample->set_visble(true);
	new_sample->build_kdtree();
	new_sample->getFrame().translate(translate + sample_idx*translate_interval);
	this->push_back(new_sample);
	new_sample->update_openglMesh();
	return new_sample;
	
}

Sample* SampleSet::add_sample_FromArray(std::vector<float>& _vertices, std::vector<int>& _faces)
{
	Sample* new_sample = new Sample();
	int sample_idx = this->size() - 1;
	++sample_idx;
	new_sample->smpId = sample_idx;
	for (int i = 0; i < _vertices.size() / 3; ++i)
	{
		pcm::NormalType normal; normal << 0.0f, 0.0f, 0.0f;
		pcm::ColorType  color = Color_Utility::span_color_from_table(sample_idx);
		new_sample->add_vertex(
			pcm::PointType(_vertices[3 * i + 0], _vertices[3 * i + 1], _vertices[3 * i + 2]),
			normal,
			color);

	}
	for (int i = 0; i < _faces.size() / 3; ++i)
	{
		TriangleType* tt = new TriangleType(*new_sample);
		tt->set_i_vetex(0, _faces[3 * i + 0]);
		tt->set_i_vetex(1, _faces[3 * i + 1]);
		tt->set_i_vetex(2, _faces[3 * i + 2]);
		tt->set_i_normal(0, _faces[3 * i + 0]);
		tt->set_i_normal(1, _faces[3 * i + 1]);
		tt->set_i_normal(2, _faces[3 * i + 2]);
		new_sample->add_triangle(*tt);
		delete tt;
	}
	new_sample->setLoaded(true);
	new_sample->set_color(Color_Utility::span_color_from_table(sample_idx));
	new_sample->build_kdtree();
	new_sample->caculateNorm(pcm::NormalType());
	new_sample->getFrame().translate(translate + sample_idx*translate_interval);
	this->push_back(new_sample);
	new_sample->update_openglMesh();
	return new_sample;
}

void SampleSet::push_back( Sample* new_sample )
{
	if (new_sample != nullptr)
	{
		set_.push_back(new_sample);
	}

}

void SampleSet::clear()
{
	while( set_.empty() == false )
	{
		delete set_.back();
		set_.pop_back();
	}
}
