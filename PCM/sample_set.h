#ifndef _EXAMPLE_SET_H
#define _EXAMPLE_SET_H
#include "sample.h"
#include <vector>
/* 
	A wrapper class of a set of examples
*/



class SampleSet
{
public:
	SampleSet(){}
	~SampleSet(){}
	//static SampleSet& get_instance()
	//{
	//	static SampleSet	instance;
	//	return instance;
	//}

	Sample* add_sample_Fromfile(std::string input_mesh_path, FileIO::FILE_TYPE = FileIO::OBJ);
	Sample* add_sample_FromArray(std::vector<float>& _vertices, std::vector<int>& _faces);
	void push_back( Sample*  );
	bool empty(){ return set_.empty(); }
	void clear();

	Sample& operator[](size_t idx)
	{
		if (idx >= set_.size())
		{
			Logger << " out of sample range\n ";
			return *set_[0];
		}
		return *set_[idx];
	}

	Vertex&	operator()(IndexType sample_idx, IndexType vertex_idx)
	{
			return  (*set_[sample_idx])[vertex_idx] ;
	}

	size_t size(){ return set_.size(); }
	std::vector<Sample*>& getSmpVector(){
		return set_;

	}
private:


	SampleSet(const SampleSet& );
	void operator=(const SampleSet&);

private:
	std::vector<Sample*>	set_;

};

#endif