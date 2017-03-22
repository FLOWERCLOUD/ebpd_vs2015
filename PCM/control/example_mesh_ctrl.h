#ifndef _EXAMPLE_MESH_CTRL_
#define _EXAMPLE_MESH_CTRL_

#include <vector>
#include <sstream>
#include <fstream>
class SampleSet;
class ManipulatedObject;
void exportObj( const std::vector<float>& inputVertice ,std::vector<int>&  faces,std::string file_paths);
class Example_mesh_ctrl
{
public:
	void genertateVertices(std::string _file_paths,std::string name);
	void load_example(std::string _file_paths, std::string name);
	void setupExample(std::string _file_paths,std::string name);
	void genertateVertices(std::vector<float>& inputVertices ,std::vector<int>& faces, const std::vector<int>& vertex_idex, const std::vector<float>& impulses ,float mass, float delta_t,float belta );
	void processCollide(SampleSet& smpset, std::vector<ManipulatedObject*>& selected_obj);



};



#endif