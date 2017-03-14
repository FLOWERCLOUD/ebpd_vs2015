#ifndef _EXAMPLE_WEIGHT_SOLVER_
#define  _EXAMPLE_WEIGHT_SOLVER_
#include "toolbox/maths/transfo.hpp"
#include "toolbox/maths/point3.hpp"
#include "solver/fadiff.h"
#include "solver/badiff.h"
#include <vector>
#include <map>

class ExampleSover
{
public:
	ExampleSover(  
		const std::vector<float>& inputVertices, int numVertices,
		const std::vector<Tbx::Transfo>& transfosOfExamples,int numBone, int numExample,int numbIndices,
		const std::vector<float>& boneWeights,
		const std::vector<int>& boneWightIdx )
	{

		m_inputVertices = inputVertices;
		m_numVertices = numVertices;
		m_transfosOfExamples = transfosOfExamples;
		m_numBone = numBone;
		m_numExample = numExample;
		m_numbIndices = numbIndices;
		m_boneWeights =boneWeights;
		m_boneWightIdx = boneWightIdx;
	}
	bool SolveVertices(const std::map<int,Tbx::Vec3>& delta_xi, std::map<int, std::vector<float> >& delta_exampleWeightsOfVertex, std::map<int, std::vector<float> >& ori_exampleWeights);


private:
	bool solver( int vertex_idex ,Tbx::Point3& vtx, Tbx::Vec3& delta , const std::vector<float>& ori_example ,std::vector<float>& delata_example, bool isQlerp);
	std::vector<fadbad::B<fadbad::F<float>> > generateSkinningVetex(int vertex_idex ,std::vector< fadbad::B< fadbad::F<float>> >& vtx, std::vector< fadbad::B<fadbad::F<float>> >& ori_exampleWeights , bool isQlerp);
	bool generateSkinningVetex(int vertex_idex ,Tbx::Point3& vtx, const std::vector<float>& ori_exampleWeights , bool isQlerp);
	std::vector< fadbad::B<fadbad::F<float>> > 
		Skinning_function( std::vector<fadbad::B<fadbad::F<float>> >& ori_exampleWeights ,int num_example ,int vertex_idex , bool isQlerp);
	std::vector<float> m_inputVertices;
	int m_numVertices;
	std::vector<Tbx::Transfo> m_transfosOfExamples;
	int m_numBone;
	int m_numExample;
	int m_numbIndices;
	std::vector<float> m_boneWeights;
	std::vector<int> m_boneWightIdx;

};




#endif