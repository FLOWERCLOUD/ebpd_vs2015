#pragma once

#include <vector>
#include <DirectXMath.h>
#include "RigidTransform.h"
#include "HorseObject.h"
#include <iostream>
class HorseObject;
namespace SSDR
{
    // Input data structure
    struct Input
    {
        //Number of vertices
        int numVertices;
        //Number of example data
        int numExamples;
        //! Bound vertex coordinates (number of vertices)
        std::vector<DirectX::XMFLOAT3A> bindModel;
        //! Exemplary shape vertex coordinates (number of example data x number of vertices)
        std::vector<DirectX::XMFLOAT3A> sample;

        Input() : numVertices(0), numExamples(0) {}
        ~Input() {}
    };

    // output data structure
    struct Output
    {
        // Number of bones
        int numBones;
        //! Skinning weight (number of vertices x index number)
        std::vector<float> weight;
        //! Index (number of vertices x number of indexes)
        std::vector<int> index;
        //! Skinning matrix (number of example data x number of bones)
        std::vector<RigidTransform> boneTrans;
    };

//calculation parameter structure
    struct Parameter
    {
		//Minimum bone count
        int numMinBones;
        // Maximum number of bones bound per vertex
        int numIndices;
		// Maximum number of iterations
        int numMaxIterations;
    };

	typedef struct RTransform
	{
		float rotation[4];
		float translation[3];
	}RTransform;

	typedef struct cOutput
	{
		// Number of bones
		int numBones;
		//! Skinning weight (number of vertices x index number)
		std::vector<float> weight;
		//! Index (number of vertices x number of indexes)
		std::vector<int> index;
		//! Skinning matrix (number of example data x number of bones)
		std::vector<RTransform> boneTrans;

	} cOutput;
    extern double Decompose(Output& output, const Input& input, const Parameter& param);
    extern double ComputeApproximationErrorSq(const Output& output, const Input& input, const Parameter& param);
	extern void WriteRigToFile(const Output& ssdroutput,const Input& ssdrIn,const Parameter& ssdrParam ,std::string file_paths);
	extern void WriteRigToFileFormat2(const Output& output,const Input& ssdrIn,const Parameter& ssdrParam ,
		const HorseObject* const obj,
		std::string _file_paths_dir,std::string _fine_prifixname);

    extern void GetRigFromFile(Output& result , std::string file_paths);
	extern void WriteAnimationToFile(std::string file_paths,
		const std::vector<RigidTransform>& boneAnim, 
		const HorseObject* const obj,
		//const HorseObject::CustomVertex* const vertexBufferCPU,
		const std::vector<DWORD>& index,
		int Numfaces,
		const Output& output,const Input& ssdrIn ,const Parameter& ssdrParam );
	extern void rtRigidToCom(const RigidTransform& rt , RTransform&  ct);
	extern void ComTOrtRigid( const  RTransform& ct ,RigidTransform& rt );
}
