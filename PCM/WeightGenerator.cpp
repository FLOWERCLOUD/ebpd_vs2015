#ifndef NO_TETGEN
#define TETLIBRARY
#include <igl/copyleft/tetgen/tetrahedralize.h>
#endif
#include "WeightGenerator.h"
#include "tetgen.h" 
#include "igl/boundary_conditions.h"
#include <igl/writeMESH.h>
#include "LBS_Control.h"
#include "triangle.h"
#include "GlobalObject.h"
#include <iostream>
#include <fstream>
#include <deque>


#ifndef NO_MOSEK
#  include <igl/mosek/bbw.h>
#endif
#include <Eigen/Dense>

using namespace std;
using namespace igl;
using namespace Eigen;

bool mesh_to_tetgenio(
	const std::vector<std::vector<float > > & V,
	const std::vector<std::vector<int> > & F,
	tetgenio & in)
{
	using namespace std;
	// all indices start from 0
	in.firstnumber = 0;

	in.numberofpoints = V.size();
	in.pointlist = new REAL[in.numberofpoints * 3];
	// loop over points
	for (int i = 0; i < (int)V.size(); i++)
	{
		assert(V[i].size() == 3);
		in.pointlist[i * 3 + 0] = V[i][0];
		in.pointlist[i * 3 + 1] = V[i][1];
		in.pointlist[i * 3 + 2] = V[i][2];
	}

	in.numberoffacets = F.size();
	in.facetlist = new tetgenio::facet[in.numberoffacets];
	in.facetmarkerlist = new int[in.numberoffacets];

	// loop over face
	for (int i = 0; i < (int)F.size(); i++)
	{
		in.facetmarkerlist[i] = i;
		tetgenio::facet * f = &in.facetlist[i];
		f->numberofpolygons = 1;
		f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
		f->numberofholes = 0;
		f->holelist = NULL;
		tetgenio::polygon * p = &f->polygonlist[0];
		p->numberofvertices = F[i].size();
		p->vertexlist = new int[p->numberofvertices];
		// loop around face
		for (int j = 0; j < (int)F[i].size(); j++)
		{
			p->vertexlist[j] = F[i][j];
		}
	}
	return true;



}

bool tetgenio_to_tetmesh(
	const tetgenio & out,
	std::vector<std::vector<float > > & V,
	std::vector<std::vector<int> > & T,
	std::vector<std::vector<int> > & F)
{
	using namespace std;
	// process points
	if (out.pointlist == NULL)
	{
		cerr << "^tetgenio_to_tetmesh Error: point list is NULL\n" << endl;
		return false;
	}
	V.resize(out.numberofpoints, vector<float>(3));
	// loop over points
	for (int i = 0; i < out.numberofpoints; i++)
	{
		V[i][0] = out.pointlist[i * 3 + 0];
		V[i][1] = out.pointlist[i * 3 + 1];
		V[i][2] = out.pointlist[i * 3 + 2];
	}


	// process tets
	if (out.tetrahedronlist == NULL)
	{
		cerr << "^tetgenio_to_tetmesh Error: tet list is NULL\n" << endl;
		return false;
	}

	// When would this not be 4?
	assert(out.numberofcorners == 4);
	T.resize(out.numberoftetrahedra, vector<int>(out.numberofcorners));
	int min_index = 1e7;
	int max_index = -1e7;
	// loop over tetrahedra
	for (int i = 0; i < out.numberoftetrahedra; i++)
	{
		for (int j = 0; j < out.numberofcorners; j++)
		{
			int index = out.tetrahedronlist[i * out.numberofcorners + j];
			T[i][j] = index;
			min_index = (min_index > index ? index : min_index);
			max_index = (max_index < index ? index : max_index);
		}
	}
	assert(min_index >= 0);
	assert(max_index >= 0);
	assert(max_index < (int)V.size());

	cout << out.numberoftrifaces << endl;

	// When would this not be 4?
	F.clear();
	// loop over tetrahedra
	for (int i = 0; i < out.numberoftrifaces; i++)
	{
		if (out.trifacemarkerlist[i] >= 0)
		{
			vector<int> face(3);
			for (int j = 0; j < 3; j++)
			{
				face[j] = out.trifacelist[i * 3 + j];
			}
			F.push_back(face);
		}
	}

	return true;
}




#if 0

inline std::vector<Handle*> gather_bones(const std::vector<Handle*> & BR)
{
	// Insert roots into search queue
	std::list<Handle*> Q;
	for (
		std::vector<Handle*>::const_iterator bit = BR.begin();
		bit != BR.end();
		bit++)
	{
		Q.push_back(*bit);
	}
	// Keep track of all bones that get popped 
	std::vector<Handle*> B;
	while (!Q.empty())
	{
		// Read from front because we want to keep order QUEUE
		Handle * b = Q.front();
		Q.pop_front();
		// Add to list
		B.push_back(b);
		// Add children to queue
		std::vector<Handle*> children = b->get_children();
		Q.insert(Q.end(), children.begin(), children.end());
	}
	return B;
}



inline void gather_positions_and_connectivity(
	const std::vector<Handle*> & BR,
	Eigen::MatrixXd & V,
	Eigen::VectorXi & P,
	Eigen::MatrixXi & BE,
	Eigen::VectorXi & WI)
{
	using namespace std;
	vector<Handle *> B = gather_bones(BR);
	// Build map from bone pointers to index in B
	map<const Handle *, int> B2I;
	// Map NULL to -1
	B2I[NULL] = -1;
	int i = 0;
	for (vector<Handle*>::iterator bit = B.begin(); bit != B.end(); bit++)
	{
		B2I[*bit] = i;
		i++;
	}

	// count weighted roots
	int weighted_roots = 0;
	for (vector<Handle*>::iterator bit = B.begin(); bit != B.end(); bit++)
	{
		if ((*bit)->is_root() && (*bit)->get_wi() >= 0)
		{
			weighted_roots++;
		}
	}

	// Resize list of vertices, one for each "Bone" including roots
	// Resize list of edges, one for each bone segment, so excluding roots
	V.resize(B.size(), 3);
	BE.resize(B.size() - BR.size(), 2);
	// Only consider point handles at weighted roots
	P.resize(weighted_roots);
	WI.resize(weighted_roots + BE.rows());
	int e = 0;
	int p = 0;
	// loop over bones
	for (vector<Handle*>::iterator bit = B.begin(); bit != B.end(); bit++)
	{
		// Store position
		V.row(B2I[*bit]) = (*bit)->rest_tip();
		// If not root, then store connectivity
		if (!(*bit)->is_root())
		{
			// Bone edge
			BE(e, 0) = B2I[(*bit)->get_parent()];
			BE(e, 1) = B2I[*bit];
			// Bone edges are expected to have weight indices
			assert((*bit)->get_wi() >= 0);
			WI(P.size() + e) = (*bit)->get_wi();
			e++;
		}
		else if ((*bit)->get_wi() >= 0)
		{
			// Point handle
			P(p) = B2I[*bit];
			WI(p) = (*bit)->get_wi();
			p++;
		}
	}

}




bool boundary_conditionsWrapper(
	const Eigen::MatrixXd & V,
	const Eigen::MatrixXi & Ele,
	const std::vector<Handle*> & BR,
	Eigen::VectorXi & b,
	Eigen::MatrixXd & bc)
{
	using namespace std;
	using namespace Eigen;
	using namespace igl;

	MatrixXd C;
	VectorXi P;
	MatrixXi BE;
	MatrixXi CE;
	VectorXi WI;
	gather_positions_and_connectivity(BR, C, P, BE, WI);

	// Compute boundary conditions in C,P,BE,CE style
	Eigen::MatrixXd bc_temp;
	bool ret = ::boundary_conditions(V, Ele, C, P, BE, CE, b, bc_temp);

	if (!ret)
	{
		return false;
	}
	// But now columns are ordered according to [P,BE]
	bc.resize(bc_temp.rows(), bc_temp.cols());
	slice_into(bc_temp, colon<int>(0, bc.rows() - 1), WI, bc);
	return ret;

}

#endif

void computeWeights(Sample* sample, std::vector<Handle*>& _handles, std::vector<std::vector<float>>& _wieghts)
{
	using namespace std;
	// #V by 3 Matrix of mesh vertex 3D positions
	Eigen::MatrixXd V;

	// #F by 3 Matrix of face (triangle) indices
	Eigen::MatrixXi F;
	// #Tets by 4 Matrix of tet (indices), empty means surface
	Eigen::MatrixXi Tets;
	// Original #V by #original_weights Matrix of per-mesh vertex, per-handle
	// weights unsorted.
	Eigen::MatrixXd OW;
	// Extra weights, #V by #extra_weights Matrix of per-mesh vertex,
	// per-handle weights (probably don't partition unity)
	Eigen::MatrixXd EW;
	
	{
		V.resize(sample->num_vertices(),3);
		pcm::Matrix3X& vertex_matrix = sample->vertices_matrix();
		for (int i = 0; i < sample->num_vertices(); ++i)
		{
			V(i,0) = vertex_matrix(0, i);
			V(i,1) = vertex_matrix(1, i);
			V(i,2) = vertex_matrix(2, i);
		}
		F.resize(sample->num_triangles(), 3);
		for (int i = 0; i < sample->num_triangles(); ++i)
		{
			TriangleType& tt =sample->getTriangle(i);
			F(i,0) =  tt.get_i_vertex(0);
			F(i,1) =  tt.get_i_vertex(1);
			F(i,2) =  tt.get_i_vertex(2);
		}
	}



	//std::vector<std::vector<float > >  V_vec;
	//const std::vector<std::vector<int> >  F_vec;
	const std::string switches;
	//std::vector<std::vector<float > >  TV_vec;
	//std::vector<std::vector<int > >  TT_vec;
	//std::vector<std::vector<int> >  TF_vec;

	verbose("Computing BBW weights\n");
	// Determine which Bones have weights
	bool success = false;// = distribute_weight_indices(skinning->skel->roots);
	//if (!success)
	//{
	//	return;
	//}

	// Boundary faces
	MatrixXi BF;
	//V by 3 Matrix of mesh vertex 3D positions
	Eigen::MatrixXd VT;
	cout << "tetgen begin()" << endl;
	int status = igl::copyleft::tetgen::tetrahedralize(V, F,
		"Ypq100",
		VT,
		Tets,
		BF);
	cout << "tetgen end()" << endl;
	if (BF.rows() != F.rows())
	{
		//assert(BF.maxCoeff() == skinning->F.maxCoeff());
		cout << "^%s: Warning: boundary faces != orignal faces\n"<<endl;
	}
	if (status != 0)
	{
		cout <<
			"************************************************************\n"
			"************************************************************\n"
			"************************************************************\n"
			"************************************************************\n"
			"* ^%s: tetgen failed. Just meshing convex hull\n"
			"************************************************************\n"
			"************************************************************\n"
			"************************************************************\n"
			"************************************************************\n" << endl;
		status =
			igl::copyleft::tetgen::tetrahedralize(
				V, F, "q1.414", VT, Tets, BF);
		assert( F.maxCoeff() < V.rows());
		if (status != 0)
		{
			cout << "^%s: tetgen failed again.\n" << endl;
			return;
		}
	}
	if (1)
		igl::writeMESH("./Keg.mesh", VT, Tets, F);


	Eigen::MatrixXd C; //Node positon
	Eigen::VectorXi P;  //Point handle
	Eigen::MatrixXi BE; //Bone Edge
	Eigen::MatrixXi CE; //Cage Edge
	P.resize(_handles.size());
	C.resize(_handles.size(), 3);
	for (int i = 0; i < _handles.size(); ++i)
	{
		qglviewer::Vec pos = _handles[i]->getLocalPosition();
		C(i, 0) = pos.x;
		C(i, 1) = pos.y;
		C(i, 2) = pos.z;
		P(i) = _handles[i]->handle_idx_;
	}

	// Get boundary conditions
	VectorXi b; //boundary
	MatrixXd bc; //boundary condiction
	igl::boundary_conditions(VT, Tets, C, P, BE, CE, b, bc);
//	boundary_conditionsWrapper(V, Tets, vector<Handle*>(), b, bc);

	//now caculate weight
	igl::BBWData bbw_data;
	bbw_data.active_set_params.max_iter = 10;
	bbw_data.verbosity = 2;
	success = igl::bbw(
		VT,
		Tets,
		b,
		bc,
		bbw_data,
		OW
	);

	if (!success)
	{
		return;
	}

	for (size_t i = 0; i < OW.rows(); i++)
	{
		for (size_t j = 0; j < OW.cols(); j++)
		{
			if (OW(i, j) < 0.0)
				OW(i, j) = 0.0;
		}
	}
	// Normalize weights to sum to one
	OW = (OW.array().colwise() /
		OW.rowwise().sum().array()).eval();
	EW.resize(OW.rows(), 0);
	_wieghts.clear();
	_wieghts.resize(OW.rows());
	for (size_t i = 0; i < OW.rows(); i++)
	{
		for (size_t j = 0; j < OW.cols(); j++)
		{
			_wieghts[i].push_back( OW(i, j)) ;
		}		
	}
	writeWeightToFile(string("./keg.weight"), _wieghts);
	verbose("Computing BBW weights done\n");
}

static void getKbiggest(std::vector<float>& _ori, std::vector<float>& out, std::vector<int>& out_idx ,int _numIndices)
{
	if (_ori.size() <= _numIndices)
		return;
	out.clear();
	out_idx.clear();
	std::deque<float> k_big;
	std::deque<int> k_big_idx;
	for (size_t j = 0; j < _ori.size(); j++)
	{
		if (k_big.size() < _numIndices)
		{
			if (k_big.size() && _ori[j] <= k_big.front())
			{
				k_big.push_front(_ori[j]);
				k_big_idx.push_front(j);
			}
			else if (k_big.size() && _ori[j] >= k_big.back())
			{
				k_big.push_back(_ori[j]);
				k_big_idx.push_back(j);
			}
			else if (k_big.size())
			{
				k_big.push_front(_ori[j]);
				k_big_idx.push_front(j);
				//排序，保持左边比右边小
				//因为除了第一个，后面的已经有序，故使用插入排序,复杂度o(k)
				int m = 0;
				float tmp = k_big[m];
				int tmp_idx = k_big_idx[m];
				int n = m + 1;
				while (n < k_big.size() && tmp > k_big[n])
				{
					k_big[n - 1] = k_big[n];
					k_big_idx[n - 1] = k_big_idx[n];
					n++;
				}
				k_big[n - 1] = tmp;
				k_big_idx[n - 1] = tmp_idx;
			}
			else if (!k_big.size())
			{
				k_big.push_front(_ori[j]);
				k_big_idx.push_front(j);
			}

		}
		else
		{
			if (k_big.size() && _ori[j] >= k_big.back())
			{
				k_big.push_back(_ori[j]);
				k_big_idx.push_back(j);
				k_big.pop_front();
				k_big_idx.pop_front();

			}
			else if (k_big.size() && _ori[j] > k_big.front())
			{

				k_big.pop_front();
				k_big_idx.pop_front();
				k_big.push_front(_ori[j]);
				k_big_idx.push_front(j);
				//排序，保持左边比右边小
				//因为除了第一个，后面的已经有序，故使用插入排序,复杂度o(k)
				int m = 0;
				float tmp = k_big[m];
				int tmp_idx = k_big_idx[m];
				int n = m + 1;
				while (n < k_big.size() && tmp > k_big[n])
				{
					k_big[n - 1] = k_big[n];
					k_big_idx[n - 1] = k_big_idx[n];
					n++;
				}
				k_big[n - 1] = tmp;
				k_big_idx[n - 1] = tmp_idx;

			}


		}

	}
	while (k_big.size())
	{
		out.push_back(k_big.back());
		out_idx.push_back(k_big_idx.back());
		k_big.pop_back();
		k_big_idx.pop_back();
	}



}



void computeWeights(Sample* sample, std::vector<Handle*>& _handles, 
	std::vector<float>& _wieghts,
	std::vector<int>& _wieghts_idx, 
	 int _numIndices)
{
	std::vector<std::vector<float>> weights;
	if (0)
		computeWeights(sample, _handles, weights);
	else
		getWeightFromFile(string("./keg.weight"), weights);
	//
	_wieghts.resize(weights.size() * _numIndices);
	_wieghts_idx.resize(weights.size() * _numIndices);
	for (size_t i = 0; i < weights.size(); i++)
	{
		std::vector<float>& bone_weight = weights[i];
		vector<float> normalize_wieght;
		vector<int> normalize_idx;
		getKbiggest(bone_weight, normalize_wieght, normalize_idx, _numIndices);
		float sum = 0.0f;
		for (float weight : normalize_wieght)
		{
			sum += weight;
		}
		for (float& weight : normalize_wieght)
		{
			weight/=sum;
		}

		for (size_t m = 0; m < _numIndices; m++)
		{
			_wieghts[i * _numIndices + m] = normalize_wieght[m];
			_wieghts_idx[i * _numIndices + m] = normalize_idx[m];
		}
		
	}
}

void writeWeightToFile(std::string& path, std::vector<std::vector<float>>& _weights)
{
	ofstream writer(path);
	if (!_weights.size())
		return;
	writer << _weights.size() << " " << _weights[0].size() << endl;
	for (int i = 0; i < _weights.size(); ++i)
	{
		for (size_t j = 0; j < _weights[i].size(); j++)
		{
			writer << _weights[i][j]<<" ";
		}
		writer << endl;

	}
	writer.close();

}

void getWeightFromFile(std::string& path, std::vector<std::vector<float>>& _weights)
{
	_weights.clear();
	ifstream reader(path);
	int rows = 0,cols = 0;
	reader >> rows >> cols;
	_weights.resize(rows);
	for (int i = 0; i < rows; ++i)
	{
		_weights[i].resize(cols);
		for (size_t j = 0; j < _weights[i].size(); j++)
		{
			reader >> _weights[i][j];
		}
	}
	reader.close();

}
