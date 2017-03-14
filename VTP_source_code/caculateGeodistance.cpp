#include "caculateGeodistance.h"
#include "geodesic_mesh.h"
#include "geodesic_algorithm_exact.h" 

void caculteGeodistance(const std::vector<double>& points, const std::vector<unsigned>& faces,
							int targetVertex,std::vector<float>& distance,int propagate_depth,
							void* drawbackback  ,int drawback_index )
{
	// Build Mesh
	clock_t start = clock();
	geodesic::Mesh mesh;
	mesh.initialize_mesh_data(points, faces);		//create internal mesh data structure including edges
	{


	std::cout << "Build Mesh Success..." << std::endl;
	clock_t stop = clock();
	float m_time_consumed = (static_cast<double>(stop) - static_cast<double>(start)) / CLOCKS_PER_SEC;
	std::cout<<"Build Mesh "<<m_time_consumed<<std::endl;
	
	}
	{
	clock_t start = clock();
	geodesic::GeodesicAlgorithmExact algorithm(&mesh);
	// Propagation


	algorithm.propagate(targetVertex ,propagate_depth ,drawbackback, 0 );	//cover the whole mesh
	clock_t stop = clock();
	float m_time_consumed = (static_cast<double>(stop) - static_cast<double>(start)) / CLOCKS_PER_SEC;
	std::cout<<"propagate "<<m_time_consumed<<std::endl;
	// Print Statistics
	std::cout << std::endl;
	algorithm.print_statistics(); 
	}

	distance.resize( mesh.vertices().size());
	for(unsigned i=0; i<mesh.vertices().size(); ++i)
	{
		distance[i] = mesh.vertices()[i].geodesic_distance();
		if (distance[i] > 1000)
		{
			distance[i] = 1000;
		}
	}
}
