#include "example_mesh_ctrl.h"
#include "ebpd/ExampleWeightSover.h"
#include "VTP_source_code/caculateGeodistance.h"
#include "VTP_source_code/geodesic_mesh.h"
#include "toolbox/maths/transfo.hpp"
#include "toolbox/maths/quat_cu.hpp"
#include "toolbox/maths/vec3.hpp"
#include "toolbox/maths/color.hpp"
#include "main_window.h"
#include "sample_set.h"
#include "sample.h"
#include "triangle.h"
#include "file_io.h"
#include "color_table.h"
#include "GlobalObject.h"
#include "manipulate_object.h"
#include "LBS_Control.h"
#include <sstream>

#define  Debug_Time true; 

std::vector<float> g_inputVertices;
std::vector<int> g_faces;
std::vector<Tbx::Transfo> g_transfos;
std::vector<std::vector<TransAndRotation>> g_transfos_2;
std::vector<int> g_boneWightIdx;
std::vector<float> g_boneWeights;
std::vector<float> g_exampleWeights; //frist sort example all of the weight of vertex,then next example
std::vector<MeshControl*> g_MeshControl;

int g_numExample =0;
int g_numBone = 0;
int g_numVertices = 0;
int g_numIndices = 4; //represents the max number of infuence of bone in vertex
static void exportObj( const std::vector<float>& inputVertice ,std::vector<int>&  faces, std::vector<Tbx::Color>& colors, std::string file_paths );
/*

inputVertices = numVertices
transfosOfExamples = numBone x numExample
boneWeights = numIndices x numVertices
boneWightIdx = numIndices x numVertices
exampleWeights = numVertices x numExample

*/
ExampleSover* exampleSolver = NULL;

void rebuildExampleSover()
{
	if(exampleSolver)
		delete exampleSolver;
	exampleSolver = new ExampleSover( g_inputVertices,g_numVertices,
		g_transfos,g_numBone,g_numExample,g_numIndices,
		g_boneWeights,g_boneWightIdx);

}



void GetRigFromFile(std::vector<Tbx::Transfo>& transfos ,
					std::vector<int>& boneWightIdx ,std::vector<float>& boneWeights ,
					int& _numVertices, int& _numBone, int& _numExample,int& _numbIndices,
					std::string file_paths)
{
	using namespace std;
	ifstream ifs(file_paths);


	int numVertices;
	int numExamples;
	int numIndices;
	int numBones;

	const int LINE_LENGTH = 200;
	char str[LINE_LENGTH];
	//skip #
	while (ifs.getline( str,LINE_LENGTH))
	{
		//		cout<<" read from file"<<str<<std::endl;
		if( str[0] !='#')break;
	}
	stringstream s( str);
	s>>numExamples>>numBones;
	//	cout<<numExamples<<" "<<numBones<<endl;
	transfos.resize(numExamples * numBones);

	for (int s = 0; s < numExamples; ++s)
	{
		int curexam;
		ifs>>curexam;
		for (int b = 0; b < numBones; ++b)
		{
			int curbone;
			ifs>>curbone;

			Tbx::Vec3 t;
			float r_x,r_y,r_z,r_w;
			ifs>>r_x>>r_y>>r_z>>r_w;
			ifs>>t.x>>t.y>>t.z;
			Tbx::Quat_cu r(r_w,r_x,r_y,r_z); 
			Tbx::Transfo at(r.to_matrix3(),t);
			transfos[s * numBones + b] = at;
			//			cout<<r.x<<" "<<r.y<<" "<<r.z<<" "<<r.w<<" ";
			//			cout<<t.x<<" "<<t.y<<" "<<t.z<<endl;
		}
		//		cout<<endl;
	}

	//skip #
	ifs.getline( str,LINE_LENGTH);
	while (ifs.getline( str,LINE_LENGTH))
	{
		//		cout<<" read from file"<<str<<std::endl;
		if( str[0] !='#')break;
	}
	s = stringstream("");
	s<<str;
	s>>numVertices>>numBones>>numIndices;
	//	cout<<numVertices<<" "<<numBones<<std::endl;
	boneWeights.resize(numVertices * numIndices, 0.0f);
	boneWightIdx.resize(numVertices * numIndices, 0);
	for (int v = 0; v < numVertices; ++v)
	{
		int curvertice;
		ifs>>curvertice;
		//		cout<<curvertice<<" ";
		for (int b = 0; b < numIndices; ++b)
		{
			//int  curbone;
			//ifs>>curbone;
			//			cout<<curbone<<" ";
			ifs>>boneWightIdx[v * numIndices + b];
			ifs>>boneWeights[v * numIndices + b];
			//			cout<<result.weight[v * numIndices + b]<<" ";

		}
		//		cout<<endl;

	}
	_numVertices = numVertices;
	_numBone = numBones;
	_numExample = numExamples;
	_numbIndices = numIndices;

	ifs.close();
}


void WriteAnimationToFile(std::string file_paths,
						  const std::vector<float>& OutputVetices, const std::vector<int>& faces )
{
	using namespace  std;

	stringstream s;
	int numVertices = OutputVetices.size()/3;
	int numFaces = faces.size()/3;
	std::string fullpath;
	s>>fullpath;
	ofstream ofs(fullpath);
	ofs.setf( ios::fixed ,ios::floatfield);
	ofs.precision(6);
	for (int v = 0; v < numVertices; ++v)
	{
		ofs<<"v "<<OutputVetices[3*v+0]<<" "<<OutputVetices[3*v+1]<<" "<<OutputVetices[3*v+2]<<std::endl;
	}
	for( int i = 0 ;i< numFaces ;++i)
	{
		ofs<<"f "<< faces[ 3 * i + 0]+1<<" "<<faces[ 3 * i + 1]+1<<" "<<faces[ 3 * i + 2]+1<<std::endl;
	}

	ofs.close();


}


bool genetatedVertice( std::vector<float>& OutputVetices , const std::vector<float>& inputVertices, int numVertices,
					  const std::vector<Tbx::Transfo>& transfosOfExamples,int numBone, int numExample,int numbIndices,
					  const std::vector<float>& boneWeights,
					  const std::vector<int>& boneWightIdx,
					  const std::vector<float> exampleWeights)

{
	bool isQlerp = true;
	assert( inputVertices.size() ==  3* numVertices );
	OutputVetices.resize( inputVertices.size());

	for( int i_vertex = 0; i_vertex < numVertices; ++i_vertex)
	{
		float x = inputVertices[3*i_vertex];
		float y = inputVertices[3*i_vertex+1];
		float z = inputVertices[3*i_vertex+2];
		Tbx::Point3 cur_point(x,y,z);
		Tbx::Point3 acc_point;


		for (int i_indice= 0; i_indice < numbIndices ;++i_indice)
		{
			int i_bone = boneWightIdx[i_vertex*numbIndices+i_indice];

			Tbx::Vec3 acc_translate;
			Tbx::Mat3 acc_rotate;
			Tbx::Quat_cu acc_rotate_quat;
			for (int i_example = 0 ;i_example< numExample ;++i_example)
			{

				const Tbx::Transfo& transfo = transfosOfExamples[i_example*numBone+i_bone];

				Tbx::Vec3& translate = transfo.get_translation();
				Tbx::Mat3& rotate = transfo.get_mat3();
				Tbx::Quat_cu rotate_quat(transfo);
				float exmaple_weight = exampleWeights[i_example* numVertices + i_vertex];
				acc_translate += translate* exmaple_weight;
				if(0==i_example)
				{
					//let it identity
					acc_rotate_quat = rotate_quat*exmaple_weight;
				}
				else
				{
					acc_rotate_quat =  acc_rotate_quat + rotate_quat*exmaple_weight;
				}
			}
			if(isQlerp)
				acc_rotate_quat = acc_rotate_quat/acc_rotate_quat.norm(); 
			acc_rotate = acc_rotate_quat.to_matrix3();
			Tbx::Transfo final_transfo(acc_rotate ,acc_translate );
			float bone_weight = boneWeights[i_vertex*numbIndices+i_indice];
			acc_point = acc_point + bone_weight * (final_transfo*cur_point);
		}

		OutputVetices[3*i_vertex] = acc_point.x;
		OutputVetices[3*i_vertex+1] = acc_point.y;
		OutputVetices[3*i_vertex+2] = acc_point.z;
	}


	return true;
}

typedef Tbx::Vec3 ColorType2;
typedef Tbx::Vec3 PointType2;
typedef Tbx::Vec3 NormalType2;
static void skip_this_line(FILE* fd)
{
	int ret =0;
	while( ret = fgetc(fd)!= '\n'&& ret !=EOF );

}
static bool importObj(std::vector<float>& inputVertice ,std::vector<int>&  faces, std::string file_paths)
{
	inputVertice.clear();
	faces.clear();
	FILE* in_file = fopen( file_paths.c_str(), "r");

	if(in_file ==NULL)
		return false;
	char pref[3];

	FILE* fd = in_file ;
	//	FILE* fd = get_mesh_model_scale( filename.c_str(), &(new_sample->n_vertex),&(new_sample->n_normal),&(new_sample->n_triangle) );
	//	Logger<<"nvertex "<<new_sample->n_vertex<<"nnormal "<<new_sample->n_normal<<"ntriangle"<<new_sample->n_triangle<<std::endl;
	if(fd)
	{
		float vx,vy,vz,nx,ny,nz,cx,cy,cz;
		std::vector<PointType2> v;
		std::vector<ColorType2> cv;
		std::vector<NormalType2> nv;
		std::vector<int> ttv;
		while( fscanf(fd ,"%2s",pref) != EOF)
		{
			if(strcmp(pref,"v")==0)
			{

				fscanf(fd, "%f %f %f",&vx,&vy,&vz);
				v.push_back( PointType2(vx,vy,vz));
			}else if(strcmp(pref,"vn") ==0)
			{
				fscanf(fd,"%f %f %f",&(nx),&(ny),&(nz));
				nv.push_back( NormalType2(nx,ny,nz));
			}else if( strcmp(pref,"f")==0)
			{   

				for(int i_v = 0; i_v<3 ; i_v++)
				{
					int i_temp_v;
					int i_temp_n;
					fscanf(fd, "%d",&i_temp_v);
					ttv.push_back( i_temp_v - 1);
					if((pref[0] = (char)getc(fd)) ==' ')
					{
						while( (pref[0] = (char)getc(fd)) ==' ');
						if( pref[0] == '\n'||pref[0]==EOF )
						{

						}else
						{
							ungetc(pref[0],fd);
						}
					}
					else if(pref[0] =='/')
					{

						if( (pref[0] = (char)getc(fd))!='/')
						{
							//while((char)get(fd) ==" ");
							ungetc(pref[0],fd);
							int nouse;
							fscanf(fd ,"%d",&nouse);
							if( (pref[0] = (char)getc(fd))!='/')
							{
								fscanf(fd,"%d",&i_temp_n);
								//tt->set_i_normal(i_v ,i_temp_n);
							}

						}else{
							fscanf(fd,"%d",&i_temp_n);
							//tt->set_i_normal(i_v ,i_temp_n);
						}								

					}

				}

				//					ttv.push_back(tt);
			}else if(pref[0] =='#')
			{
				skip_this_line(fd);
			}
			else skip_this_line(fd);
		}
		for(int i = 0 ; i< v.size();++i)
		{
			inputVertice.push_back(v[i].x );
			inputVertice.push_back(v[i].y );
			inputVertice.push_back(v[i].z );

		}
		for(int i = 0 ; i< ttv.size();++i)
		{

			faces.push_back( ttv[i] );

		}
		//	new_sample->add_triangle(triangle_array= ttv;
		fclose(fd);
	}else
	{		
		return false;
	}
	return true;

}

static void exportObj( const std::vector<float>& inputVertice ,std::vector<int>&  faces, std::vector<Tbx::Color>& colors, std::string file_paths )
{
	using namespace std;
	static int count = 0;
	static int frame = 0;
	count++;

	frame++;
	count = 0;
	int numVertices = inputVertice.size()/3;
	int numFaces = faces.size();
	stringstream s;
	s<<file_paths<<frame<<".obj";
	std::string fullpath;
	s>>fullpath;
	ofstream ofs(file_paths);
	ofs.setf( ios::fixed ,ios::floatfield);
	ofs.precision(6);

	bool hasColor = false;
	if ( colors.size() == inputVertice.size()/3)
	{
		hasColor = true;
	}
	if(hasColor)
	{
		ofs<<"COFF"<<endl;
		ofs<< inputVertice.size()/3 <<" "<<faces.size()/3 <<" "<< 0 <<endl;
	}

	for (int v = 0; v < numVertices; ++v)
	{
		if(!hasColor)
			ofs<<"v "<<inputVertice[3*v+0]<<" "<<inputVertice[3*v+1]<<" "<<inputVertice[3*v+2]<<std::endl;
		else
			ofs<<"v "<<inputVertice[3*v+0]<<" "<<inputVertice[3*v+1]<<" "<<inputVertice[3*v+2]<<" "<<(int)(255*colors[v].r)<<" "<<(int)(255*colors[v].g)<<" "<<(int)(255*colors[v].b)<<std::endl;
	}
	for( int i = 0 ;i< numFaces/3 ;++i)
	{
		ofs<<"f "<< faces[3*i]+1 <<" "<<faces[3*i+1]+1 <<" "<<faces[3*i+2]+1<<std::endl;
	}

	ofs.close();



}



void exportObj( const std::vector<float>& inputVertice ,std::vector<int>&  faces,std::string file_paths)
{
	std::vector<Tbx::Color> nocolor;
	exportObj( inputVertice ,faces,nocolor,file_paths);
}

void drawBack( geodesic::Mesh& mesh , int idex)
{
	std::string path = "D:/mprojects/EBPD/ebpd/VTP_source_code/data/";
	std::vector<float> distance;

	distance.resize( mesh.vertices().size());
	for(unsigned i=0; i<mesh.vertices().size(); ++i)
	{
		distance[i] = mesh.vertices()[i].geodesic_distance();
	}



	std::stringstream ss;
	ss = std::stringstream("");
	ss<<path<< idex<<"_geodistance.obj";

	std::vector<Tbx::Color> colors(distance.size());
	float maxdistance = -0.1f;
	for (int j = 0; j < distance.size(); j++)
	{
		float distance_target = distance[j];
		if(distance_target<0) distance_target = 0.0f;
		if(distance_target>10000) distance_target = 0.0f;
		if(distance_target > maxdistance)
			maxdistance = distance_target;
	}
	for (int j = 0; j < distance.size(); j++)
	{

		float distance_target = distance[j];
		float infuence = 1- distance_target/maxdistance;
		if( infuence<0) infuence = 0;
		Tbx::Color c = Tbx::Color::heat_color( infuence );
		//if( distance_target/maxdistance < 0.125)
		//	c = Tbx::Color(1.0f,1.0f,1.0f,1.0f);
		if( distance_target/maxdistance < 0.0001)
			c = Tbx::Color(1.0f,1.0f,1.0f,1.0f);
		colors[j] = c;
	}


	std::string com_path;
	ss>>com_path;
	exportObj( g_inputVertices,g_faces, colors ,com_path);



}

static void findKNearest( int nearDepth ,std::map<int,std::map<int,float> >& distanceOfVertex ,std::vector<int>& targetVertexIdx)
{
	// std::vector<int> a;
	//for each (auto i_vertex in a)
	//{

	//}
	std::vector<float> distance;
	std::vector<double> inputVertice(g_inputVertices.size());
	std::vector<unsigned> faces(g_faces.size());
	for (int i = 0; i < g_inputVertices.size(); i++)
	{
		inputVertice[i] = g_inputVertices[i];
	}
	for (int i = 0; i < g_faces.size(); i++)
	{
		faces[i] = g_faces[i];
	}
	clock_t start = clock();
	for( int i_vertx:targetVertexIdx)
	{
		caculteGeodistance( inputVertice,faces,i_vertx,distance,nearDepth,drawBack,0);
		for (int i = 0; i < distance.size(); i++)
		{
			distanceOfVertex[i_vertx][i] = distance[i];
		}
		
	}
	clock_t stop = clock();
	float m_time_consumed = (static_cast<double>(stop) - static_cast<double>(start)) / CLOCKS_PER_SEC;
	std::cout<<"caculteGeodistance "<<m_time_consumed<<std::endl;
	//for each (int i_vertex in targetVertexIdx)
	//{

	//}
	//int i_vertx = targetVertexIdx;


}

inline float kernelFunction(float kernelRadius , float distance)
{
	if( kernelRadius <=0)return 0.0f;
	float result;
	if( distance < kernelRadius)
	{
		float ratio = distance/kernelRadius;
		result = 2*ratio*ratio*ratio -3* ratio*ratio+1;
	}else
	{
		result = 0.0f;
	}
	return result;

}

static void writeColorGeoDistance( std::vector<int>& i_vertexs, std::string path)
{
	std::map<int,std::map<int,float> > distanceOfVertex;
	findKNearest( 100, distanceOfVertex ,i_vertexs);

	std::stringstream ss;

	for( auto element : distanceOfVertex)
	{
		int i_vertex = element.first;
		ss = std::stringstream("");
		ss<<path<< i_vertex<<"_geodistance.obj";
		std::map<int,float>& distances = element.second;
		std::vector<Tbx::Color> colors(g_numVertices);
		float maxdistance = -0.1f;
		for ( auto element2 : distances)
		{
			int i_target = element2.first;
			float distance_target = element2.second;
			if(distance_target<0) distance_target = 0;
			if(distance_target>10000) distance_target = 0.0f;
			if(distance_target > maxdistance)
				maxdistance = distance_target;
			
		}
		for ( auto element2 : distances)
		{
			int i_target = element2.first;
			float distance_target = element2.second;
			float infuence = 1- distance_target/maxdistance;
			if( infuence<0) infuence = 0;
			Tbx::Color c = Tbx::Color::heat_color( infuence );
			//if( distance_target/maxdistance < 0.125)
			//	c = Tbx::Color(1.0f,1.0f,1.0f,1.0f);
			if( distance_target/maxdistance < 0.0001)
				c = Tbx::Color(1.0f,1.0f,1.0f,1.0f);
			colors[i_target] = c;
		}
		std::string com_path;
		ss>>com_path;
		exportObj( g_inputVertices,g_faces, colors ,com_path);
	}
	



}

static void propagateExampleWeight(std::map<int, std::vector<float> >& delta_exampleWeightsOfVertex ,std::vector<float>& delta_exampleWeights )
{
#ifdef Debug_Time
	clock_t start = clock();
#endif // DEBUG
	
	delta_exampleWeights.resize(g_numVertices*g_numExample,0.0f);
	std::vector<int> targetVertexIdx;
	auto iter = delta_exampleWeightsOfVertex.begin();
	for (int i = 0; iter != delta_exampleWeightsOfVertex.end(); i++,++iter)
	{
		int i_vertex = iter->first;
		const std::vector<float>& delta_example  = iter->second;
		for (int j = 0; j < g_numExample; j++)
		{
			delta_exampleWeights[i_vertex+j*g_numVertices] = delta_example[j];
		}
		targetVertexIdx.push_back(i_vertex);
	}
	int depth =100;
	float kernelRadius = 1.5f; //control radius of the kernel
	std::map<int,std::map<int,float> > distanceOfVertex;
	{

	#ifdef Debug_Time
		clock_t start = clock();
	#endif // DEBUG
	findKNearest( depth , distanceOfVertex ,targetVertexIdx);
	#ifdef Debug_Time
		clock_t stop = clock();
		float m_time_consumed = (static_cast<double>(stop) - static_cast<double>(start)) / CLOCKS_PER_SEC;
		std::cout<<"findKNearest "<<m_time_consumed<<std::endl;
	#endif // DEBUG

	}

	for( auto element : distanceOfVertex)
	{
		int i_vertex = element.first;
		std::map<int,float>& distances = element.second;
		for ( auto element2 : distances)
		{
			int i_target = element2.first;
			float distance_target = element2.second;
			float result = kernelFunction( kernelRadius,distance_target);

			for (int j = 0; j < g_numExample; j++)
			{
				if (i_target == i_vertex)
					continue;
				delta_exampleWeights[i_target+j*g_numVertices] += result*delta_exampleWeights[i_vertex+j*g_numVertices];
			}
		}
	}
#ifdef Debug_Time
	clock_t stop = clock();
	float m_time_consumed = (static_cast<double>(stop) - static_cast<double>(start)) / CLOCKS_PER_SEC;
	std::cout<<"propagateExampleWeight "<<m_time_consumed<<std::endl;
#endif // DEBUG

	

}




void Example_mesh_ctrl::genertateVertices(std::string _file_paths,std::string name)
{
	std::string input_mesh_path = _file_paths+name+".obj";
	std::string output_mesh_path = _file_paths+name+"init_out.obj";
	std::string rig_path = _file_paths+name+".rig";
	std::string file_paths;
	using namespace std;
	importObj( g_inputVertices,g_faces,input_mesh_path);
	g_numVertices = g_inputVertices.size()/3;

	GetRigFromFile(g_transfos ,
		g_boneWightIdx ,g_boneWeights ,g_numVertices,g_numBone, g_numExample,g_numIndices,
		rig_path);
	std::vector<float> OutputVetices;
	if(!g_exampleWeights.size())
	{
		g_exampleWeights.resize( g_numVertices*g_numExample,0.0f);
		for (int i = 0; i < g_numVertices; i++)
		{
			g_exampleWeights[i+0*g_numVertices]  = 1.0f;
			g_exampleWeights[i+1*g_numVertices]  = 0.0f;
			g_exampleWeights[i+2*g_numVertices]  = 0.0f;
		}
	}
	genetatedVertice(  OutputVetices , g_inputVertices, g_numVertices,
		g_transfos ,g_numBone, g_numExample,g_numIndices,
		g_boneWeights,
		g_boneWightIdx,
		g_exampleWeights);
	exportObj( OutputVetices,g_faces,output_mesh_path);
	std::vector<int> i_vertexs;
	//i_vertexs.push_back(0);
	i_vertexs.push_back(8);
	i_vertexs.push_back(2850);
	writeColorGeoDistance( i_vertexs, _file_paths+name);




	rebuildExampleSover();
	std::map<int,Tbx::Vec3> delta_xi;
//	delta_xi[0] = Tbx::Vec3(-0.217,0.3,-0.0);  //test
	delta_xi[8] = Tbx::Vec3(-0.1825,0.56,-0.0);  //test
	//delta_xi[20] = Tbx::Vec3(0.217,0.3,-0.0);  //test
	delta_xi[2850] = Tbx::Vec3(-0.877,0.8916,-0.0011);
	std::map<int, std::vector<float> > delta_exampleWeightsOfVertex; 
	std::map<int, std::vector<float> > ori_exampleWeights;
	for (int i_vertex = 0; i_vertex < g_numVertices; i_vertex++)
	{
		ori_exampleWeights[i_vertex] = std::vector<float>(g_numExample);
		for (int i_example = 0; i_example < g_numExample; i_example++)
		{
			ori_exampleWeights[i_vertex][i_example] =  g_exampleWeights[i_vertex + i_example*g_numVertices];
		}
	}


	if(exampleSolver)
		exampleSolver->SolveVertices(delta_xi,delta_exampleWeightsOfVertex,ori_exampleWeights);
	std::vector<float> delta_exampleWeights;
	propagateExampleWeight(delta_exampleWeightsOfVertex,delta_exampleWeights);

	float lamda = 0.5f;
	int fade_count = 5;
	for (int i = 0; i < fade_count; i++)
	{
		for (int i_vertex = 0; i_vertex < g_numVertices; i_vertex++)
		{
			float weight_sum = 0.0f;
			for (int i_example = 0; i_example < g_numExample; i_example++)
			{
				g_exampleWeights[i_vertex + i_example*g_numVertices] += (1-lamda)* delta_exampleWeights[i_vertex + i_example*g_numVertices];
				 weight_sum += g_exampleWeights[i_vertex + i_example*g_numVertices];
			}
			for (int i_example = 0; i_example < g_numExample; i_example++)
			{
				g_exampleWeights[i_vertex + i_example*g_numVertices] /= weight_sum; //clamp example between 0 to 1;

			}
		}
		for (int i_vertex = 0; i_vertex < g_numVertices; i_vertex++)
		{
			for (int i_example = 0; i_example < g_numExample; i_example++)
			{
				delta_exampleWeights[i_vertex + i_example*g_numVertices] *= lamda;
			}
		}




		genetatedVertice(  OutputVetices , g_inputVertices, g_numVertices,
			g_transfos ,g_numBone, g_numExample,g_numIndices,
			g_boneWeights,
			g_boneWightIdx,
			g_exampleWeights);
		stringstream ss;
		ss<<_file_paths<<name<<"_out"<<i<<".obj";
		string outputpath;
		ss>>outputpath;
		exportObj( OutputVetices,g_faces,outputpath);
	}

	

}

// load mesh and its exmaple files into scene for manipulation in the future
void Example_mesh_ctrl::load_example(std::string _file_paths, std::string name)
{
	for (int i = 0; i < g_MeshControl.size(); i++)
	{
		delete  g_MeshControl[i];
	}
	g_MeshControl.clear();
	std::string input_mesh_path = _file_paths + name + ".obj";
	std::string output_mesh_path = _file_paths + name + "init_out.obj";
	std::string rig_path = _file_paths + name + ".rig";
	std::string file_paths;
	using namespace std;
	//load first example
	importObj(g_inputVertices, g_faces, input_mesh_path);
	Sample* new_sample = FileIO::load_point_cloud_file(input_mesh_path, FileIO::OBJ);
	int sample_idx = 0;
	qglviewer::Vec translate(0.0f,0.0f,-1.4f);
	qglviewer::Vec translate_interval(0.0f,0.0f,1.0f);

	//load other example
	GetRigFromFile(g_transfos,
		g_boneWightIdx, g_boneWeights, g_numVertices, g_numBone, g_numExample, g_numIndices,
		rig_path);

	//convert format
	g_transfos_2.clear();
	g_transfos_2.resize(g_numExample);
	for (int i_example = 0; i_example < g_numExample; ++i_example)
	{
		for (int i = 0; i < g_numBone; ++i)
		{
			Tbx::Vec3 translate = g_transfos[i_example * g_numBone +i].get_translation();
			Tbx::Mat3 rotation = g_transfos[i_example * g_numBone + i].get_mat3();
			Tbx::Quat_cu rotate_quat(rotation);
			TransAndRotation translate_and_rotate;
			translate_and_rotate.translation_ = qglviewer::Vec(translate.x, translate.y, translate.z);
			translate_and_rotate.rotation_ = qglviewer::Quaternion(rotate_quat.i(), rotate_quat.j(), rotate_quat.k(), rotate_quat.w());
			g_transfos_2[i_example].push_back(translate_and_rotate);
		}

	}

	
	if (new_sample != nullptr)
	{
		new_sample->getFrame().translate(translate + sample_idx*translate_interval);
		new_sample->setLoaded(true);
		new_sample->set_color(Color_Utility::span_color_from_table(sample_idx));
		SampleSet& smpset = (*Global_SampleSet);
		smpset.push_back(new_sample);
		new_sample->smpId = sample_idx;
		MeshControl* p_mesh_control = new MeshControl(*new_sample);
		p_mesh_control->bindControl(g_inputVertices , g_transfos_2[sample_idx], g_boneWeights, g_boneWightIdx, g_numIndices, g_numBone, g_numVertices, true);
		g_MeshControl.push_back(p_mesh_control);
		sample_idx++;

	}

	std::vector<float> OutputVetices;
	if (1)
	{
		g_exampleWeights.clear();
		g_exampleWeights.resize(g_numVertices*g_numExample, 0.0f);

		for (int j = 1; j < g_numExample; ++j)
		{
			g_exampleWeights.clear();
			g_exampleWeights.resize(g_numVertices*g_numExample, 0.0f);
			for (int i = 0; i < g_numVertices; i++)
			{
				g_exampleWeights[i + j * g_numVertices] = 1.0f;
			}
			//assume the last sample to be what we will manipulate
			genetatedVertice(OutputVetices, g_inputVertices, g_numVertices,
				g_transfos, g_numBone, g_numExample, g_numIndices,
				g_boneWeights,
				g_boneWightIdx,
				g_exampleWeights);
			Sample* sample_manipulate = new Sample();
			if (sample_manipulate != nullptr)
			{
				sample_manipulate->getFrame().translate(translate+ sample_idx*translate_interval);
				for (int i = 0; i < OutputVetices.size() / 3; ++i)
				{
					pcm::NormalType normal; normal << 1.0f, 0.0f, 0.0f;						
					pcm::ColorType  color = Color_Utility::span_color_from_table(sample_idx);
					sample_manipulate->add_vertex(
						pcm::PointType(OutputVetices[3 * i + 0], OutputVetices[3 * i + 1], OutputVetices[3 * i + 2]),
						normal,
						color);

				}
				for (int i = 0; i < g_faces.size() / 3; ++i)
				{
					TriangleType* tt = new TriangleType(*sample_manipulate);
					tt->set_i_vetex(0, g_faces[3 * i + 0]);
					tt->set_i_vetex(1, g_faces[3 * i + 1]);
					tt->set_i_vetex(2, g_faces[3 * i + 2]);
					tt->set_i_normal(0, g_faces[3 * i + 0]);
					tt->set_i_normal(1, g_faces[3 * i + 1]);
					tt->set_i_normal(2, g_faces[3 * i + 2]);
					sample_manipulate->add_triangle(*tt);
					delete tt;
				}


				sample_manipulate->setLoaded(true);
				sample_manipulate->set_color(Color_Utility::span_color_from_table(sample_idx));
				sample_manipulate->build_kdtree();
				Box box = sample_manipulate->getBox();
				ScalarType dia_distance = box.diag();
				SampleSet& smpset = (*Global_SampleSet);
				smpset.push_back(sample_manipulate);
				sample_manipulate->smpId = sample_idx;
				MeshControl* p_mesh_control = new MeshControl(*sample_manipulate);
				p_mesh_control->bindControl(g_inputVertices, g_transfos_2[sample_idx], g_boneWeights, g_boneWightIdx,
					g_numIndices, g_numBone, g_numVertices, true);
				g_MeshControl.push_back(p_mesh_control);
				sample_idx++;
			}
		}
	}


	//assume the last sample to be what we will manipulate,set the weight to be initial
	g_exampleWeights.clear();
	g_exampleWeights.resize(g_numVertices*g_numExample, 0.0f);
	for (int j = 0; j < 1; ++j)
	{
		for (int i = 0; i < g_numVertices; i++)
		{
			g_exampleWeights[i + j * g_numVertices] = 1.0f;
		}
	}
	genetatedVertice(OutputVetices, g_inputVertices, g_numVertices,
		g_transfos, g_numBone, g_numExample, g_numIndices,
		g_boneWeights,
		g_boneWightIdx,
		g_exampleWeights);
	Sample* sample_manipulate = new Sample();
	if (sample_manipulate != nullptr)
	{
		sample_manipulate->getFrame().translate(translate + sample_idx*translate_interval);
		for (int i = 0; i < OutputVetices.size() / 3; ++i)
		{
			pcm::NormalType normal; normal << 1.0f, 0.0f, 0.0f;
			pcm::ColorType  color = Color_Utility::span_color_from_table(sample_idx);
			sample_manipulate->add_vertex(
				pcm::PointType(OutputVetices[3 * i + 0], OutputVetices[3 * i + 1], OutputVetices[3 * i + 2]),
				normal,
				color);

		}
		for (int i = 0; i < g_faces.size() / 3; ++i)
		{
			TriangleType* tt = new TriangleType(*sample_manipulate);
			tt->set_i_vetex(0, g_faces[3 * i + 0]);
			tt->set_i_vetex(1, g_faces[3 * i + 1]);
			tt->set_i_vetex(2, g_faces[3 * i + 2]);
			tt->set_i_normal(0, g_faces[3 * i + 0]);
			tt->set_i_normal(1, g_faces[3 * i + 1]);
			tt->set_i_normal(2, g_faces[3 * i + 2]);
			sample_manipulate->add_triangle(*tt);
			delete tt;
		}


		sample_manipulate->setLoaded(true);
		sample_manipulate->set_color(Color_Utility::span_color_from_table(sample_idx));
		sample_manipulate->build_kdtree();
		Box box = sample_manipulate->getBox();
		ScalarType dia_distance = box.diag();
		SampleSet& smpset = (*Global_SampleSet);
		smpset.push_back(sample_manipulate);
		sample_manipulate->smpId = sample_idx;
		sample_idx++;
	}
	for (int i = 0; i < g_MeshControl.size(); ++i)
	{
		g_MeshControl[i]->updateSample();
	}

	rebuildExampleSover();
}
void Example_mesh_ctrl::setupExample(std::string _file_paths,std::string name)
{
	std::string input_mesh_path = _file_paths+name+".obj";
	std::string output_mesh_path = _file_paths+name+"init_out.obj";
	std::string rig_path = _file_paths+name+".rig";
	std::string file_paths;
	using namespace std;
	importObj( g_inputVertices,g_faces,input_mesh_path);
	g_numVertices = g_inputVertices.size()/3;

	GetRigFromFile(g_transfos ,
		g_boneWightIdx ,g_boneWeights ,g_numVertices,g_numBone, g_numExample,g_numIndices,
		rig_path);
	std::vector<float> OutputVetices;
	if(!g_exampleWeights.size())
	{
		g_exampleWeights.resize( g_numVertices*g_numExample,0.0f);
		for (int i = 0; i < g_numVertices; i++)
		{
			g_exampleWeights[i+0*g_numVertices]  = 1.0f;
			g_exampleWeights[i+1*g_numVertices]  = 0.0f;
			g_exampleWeights[i+2*g_numVertices]  = 0.0f;
		}
	}
	rebuildExampleSover();
}

void Example_mesh_ctrl::genertateVertices(std::vector<float>& inputVertices ,std::vector<int>& faces, const std::vector<int>& vertex_idexs, const std::vector<float>& impulses ,float mass,float delta_t,float belta)
{
	static int count=0;
	if ( vertex_idexs.size() < 1 || 1 == count)
	{
		genetatedVertice(  inputVertices , g_inputVertices, g_numVertices,
			g_transfos ,g_numBone, g_numExample,g_numIndices,
			g_boneWeights,
			g_boneWightIdx,
			g_exampleWeights);
		faces = g_faces;
		return;
	}
	count++;
	if( 1 ==count)
	{
		std::cout<<std::endl;
		std::cout<<"input impluse "<<std::endl;
	}
	std::map<int,Tbx::Vec3> delta_xi;
	for (int i = 0; i < vertex_idexs.size(); i++)
	{
		Tbx::Vec3 impluse( impulses[3*i],impulses[3*i+1],impulses[3*i+2]);
		int i_vertex = vertex_idexs[i];
		if(impluse.norm() > 1e-6 )
			delta_xi[i_vertex] = delta_t/mass* max( impluse.norm()-belta, 0 )*impluse/impluse.norm();
		else
			delta_xi[i_vertex] = 0.0f;
		if( 1 ==count)
		{
			std::cout<< i_vertex<<" "<< "impluse norm "<<impluse.norm()<<" "<<impulses[3*i]<<" "<<impulses[3*i+1]<<" "<<impulses[3*i+2]<<std::endl;
			delta_xi[i_vertex].print();
		}

		//if(impluse.norm()>belta )
		//{
		//	std::cout<< i_vertex<< "impluse norm "<<impluse.norm()<<std::endl;
		//}
	}
	//	delta_xi[0] = Tbx::Vec3(-0.217,0.3,-0.0);  //test
	//	delta_xi[8] = Tbx::Vec3(-0.1825,0.56,-0.0);  //test
	//delta_xi[20] = Tbx::Vec3(0.217,0.3,-0.0);  //test
	if(!g_exampleWeights.size())
	{
		g_exampleWeights.resize( g_numVertices*g_numExample,0.0f);
		for (int i = 0; i < g_numVertices; i++)
		{
			g_exampleWeights[i+0*g_numVertices]  = 1.0f;
			g_exampleWeights[i+1*g_numVertices]  = 0.0f;
			g_exampleWeights[i+2*g_numVertices]  = 0.0f;
		}
	}

	std::map<int, std::vector<float> > delta_exampleWeightsOfVertex; 
	std::map<int, std::vector<float> > ori_exampleWeights;
	for (int i_vertex = 0; i_vertex < g_numVertices; i_vertex++)
	{
		ori_exampleWeights[i_vertex] = std::vector<float>(g_numExample);
		for (int i_example = 0; i_example < g_numExample; i_example++)
		{
			ori_exampleWeights[i_vertex][i_example] =  g_exampleWeights[i_vertex + i_example*g_numVertices];
		}
	}


	if(exampleSolver)
		exampleSolver->SolveVertices(delta_xi,delta_exampleWeightsOfVertex,ori_exampleWeights);
	std::vector<float> delta_exampleWeights;
	propagateExampleWeight(delta_exampleWeightsOfVertex,delta_exampleWeights);

	float lamda = 0.1f;
	int fade_count = 1;
	for (int i = 0; i < fade_count; i++)
	{
		for (int i_vertex = 0; i_vertex < g_numVertices; i_vertex++)
		{
			float weight_sum = 0.0f;
			for (int i_example = 0; i_example < g_numExample; i_example++)
			{
				g_exampleWeights[i_vertex + i_example*g_numVertices] += (1-lamda)* delta_exampleWeights[i_vertex + i_example*g_numVertices];
				weight_sum += g_exampleWeights[i_vertex + i_example*g_numVertices];
			}
			for (int i_example = 0; i_example < g_numExample; i_example++)
			{
				g_exampleWeights[i_vertex + i_example*g_numVertices] /= weight_sum; //clamp example between 0 to 1;

			}
		}
		for (int i_vertex = 0; i_vertex < g_numVertices; i_vertex++)
		{
			for (int i_example = 0; i_example < g_numExample; i_example++)
			{
				delta_exampleWeights[i_vertex + i_example*g_numVertices] *= lamda;
			}
		}




		genetatedVertice(  inputVertices , g_inputVertices, g_numVertices,
			g_transfos ,g_numBone, g_numExample,g_numIndices,
			g_boneWeights,
			g_boneWightIdx,
			g_exampleWeights);

		exportObj( inputVertices,g_faces,"./resource/meshes/keg3/bullet_1.obj");

	}
	faces = g_faces;
}

void Example_mesh_ctrl::processCollide(SampleSet& smpset, std::vector<ManipulatedObject*>& selected_obj)
{
	int idx_to_manipulate = smpset.size()-1;
	if (idx_to_manipulate < 0)return;

	std::vector<float> vertices_beforCollide;
	genetatedVertice(vertices_beforCollide, g_inputVertices, g_numVertices,
		g_transfos, g_numBone, g_numExample, g_numIndices,
		g_boneWeights,
		g_boneWightIdx,
		g_exampleWeights);

	std::map<int, std::vector<float> > delta_exampleWeightsOfVertex;
	std::map<int, std::vector<float> > ori_exampleWeights;
	for (int i_vertex = 0; i_vertex < g_numVertices; i_vertex++)
	{
		ori_exampleWeights[i_vertex] = std::vector<float>(g_numExample);
		for (int i_example = 0; i_example < g_numExample; i_example++)
		{
			ori_exampleWeights[i_vertex][i_example] = g_exampleWeights[i_vertex + i_example*g_numVertices];
		}
	}

	std::map<int, Tbx::Vec3> delta_xi;
	for (ManipulatedObject* obj : selected_obj)
	{
		if (obj->getType() == ManipulatedObject::VERTEX)
		{
			ManipulatedObjectIsVertex* vtxobj = (ManipulatedObjectIsVertex*)obj;
			int idx = vtxobj->getVertexIdx();
			qglviewer::Vec manipulate_pos = vtxobj->getLocalPosition();
			qglviewer::Vec ori_pos(vertices_beforCollide[3 * idx + 0], vertices_beforCollide[3 * idx + 1], vertices_beforCollide[3 * idx + 2]);
			qglviewer::Vec delta = manipulate_pos - ori_pos;
			delta_xi[idx] = Tbx::Vec3(delta.x, delta.y, delta.z);
		}
		

	}

	if (exampleSolver)
		exampleSolver->SolveVertices(delta_xi, delta_exampleWeightsOfVertex, ori_exampleWeights);
	std::vector<float> delta_exampleWeights;
	propagateExampleWeight(delta_exampleWeightsOfVertex, delta_exampleWeights);

	float lamda = 0.1f;
	int fade_count = 1;
	std::vector<float> vertices_afterCollide;
	for (int i = 0; i < fade_count; i++)
	{
		for (int i_vertex = 0; i_vertex < g_numVertices; i_vertex++)
		{
			float weight_sum = 0.0f;
			for (int i_example = 0; i_example < g_numExample; i_example++)
			{
				g_exampleWeights[i_vertex + i_example*g_numVertices] += (1 - lamda)* delta_exampleWeights[i_vertex + i_example*g_numVertices];
				weight_sum += g_exampleWeights[i_vertex + i_example*g_numVertices];
			}
			for (int i_example = 0; i_example < g_numExample; i_example++)
			{
				g_exampleWeights[i_vertex + i_example*g_numVertices] /= weight_sum; //clamp example between 0 to 1;

			}
		}
		for (int i_vertex = 0; i_vertex < g_numVertices; i_vertex++)
		{
			for (int i_example = 0; i_example < g_numExample; i_example++)
			{
				delta_exampleWeights[i_vertex + i_example*g_numVertices] *= lamda;
			}
		}

		genetatedVertice(vertices_afterCollide, g_inputVertices, g_numVertices,
			g_transfos, g_numBone, g_numExample, g_numIndices,
			g_boneWeights,
			g_boneWightIdx,
			g_exampleWeights);
		//
		Sample& smp= smpset[idx_to_manipulate];
		for ( int i = 0 ; i <smp.num_vertices() ; ++i)
		{
			Vertex& vtx = smp[i];
			vtx.set_position(pcm::PointType(vertices_afterCollide[3 * i + 0],
				vertices_afterCollide[3 * i + 1],
				vertices_afterCollide[3 * i + 2]));
		}
		
	}



}