#include "Rigidbodydemo.h"
#include "Bullet3Common/b3FileUtils.h"
#include "bullet/ThirdPartyLibs/stb_image/stb_image.h"
#include "bullet/Importers/ImportMeshUtility/b3ImportMeshUtility.h"
#include "bullet/Importers/ImportObjDemo/LoadMeshFromObj.h"
#include "GLInstancingRenderer.h"
#include "GLInstanceGraphicsShape.h"
#include "ConvexDecomposition/ConvexDecomposition.h"
#include "ConvexDecomposition/cd_wavefront.h"
#include "HACD/hacdVector.h"
#include "HACD/hacdCircularList.h"
#include "HACD/hacdVector.h"
#include "HACD/hacdICHull.h"
#include "HACD/hacdGraph.h"
#include "HACD/hacdHACD.h"
#include "BulletCollision/CollisionShapes/btShapeHull.h"
#include "BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h"
#include "BulletCollision/Gimpact/btCompoundFromGimpact.h"
#include "Bullet3Common/b3MinMax.h"
#include "control/animated_mesh_ctrl.hpp"
#include "control/cuda_ctrl.hpp"
#include <QGLWidget>
#include <iostream>

using std::cout;
using std::endl;
#include <fstream>
#include <sstream>

#define ARRAY_SIZE_Y 5
#define ARRAY_SIZE_X 5
#define ARRAY_SIZE_Z 5
GLInstanceGraphicsShape* Glmesh_Ori= NULL;
GLInstanceGraphicsShape* GlmeshForRender= NULL;
btCollisionShape* m_trimeshShape  = NULL;

static std::vector<int> g_vertex_indexs;
static std::vector<float> g_impluses;
static std::vector<int> g_faces;
/***************************THE FAMOUS BUNNY TRIMESH********************************************/

#define REAL btScalar
	const int NUM_TRIANGLES =902;
const int NUM_VERTICES = 453;
const int NUM_INDICES  = NUM_TRIANGLES * 3;
static void skip_this_line(FILE* fd)
{
	int ret =0;
	while( ret = fgetc(fd)!= '\n'&& ret !=EOF );

}
typedef btVector3 ColorType;
typedef btVector3 PointType;
typedef btVector3 NormalType;

static bool importObj( GLInstanceGraphicsShape*& const glmesh ,std::string file_paths)
{
	if (!glmesh)
	{
		glmesh = new GLInstanceGraphicsShape();
	}
	FILE* in_file = fopen( file_paths.c_str(), "r");

	if(in_file ==NULL)
		return false;
	char pref[3];
	if(glmesh->m_vertices)
		glmesh->m_vertices->clear();
	else
		glmesh->m_vertices = new b3AlignedObjectArray<GLInstanceVertex>();
	if(glmesh->m_indices)
		glmesh->m_indices->clear();
	else
		glmesh->m_indices = new b3AlignedObjectArray<int>();
	FILE* fd = in_file ;
	//	FILE* fd = get_mesh_model_scale( filename.c_str(), &(new_sample->n_vertex),&(new_sample->n_normal),&(new_sample->n_triangle) );
	//	Logger<<"nvertex "<<new_sample->n_vertex<<"nnormal "<<new_sample->n_normal<<"ntriangle"<<new_sample->n_triangle<<std::endl;
	if(fd)
	{
		float vx,vy,vz,nx,ny,nz,cx,cy,cz;
		std::vector<PointType> v;
		std::vector<ColorType> cv;
		std::vector<NormalType> nv;
		std::vector<int> ttv;
		while( fscanf(fd ,"%2s",pref) != EOF)
		{
			if(strcmp(pref,"v")==0)
			{

				fscanf(fd, "%f %f %f",&vx,&vy,&vz);
				v.push_back( PointType(vx,vy,vz));
			}else if(strcmp(pref,"vn") ==0)
			{
				fscanf(fd,"%f %f %f",&(nx),&(ny),&(nz));
				nv.push_back( NormalType(nx,ny,nz));
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
							fscanf(fd ,"%d");
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
			GLInstanceVertex vx;
			vx.xyzw[0] = v[i].getX();
			vx.xyzw[1] = v[i].getY();
			vx.xyzw[2] = v[i].getZ();
			vx.xyzw[3] = 0.0f;
			glmesh->m_vertices->push_back(vx );
			
		}
		glmesh->m_numvertices = v.size();
		for(int i = 0 ; i< ttv.size();++i)
		{

			glmesh->m_indices->push_back( ttv[i] );

		}
		glmesh->m_numIndices = ttv.size();
		
		//	new_sample->add_triangle(triangle_array= ttv;
		fclose(fd);
	}else
	{		
		return false;
	}


}
static void exportObj( GLInstanceGraphicsShape* glmesh ,std::string file_paths)
{
	using namespace std;
	static int count = 0;
	static int frame = 0;
	count++;
	if( count == 48 )
	{
		frame++;
		count = 0;
		int numVertices = glmesh->m_numvertices;
		int numFaces = glmesh->m_numIndices;
		stringstream s;
		s<<file_paths<<frame<<".obj";
		std::string fullpath;
		s>>fullpath;
		ofstream ofs(fullpath);
		ofs.setf( ios::fixed ,ios::floatfield);
		ofs.precision(6);
		for (int v = 0; v < numVertices; ++v)
		{

			GLInstanceVertex& vertex =  (*glmesh->m_vertices)[v];
			ofs<<"v "<<vertex.xyzw[0]<<" "<<vertex.xyzw[1]<<" "<<vertex.xyzw[2]<<std::endl;
		}
		for( int i = 0 ;i< numFaces/3 ;++i)
		{
			ofs<<"f "<< (*glmesh->m_indices)[3*i]+1 <<" "<<(*glmesh->m_indices)[3*i+1]+1 <<" "<<(*glmesh->m_indices)[3*i+2]+1<<std::endl;
		}


		ofs.close();
	}




}

static void convertOriObjForRender(GLInstanceGraphicsShape* ori_glmesh , GLInstanceGraphicsShape& out_glmesh)
{
	b3AlignedObjectArray<GLInstanceVertex>* vertices = new b3AlignedObjectArray<GLInstanceVertex>;

		//		int numVertices = obj->vertexCount;
		//	int numIndices = 0;
		b3AlignedObjectArray<int>* indicesPtr = new b3AlignedObjectArray<int>;


			int faceCount = ori_glmesh->m_numIndices;


			for (int f=0;f<faceCount;f+=3)
			{

				//btVector3 normal(face.m_plane[0],face.m_plane[1],face.m_plane[2]);
				if (1)
				{
					btVector3 normal(0,1,0);
					int vtxBaseIndex = vertices->size();


					if (f<0 && f>=faceCount)
					{
						continue;
					}

					GLInstanceVertex vtx0;				
					vtx0.xyzw[0] = (*ori_glmesh->m_vertices)[ (*ori_glmesh->m_indices)[f]  ].xyzw[0];
					vtx0.xyzw[1] = (*ori_glmesh->m_vertices)[ (*ori_glmesh->m_indices)[f]  ].xyzw[1];
					vtx0.xyzw[2] = (*ori_glmesh->m_vertices)[ (*ori_glmesh->m_indices)[f]  ].xyzw[2];
					vtx0.xyzw[3] = 0.f;

					GLInstanceVertex vtx1;				
					vtx1.xyzw[0] = (*ori_glmesh->m_vertices)[ (*ori_glmesh->m_indices)[f+ 1]  ].xyzw[0];
					vtx1.xyzw[1] = (*ori_glmesh->m_vertices)[ (*ori_glmesh->m_indices)[f +1]  ].xyzw[1];
					vtx1.xyzw[2] = (*ori_glmesh->m_vertices)[ (*ori_glmesh->m_indices)[f +1]  ].xyzw[2];
					vtx1.xyzw[3] = 0.f;

					GLInstanceVertex vtx2;				
					vtx2.xyzw[0] = (*ori_glmesh->m_vertices)[ (*ori_glmesh->m_indices)[f +2]  ].xyzw[0];
					vtx2.xyzw[1] = (*ori_glmesh->m_vertices)[ (*ori_glmesh->m_indices)[f +2]  ].xyzw[1];
					vtx2.xyzw[2] = (*ori_glmesh->m_vertices)[ (*ori_glmesh->m_indices)[f +2]  ].xyzw[2];
					vtx2.xyzw[3] = 0.f;

					btVector3 v0(vtx0.xyzw[0],vtx0.xyzw[1],vtx0.xyzw[2]);
					btVector3 v1(vtx1.xyzw[0],vtx1.xyzw[1],vtx1.xyzw[2]);
					btVector3 v2(vtx2.xyzw[0],vtx2.xyzw[1],vtx2.xyzw[2]);

					vertices->push_back(vtx0);
					vertices->push_back(vtx1);
					vertices->push_back(vtx2);
					indicesPtr->push_back(vtxBaseIndex);
					indicesPtr->push_back(vtxBaseIndex+1);
					indicesPtr->push_back(vtxBaseIndex+2);


				}
			}



		GLInstanceGraphicsShape* gfxShape = &out_glmesh;
		if(gfxShape->m_vertices)
			delete gfxShape->m_vertices;
		if(gfxShape->m_indices)
			delete gfxShape->m_indices;
		gfxShape->m_vertices = vertices;
		gfxShape->m_numvertices = vertices->size();
		gfxShape->m_indices = indicesPtr;
		gfxShape->m_numIndices = indicesPtr->size();
		for (int i=0;i<4;i++)
			gfxShape->m_scaling[i] = 1;//bake the scaling into the vertices

}
static void convertVecForRencer(const std::vector<float>& _vertices ,const std::vector<int>& _faces,GLInstanceGraphicsShape& out_glmesh )
{
	GLInstanceGraphicsShape* gfxShape = &out_glmesh;
	b3AlignedObjectArray<GLInstanceVertex>* vertices = NULL;
	b3AlignedObjectArray<int>* indicesPtr = NULL;
	if(gfxShape->m_vertices)
	{
		vertices = gfxShape->m_vertices;
		vertices->clear();
	}else
	{
		vertices = new b3AlignedObjectArray<GLInstanceVertex>;
		gfxShape->m_vertices = vertices;
	}
	if(gfxShape->m_indices)
	{
		indicesPtr = gfxShape->m_indices;
		indicesPtr->clear();
	}else
	{
		indicesPtr = new b3AlignedObjectArray<int>();
		gfxShape->m_indices = indicesPtr;
	}


	//		int numVertices = obj->vertexCount;
	//	int numIndices = 0;



	int faceCount = _faces.size();

	int vtxBaseIndex = 0;
	for (int f=0;f<faceCount;f+=3)
	{

		//btVector3 normal(face.m_plane[0],face.m_plane[1],face.m_plane[2]);
		if (1)
		{
			btVector3 normal(0,1,0);


			if (f<0 && f>=faceCount)
			{
				continue;
			}

			GLInstanceVertex vtx0;				
			vtx0.xyzw[0] = _vertices[ 3*_faces[f]+0  ];
			vtx0.xyzw[1] = _vertices[ 3*_faces[f]+1  ];
			vtx0.xyzw[2] = _vertices[ 3*_faces[f]+2  ];
			vtx0.xyzw[3] = 0.f;

			GLInstanceVertex vtx1;				
			vtx1.xyzw[0] = _vertices[ 3*_faces[f+ 1]+0  ];
			vtx1.xyzw[1] = _vertices[ 3*_faces[f +1]+1  ];
			vtx1.xyzw[2] = _vertices[ 3*_faces[f +1]+2  ];
			vtx1.xyzw[3] = 0.f;

			GLInstanceVertex vtx2;				
			vtx2.xyzw[0] = _vertices[ 3*_faces[f +2]+0  ];
			vtx2.xyzw[1] = _vertices[ 3*_faces[f +2]+1  ];
			vtx2.xyzw[2] = _vertices[ 3*_faces[f +2]+2  ];
			vtx2.xyzw[3] = 0.f;

			btVector3 v0(vtx0.xyzw[0],vtx0.xyzw[1],vtx0.xyzw[2]);
			btVector3 v1(vtx1.xyzw[0],vtx1.xyzw[1],vtx1.xyzw[2]);
			btVector3 v2(vtx2.xyzw[0],vtx2.xyzw[1],vtx2.xyzw[2]);

			vertices->push_back(vtx0);
			vertices->push_back(vtx1);
			vertices->push_back(vtx2);
			indicesPtr->push_back(vtxBaseIndex);
			indicesPtr->push_back(vtxBaseIndex+1);
			indicesPtr->push_back(vtxBaseIndex+2);
			vtxBaseIndex +=3;


		}
	}

	gfxShape->m_numvertices = vertices->size();

	gfxShape->m_numIndices = indicesPtr->size();
	for (int i=0;i<4;i++)
		gfxShape->m_scaling[i] = 1;//bake the scaling into the vertices
}


struct contactInfo
{
	contactInfo( btManifoldPoint& cp,int _partId0,int _index0,int _partId1,int _index1)
	{
		contactPoints = cp;
		partId0 = _partId0;
		index0 = _index0;
		partId1 = _partId1;
		index1 = _index1;
	}
	btManifoldPoint contactPoints;
	int partId0;
	int index0;
	int partId1;
	int index1;

};
std::vector<contactInfo> contactPoints;
#pragma region
REAL gVertices[NUM_VERTICES * 3] = {
	REAL(-0.334392), REAL(0.133007), REAL(0.062259),
	REAL(-0.350189), REAL(0.150354), REAL(-0.147769),
	REAL(-0.234201), REAL(0.343811), REAL(-0.174307),
	REAL(-0.200259), REAL(0.285207), REAL(0.093749),
	REAL(0.003520), REAL(0.475208), REAL(-0.159365),
	REAL(0.001856), REAL(0.419203), REAL(0.098582),
	REAL(-0.252802), REAL(0.093666), REAL(0.237538),
	REAL(-0.162901), REAL(0.237984), REAL(0.206905),
	REAL(0.000865), REAL(0.318141), REAL(0.235370),
	REAL(-0.414624), REAL(0.164083), REAL(-0.278254),
	REAL(-0.262213), REAL(0.357334), REAL(-0.293246),
	REAL(0.004628), REAL(0.482694), REAL(-0.338626),
	REAL(-0.402162), REAL(0.133528), REAL(-0.443247),
	REAL(-0.243781), REAL(0.324275), REAL(-0.436763),
	REAL(0.005293), REAL(0.437592), REAL(-0.458332),
	REAL(-0.339884), REAL(-0.041150), REAL(-0.668211),
	REAL(-0.248382), REAL(0.255825), REAL(-0.627493),
	REAL(0.006261), REAL(0.376103), REAL(-0.631506),
	REAL(-0.216201), REAL(-0.126776), REAL(-0.886936),
	REAL(-0.171075), REAL(0.011544), REAL(-0.881386),
	REAL(-0.181074), REAL(0.098223), REAL(-0.814779),
	REAL(-0.119891), REAL(0.218786), REAL(-0.760153),
	REAL(-0.078895), REAL(0.276780), REAL(-0.739281),
	REAL(0.006801), REAL(0.310959), REAL(-0.735661),
	REAL(-0.168842), REAL(0.102387), REAL(-0.920381),
	REAL(-0.104072), REAL(0.177278), REAL(-0.952530),
	REAL(-0.129704), REAL(0.211848), REAL(-0.836678),
	REAL(-0.099875), REAL(0.310931), REAL(-0.799381),
	REAL(0.007237), REAL(0.361687), REAL(-0.794439),
	REAL(-0.077913), REAL(0.258753), REAL(-0.921640),
	REAL(0.007957), REAL(0.282241), REAL(-0.931680),
	REAL(-0.252222), REAL(-0.550401), REAL(-0.557810),
	REAL(-0.267633), REAL(-0.603419), REAL(-0.655209),
	REAL(-0.446838), REAL(-0.118517), REAL(-0.466159),
	REAL(-0.459488), REAL(-0.093017), REAL(-0.311341),
	REAL(-0.370645), REAL(-0.100108), REAL(-0.159454),
	REAL(-0.371984), REAL(-0.091991), REAL(-0.011044),
	REAL(-0.328945), REAL(-0.098269), REAL(0.088659),
	REAL(-0.282452), REAL(-0.018862), REAL(0.311501),
	REAL(-0.352403), REAL(-0.131341), REAL(0.144902),
	REAL(-0.364126), REAL(-0.200299), REAL(0.202388),
	REAL(-0.283965), REAL(-0.231869), REAL(0.023668),
	REAL(-0.298943), REAL(-0.155218), REAL(0.369716),
	REAL(-0.293787), REAL(-0.121856), REAL(0.419097),
	REAL(-0.290163), REAL(-0.290797), REAL(0.107824),
	REAL(-0.264165), REAL(-0.272849), REAL(0.036347),
	REAL(-0.228567), REAL(-0.372573), REAL(0.290309),
	REAL(-0.190431), REAL(-0.286997), REAL(0.421917),
	REAL(-0.191039), REAL(-0.240973), REAL(0.507118),
	REAL(-0.287272), REAL(-0.276431), REAL(-0.065444),
	REAL(-0.295675), REAL(-0.280818), REAL(-0.174200),
	REAL(-0.399537), REAL(-0.313131), REAL(-0.376167),
	REAL(-0.392666), REAL(-0.488581), REAL(-0.427494),
	REAL(-0.331669), REAL(-0.570185), REAL(-0.466054),
	REAL(-0.282290), REAL(-0.618140), REAL(-0.589220),
	REAL(-0.374238), REAL(-0.594882), REAL(-0.323298),
	REAL(-0.381071), REAL(-0.629723), REAL(-0.350777),
	REAL(-0.382112), REAL(-0.624060), REAL(-0.221577),
	REAL(-0.272701), REAL(-0.566522), REAL(0.259157),
	REAL(-0.256702), REAL(-0.663406), REAL(0.286079),
	REAL(-0.280948), REAL(-0.428359), REAL(0.055790),
	REAL(-0.184974), REAL(-0.508894), REAL(0.326265),
	REAL(-0.279971), REAL(-0.526918), REAL(0.395319),
	REAL(-0.282599), REAL(-0.663393), REAL(0.412411),
	REAL(-0.188329), REAL(-0.475093), REAL(0.417954),
	REAL(-0.263384), REAL(-0.663396), REAL(0.466604),
	REAL(-0.209063), REAL(-0.663393), REAL(0.509344),
	REAL(-0.002044), REAL(-0.319624), REAL(0.553078),
	REAL(-0.001266), REAL(-0.371260), REAL(0.413296),
	REAL(-0.219753), REAL(-0.339762), REAL(-0.040921),
	REAL(-0.256986), REAL(-0.282511), REAL(-0.006349),
	REAL(-0.271706), REAL(-0.260881), REAL(0.001764),
	REAL(-0.091191), REAL(-0.419184), REAL(-0.045912),
	REAL(-0.114944), REAL(-0.429752), REAL(-0.124739),
	REAL(-0.113970), REAL(-0.382987), REAL(-0.188540),
	REAL(-0.243012), REAL(-0.464942), REAL(-0.242850),
	REAL(-0.314815), REAL(-0.505402), REAL(-0.324768),
	REAL(0.002774), REAL(-0.437526), REAL(-0.262766),
	REAL(-0.072625), REAL(-0.417748), REAL(-0.221440),
	REAL(-0.160112), REAL(-0.476932), REAL(-0.293450),
	REAL(0.003859), REAL(-0.453425), REAL(-0.443916),
	REAL(-0.120363), REAL(-0.581567), REAL(-0.438689),
	REAL(-0.091499), REAL(-0.584191), REAL(-0.294511),
	REAL(-0.116469), REAL(-0.599861), REAL(-0.188308),
	REAL(-0.208032), REAL(-0.513640), REAL(-0.134649),
	REAL(-0.235749), REAL(-0.610017), REAL(-0.040939),
	REAL(-0.344916), REAL(-0.622487), REAL(-0.085380),
	REAL(-0.336401), REAL(-0.531864), REAL(-0.212298),
	REAL(0.001961), REAL(-0.459550), REAL(-0.135547),
	REAL(-0.058296), REAL(-0.430536), REAL(-0.043440),
	REAL(0.001378), REAL(-0.449511), REAL(-0.037762),
	REAL(-0.130135), REAL(-0.510222), REAL(0.079144),
	REAL(0.000142), REAL(-0.477549), REAL(0.157064),
	REAL(-0.114284), REAL(-0.453206), REAL(0.304397),
	REAL(-0.000592), REAL(-0.443558), REAL(0.285401),
	REAL(-0.056215), REAL(-0.663402), REAL(0.326073),
	REAL(-0.026248), REAL(-0.568010), REAL(0.273318),
	REAL(-0.049261), REAL(-0.531064), REAL(0.389854),
	REAL(-0.127096), REAL(-0.663398), REAL(0.479316),
	REAL(-0.058384), REAL(-0.663401), REAL(0.372891),
	REAL(-0.303961), REAL(0.054199), REAL(0.625921),
	REAL(-0.268594), REAL(0.193403), REAL(0.502766),
	REAL(-0.277159), REAL(0.126123), REAL(0.443289),
	REAL(-0.287605), REAL(-0.005722), REAL(0.531844),
	REAL(-0.231396), REAL(-0.121289), REAL(0.587387),
	REAL(-0.253475), REAL(-0.081797), REAL(0.756541),
	REAL(-0.195164), REAL(-0.137969), REAL(0.728011),
	REAL(-0.167673), REAL(-0.156573), REAL(0.609388),
	REAL(-0.145917), REAL(-0.169029), REAL(0.697600),
	REAL(-0.077776), REAL(-0.214247), REAL(0.622586),
	REAL(-0.076873), REAL(-0.214971), REAL(0.696301),
	REAL(-0.002341), REAL(-0.233135), REAL(0.622859),
	REAL(-0.002730), REAL(-0.213526), REAL(0.691267),
	REAL(-0.003136), REAL(-0.192628), REAL(0.762731),
	REAL(-0.056136), REAL(-0.201222), REAL(0.763806),
	REAL(-0.114589), REAL(-0.166192), REAL(0.770723),
	REAL(-0.155145), REAL(-0.129632), REAL(0.791738),
	REAL(-0.183611), REAL(-0.058705), REAL(0.847012),
	REAL(-0.165562), REAL(0.001980), REAL(0.833386),
	REAL(-0.220084), REAL(0.019914), REAL(0.768935),
	REAL(-0.255730), REAL(0.090306), REAL(0.670782),
	REAL(-0.255594), REAL(0.113833), REAL(0.663389),
	REAL(-0.226380), REAL(0.212655), REAL(0.617740),
	REAL(-0.003367), REAL(-0.195342), REAL(0.799680),
	REAL(-0.029743), REAL(-0.210508), REAL(0.827180),
	REAL(-0.003818), REAL(-0.194783), REAL(0.873636),
	REAL(-0.004116), REAL(-0.157907), REAL(0.931268),
	REAL(-0.031280), REAL(-0.184555), REAL(0.889476),
	REAL(-0.059885), REAL(-0.184448), REAL(0.841330),
	REAL(-0.135333), REAL(-0.164332), REAL(0.878200),
	REAL(-0.085574), REAL(-0.170948), REAL(0.925547),
	REAL(-0.163833), REAL(-0.094170), REAL(0.897114),
	REAL(-0.138444), REAL(-0.104250), REAL(0.945975),
	REAL(-0.083497), REAL(-0.084934), REAL(0.979607),
	REAL(-0.004433), REAL(-0.146642), REAL(0.985872),
	REAL(-0.150715), REAL(0.032650), REAL(0.884111),
	REAL(-0.135892), REAL(-0.035520), REAL(0.945455),
	REAL(-0.070612), REAL(0.036849), REAL(0.975733),
	REAL(-0.004458), REAL(-0.042526), REAL(1.015670),
	REAL(-0.004249), REAL(0.046042), REAL(1.003240),
	REAL(-0.086969), REAL(0.133224), REAL(0.947633),
	REAL(-0.003873), REAL(0.161605), REAL(0.970499),
	REAL(-0.125544), REAL(0.140012), REAL(0.917678),
	REAL(-0.125651), REAL(0.250246), REAL(0.857602),
	REAL(-0.003127), REAL(0.284070), REAL(0.878870),
	REAL(-0.159174), REAL(0.125726), REAL(0.888878),
	REAL(-0.183807), REAL(0.196970), REAL(0.844480),
	REAL(-0.159890), REAL(0.291736), REAL(0.732480),
	REAL(-0.199495), REAL(0.207230), REAL(0.779864),
	REAL(-0.206182), REAL(0.164608), REAL(0.693257),
	REAL(-0.186315), REAL(0.160689), REAL(0.817193),
	REAL(-0.192827), REAL(0.166706), REAL(0.782271),
	REAL(-0.175112), REAL(0.110008), REAL(0.860621),
	REAL(-0.161022), REAL(0.057420), REAL(0.855111),
	REAL(-0.172319), REAL(0.036155), REAL(0.816189),
	REAL(-0.190318), REAL(0.064083), REAL(0.760605),
	REAL(-0.195072), REAL(0.129179), REAL(0.731104),
	REAL(-0.203126), REAL(0.410287), REAL(0.680536),
	REAL(-0.216677), REAL(0.309274), REAL(0.642272),
	REAL(-0.241515), REAL(0.311485), REAL(0.587832),
	REAL(-0.002209), REAL(0.366663), REAL(0.749413),
	REAL(-0.088230), REAL(0.396265), REAL(0.678635),
	REAL(-0.170147), REAL(0.109517), REAL(0.840784),
	REAL(-0.160521), REAL(0.067766), REAL(0.830650),
	REAL(-0.181546), REAL(0.139805), REAL(0.812146),
	REAL(-0.180495), REAL(0.148568), REAL(0.776087),
	REAL(-0.180255), REAL(0.129125), REAL(0.744192),
	REAL(-0.186298), REAL(0.078308), REAL(0.769352),
	REAL(-0.167622), REAL(0.060539), REAL(0.806675),
	REAL(-0.189876), REAL(0.102760), REAL(0.802582),
	REAL(-0.108340), REAL(0.455446), REAL(0.657174),
	REAL(-0.241585), REAL(0.527592), REAL(0.669296),
	REAL(-0.265676), REAL(0.513366), REAL(0.634594),
	REAL(-0.203073), REAL(0.478550), REAL(0.581526),
	REAL(-0.266772), REAL(0.642330), REAL(0.602061),
	REAL(-0.216961), REAL(0.564846), REAL(0.535435),
	REAL(-0.202210), REAL(0.525495), REAL(0.475944),
	REAL(-0.193888), REAL(0.467925), REAL(0.520606),
	REAL(-0.265837), REAL(0.757267), REAL(0.500933),
	REAL(-0.240306), REAL(0.653440), REAL(0.463215),
	REAL(-0.309239), REAL(0.776868), REAL(0.304726),
	REAL(-0.271009), REAL(0.683094), REAL(0.382018),
	REAL(-0.312111), REAL(0.671099), REAL(0.286687),
	REAL(-0.268791), REAL(0.624342), REAL(0.377231),
	REAL(-0.302457), REAL(0.533996), REAL(0.360289),
	REAL(-0.263656), REAL(0.529310), REAL(0.412564),
	REAL(-0.282311), REAL(0.415167), REAL(0.447666),
	REAL(-0.239201), REAL(0.442096), REAL(0.495604),
	REAL(-0.220043), REAL(0.569026), REAL(0.445877),
	REAL(-0.001263), REAL(0.395631), REAL(0.602029),
	REAL(-0.057345), REAL(0.442535), REAL(0.572224),
	REAL(-0.088927), REAL(0.506333), REAL(0.529106),
	REAL(-0.125738), REAL(0.535076), REAL(0.612913),
	REAL(-0.126251), REAL(0.577170), REAL(0.483159),
	REAL(-0.149594), REAL(0.611520), REAL(0.557731),
	REAL(-0.163188), REAL(0.660791), REAL(0.491080),
	REAL(-0.172482), REAL(0.663387), REAL(0.415416),
	REAL(-0.160464), REAL(0.591710), REAL(0.370659),
	REAL(-0.156445), REAL(0.536396), REAL(0.378302),
	REAL(-0.136496), REAL(0.444358), REAL(0.425226),
	REAL(-0.095564), REAL(0.373768), REAL(0.473659),
	REAL(-0.104146), REAL(0.315912), REAL(0.498104),
	REAL(-0.000496), REAL(0.384194), REAL(0.473817),
	REAL(-0.000183), REAL(0.297770), REAL(0.401486),
	REAL(-0.129042), REAL(0.270145), REAL(0.434495),
	REAL(0.000100), REAL(0.272963), REAL(0.349138),
	REAL(-0.113060), REAL(0.236984), REAL(0.385554),
	REAL(0.007260), REAL(0.016311), REAL(-0.883396),
	REAL(0.007865), REAL(0.122104), REAL(-0.956137),
	REAL(-0.032842), REAL(0.115282), REAL(-0.953252),
	REAL(-0.089115), REAL(0.108449), REAL(-0.950317),
	REAL(-0.047440), REAL(0.014729), REAL(-0.882756),
	REAL(-0.104458), REAL(0.013137), REAL(-0.882070),
	REAL(-0.086439), REAL(-0.584866), REAL(-0.608343),
	REAL(-0.115026), REAL(-0.662605), REAL(-0.436732),
	REAL(-0.071683), REAL(-0.665372), REAL(-0.606385),
	REAL(-0.257884), REAL(-0.665381), REAL(-0.658052),
	REAL(-0.272542), REAL(-0.665381), REAL(-0.592063),
	REAL(-0.371322), REAL(-0.665382), REAL(-0.353620),
	REAL(-0.372362), REAL(-0.665381), REAL(-0.224420),
	REAL(-0.335166), REAL(-0.665380), REAL(-0.078623),
	REAL(-0.225999), REAL(-0.665375), REAL(-0.038981),
	REAL(-0.106719), REAL(-0.665374), REAL(-0.186351),
	REAL(-0.081749), REAL(-0.665372), REAL(-0.292554),
	REAL(0.006943), REAL(-0.091505), REAL(-0.858354),
	REAL(0.006117), REAL(-0.280985), REAL(-0.769967),
	REAL(0.004495), REAL(-0.502360), REAL(-0.559799),
	REAL(-0.198638), REAL(-0.302135), REAL(-0.845816),
	REAL(-0.237395), REAL(-0.542544), REAL(-0.587188),
	REAL(-0.270001), REAL(-0.279489), REAL(-0.669861),
	REAL(-0.134547), REAL(-0.119852), REAL(-0.959004),
	REAL(-0.052088), REAL(-0.122463), REAL(-0.944549),
	REAL(-0.124463), REAL(-0.293508), REAL(-0.899566),
	REAL(-0.047616), REAL(-0.289643), REAL(-0.879292),
	REAL(-0.168595), REAL(-0.529132), REAL(-0.654931),
	REAL(-0.099793), REAL(-0.515719), REAL(-0.645873),
	REAL(-0.186168), REAL(-0.605282), REAL(-0.724690),
	REAL(-0.112970), REAL(-0.583097), REAL(-0.707469),
	REAL(-0.108152), REAL(-0.665375), REAL(-0.700408),
	REAL(-0.183019), REAL(-0.665378), REAL(-0.717630),
	REAL(-0.349529), REAL(-0.334459), REAL(-0.511985),
	REAL(-0.141182), REAL(-0.437705), REAL(-0.798194),
	REAL(-0.212670), REAL(-0.448725), REAL(-0.737447),
	REAL(-0.261111), REAL(-0.414945), REAL(-0.613835),
	REAL(-0.077364), REAL(-0.431480), REAL(-0.778113),
	REAL(0.005174), REAL(-0.425277), REAL(-0.651592),
	REAL(0.089236), REAL(-0.431732), REAL(-0.777093),
	REAL(0.271006), REAL(-0.415749), REAL(-0.610577),
	REAL(0.223981), REAL(-0.449384), REAL(-0.734774),
	REAL(0.153275), REAL(-0.438150), REAL(-0.796391),
	REAL(0.358414), REAL(-0.335529), REAL(-0.507649),
	REAL(0.193434), REAL(-0.665946), REAL(-0.715325),
	REAL(0.118363), REAL(-0.665717), REAL(-0.699021),
	REAL(0.123515), REAL(-0.583454), REAL(-0.706020),
	REAL(0.196851), REAL(-0.605860), REAL(-0.722345),
	REAL(0.109788), REAL(-0.516035), REAL(-0.644590),
	REAL(0.178656), REAL(-0.529656), REAL(-0.652804),
	REAL(0.061157), REAL(-0.289807), REAL(-0.878626),
	REAL(0.138234), REAL(-0.293905), REAL(-0.897958),
	REAL(0.066933), REAL(-0.122643), REAL(-0.943820),
	REAL(0.149571), REAL(-0.120281), REAL(-0.957264),
	REAL(0.280989), REAL(-0.280321), REAL(-0.666487),
	REAL(0.246581), REAL(-0.543275), REAL(-0.584224),
	REAL(0.211720), REAL(-0.302754), REAL(-0.843303),
	REAL(0.086966), REAL(-0.665627), REAL(-0.291520),
	REAL(0.110634), REAL(-0.665702), REAL(-0.185021),
	REAL(0.228099), REAL(-0.666061), REAL(-0.036201),
	REAL(0.337743), REAL(-0.666396), REAL(-0.074503),
	REAL(0.376722), REAL(-0.666513), REAL(-0.219833),
	REAL(0.377265), REAL(-0.666513), REAL(-0.349036),
	REAL(0.281411), REAL(-0.666217), REAL(-0.588670),
	REAL(0.267564), REAL(-0.666174), REAL(-0.654834),
	REAL(0.080745), REAL(-0.665602), REAL(-0.605452),
	REAL(0.122016), REAL(-0.662963), REAL(-0.435280),
	REAL(0.095767), REAL(-0.585141), REAL(-0.607228),
	REAL(0.118944), REAL(0.012799), REAL(-0.880702),
	REAL(0.061944), REAL(0.014564), REAL(-0.882086),
	REAL(0.104725), REAL(0.108156), REAL(-0.949130),
	REAL(0.048513), REAL(0.115159), REAL(-0.952753),
	REAL(0.112696), REAL(0.236643), REAL(0.386937),
	REAL(0.128177), REAL(0.269757), REAL(0.436071),
	REAL(0.102643), REAL(0.315600), REAL(0.499370),
	REAL(0.094535), REAL(0.373481), REAL(0.474824),
	REAL(0.136270), REAL(0.443946), REAL(0.426895),
	REAL(0.157071), REAL(0.535923), REAL(0.380222),
	REAL(0.161350), REAL(0.591224), REAL(0.372630),
	REAL(0.173035), REAL(0.662865), REAL(0.417531),
	REAL(0.162808), REAL(0.660299), REAL(0.493077),
	REAL(0.148250), REAL(0.611070), REAL(0.559555),
	REAL(0.125719), REAL(0.576790), REAL(0.484702),
	REAL(0.123489), REAL(0.534699), REAL(0.614440),
	REAL(0.087621), REAL(0.506066), REAL(0.530188),
	REAL(0.055321), REAL(0.442365), REAL(0.572915),
	REAL(0.219936), REAL(0.568361), REAL(0.448571),
	REAL(0.238099), REAL(0.441375), REAL(0.498528),
	REAL(0.281711), REAL(0.414315), REAL(0.451121),
	REAL(0.263833), REAL(0.528513), REAL(0.415794),
	REAL(0.303284), REAL(0.533081), REAL(0.363998),
	REAL(0.269687), REAL(0.623528), REAL(0.380528),
	REAL(0.314255), REAL(0.670153), REAL(0.290524),
	REAL(0.272023), REAL(0.682273), REAL(0.385343),
	REAL(0.311480), REAL(0.775931), REAL(0.308527),
	REAL(0.240239), REAL(0.652714), REAL(0.466159),
	REAL(0.265619), REAL(0.756464), REAL(0.504187),
	REAL(0.192562), REAL(0.467341), REAL(0.522972),
	REAL(0.201605), REAL(0.524885), REAL(0.478417),
	REAL(0.215743), REAL(0.564193), REAL(0.538084),
	REAL(0.264969), REAL(0.641527), REAL(0.605317),
	REAL(0.201031), REAL(0.477940), REAL(0.584002),
	REAL(0.263086), REAL(0.512567), REAL(0.637832),
	REAL(0.238615), REAL(0.526867), REAL(0.672237),
	REAL(0.105309), REAL(0.455123), REAL(0.658482),
	REAL(0.183993), REAL(0.102195), REAL(0.804872),
	REAL(0.161563), REAL(0.060042), REAL(0.808692),
	REAL(0.180748), REAL(0.077754), REAL(0.771600),
	REAL(0.175168), REAL(0.128588), REAL(0.746368),
	REAL(0.175075), REAL(0.148030), REAL(0.778264),
	REAL(0.175658), REAL(0.139265), REAL(0.814333),
	REAL(0.154191), REAL(0.067291), REAL(0.832578),
	REAL(0.163818), REAL(0.109013), REAL(0.842830),
	REAL(0.084760), REAL(0.396004), REAL(0.679695),
	REAL(0.238888), REAL(0.310760), REAL(0.590775),
	REAL(0.213380), REAL(0.308625), REAL(0.644905),
	REAL(0.199666), REAL(0.409678), REAL(0.683003),
	REAL(0.190143), REAL(0.128597), REAL(0.733463),
	REAL(0.184833), REAL(0.063516), REAL(0.762902),
	REAL(0.166070), REAL(0.035644), REAL(0.818261),
	REAL(0.154361), REAL(0.056943), REAL(0.857042),
	REAL(0.168542), REAL(0.109489), REAL(0.862725),
	REAL(0.187387), REAL(0.166131), REAL(0.784599),
	REAL(0.180428), REAL(0.160135), REAL(0.819438),
	REAL(0.201823), REAL(0.163991), REAL(0.695756),
	REAL(0.194206), REAL(0.206635), REAL(0.782275),
	REAL(0.155438), REAL(0.291260), REAL(0.734412),
	REAL(0.177696), REAL(0.196424), REAL(0.846693),
	REAL(0.152305), REAL(0.125256), REAL(0.890786),
	REAL(0.119546), REAL(0.249876), REAL(0.859104),
	REAL(0.118369), REAL(0.139643), REAL(0.919173),
	REAL(0.079410), REAL(0.132973), REAL(0.948652),
	REAL(0.062419), REAL(0.036648), REAL(0.976547),
	REAL(0.127847), REAL(-0.035919), REAL(0.947070),
	REAL(0.143624), REAL(0.032206), REAL(0.885913),
	REAL(0.074888), REAL(-0.085173), REAL(0.980577),
	REAL(0.130184), REAL(-0.104656), REAL(0.947620),
	REAL(0.156201), REAL(-0.094653), REAL(0.899074),
	REAL(0.077366), REAL(-0.171194), REAL(0.926545),
	REAL(0.127722), REAL(-0.164729), REAL(0.879810),
	REAL(0.052670), REAL(-0.184618), REAL(0.842019),
	REAL(0.023477), REAL(-0.184638), REAL(0.889811),
	REAL(0.022626), REAL(-0.210587), REAL(0.827500),
	REAL(0.223089), REAL(0.211976), REAL(0.620493),
	REAL(0.251444), REAL(0.113067), REAL(0.666494),
	REAL(0.251419), REAL(0.089540), REAL(0.673887),
	REAL(0.214360), REAL(0.019258), REAL(0.771595),
	REAL(0.158999), REAL(0.001490), REAL(0.835374),
	REAL(0.176696), REAL(-0.059249), REAL(0.849218),
	REAL(0.148696), REAL(-0.130091), REAL(0.793599),
	REAL(0.108290), REAL(-0.166528), REAL(0.772088),
	REAL(0.049820), REAL(-0.201382), REAL(0.764454),
	REAL(0.071341), REAL(-0.215195), REAL(0.697209),
	REAL(0.073148), REAL(-0.214475), REAL(0.623510),
	REAL(0.140502), REAL(-0.169461), REAL(0.699354),
	REAL(0.163374), REAL(-0.157073), REAL(0.611416),
	REAL(0.189466), REAL(-0.138550), REAL(0.730366),
	REAL(0.247593), REAL(-0.082554), REAL(0.759610),
	REAL(0.227468), REAL(-0.121982), REAL(0.590197),
	REAL(0.284702), REAL(-0.006586), REAL(0.535347),
	REAL(0.275741), REAL(0.125287), REAL(0.446676),
	REAL(0.266650), REAL(0.192594), REAL(0.506044),
	REAL(0.300086), REAL(0.053287), REAL(0.629620),
	REAL(0.055450), REAL(-0.663935), REAL(0.375065),
	REAL(0.122854), REAL(-0.664138), REAL(0.482323),
	REAL(0.046520), REAL(-0.531571), REAL(0.391918),
	REAL(0.024824), REAL(-0.568450), REAL(0.275106),
	REAL(0.053855), REAL(-0.663931), REAL(0.328224),
	REAL(0.112829), REAL(-0.453549), REAL(0.305788),
	REAL(0.131265), REAL(-0.510617), REAL(0.080746),
	REAL(0.061174), REAL(-0.430716), REAL(-0.042710),
	REAL(0.341019), REAL(-0.532887), REAL(-0.208150),
	REAL(0.347705), REAL(-0.623533), REAL(-0.081139),
	REAL(0.238040), REAL(-0.610732), REAL(-0.038037),
	REAL(0.211764), REAL(-0.514274), REAL(-0.132078),
	REAL(0.120605), REAL(-0.600219), REAL(-0.186856),
	REAL(0.096985), REAL(-0.584476), REAL(-0.293357),
	REAL(0.127621), REAL(-0.581941), REAL(-0.437170),
	REAL(0.165902), REAL(-0.477425), REAL(-0.291453),
	REAL(0.077720), REAL(-0.417975), REAL(-0.220519),
	REAL(0.320892), REAL(-0.506363), REAL(-0.320874),
	REAL(0.248214), REAL(-0.465684), REAL(-0.239842),
	REAL(0.118764), REAL(-0.383338), REAL(-0.187114),
	REAL(0.118816), REAL(-0.430106), REAL(-0.123307),
	REAL(0.094131), REAL(-0.419464), REAL(-0.044777),
	REAL(0.274526), REAL(-0.261706), REAL(0.005110),
	REAL(0.259842), REAL(-0.283292), REAL(-0.003185),
	REAL(0.222861), REAL(-0.340431), REAL(-0.038210),
	REAL(0.204445), REAL(-0.664380), REAL(0.513353),
	REAL(0.259286), REAL(-0.664547), REAL(0.471281),
	REAL(0.185402), REAL(-0.476020), REAL(0.421718),
	REAL(0.279163), REAL(-0.664604), REAL(0.417328),
	REAL(0.277157), REAL(-0.528122), REAL(0.400208),
	REAL(0.183069), REAL(-0.509812), REAL(0.329995),
	REAL(0.282599), REAL(-0.429210), REAL(0.059242),
	REAL(0.254816), REAL(-0.664541), REAL(0.290687),
	REAL(0.271436), REAL(-0.567707), REAL(0.263966),
	REAL(0.386561), REAL(-0.625221), REAL(-0.216870),
	REAL(0.387086), REAL(-0.630883), REAL(-0.346073),
	REAL(0.380021), REAL(-0.596021), REAL(-0.318679),
	REAL(0.291269), REAL(-0.619007), REAL(-0.585707),
	REAL(0.339280), REAL(-0.571198), REAL(-0.461946),
	REAL(0.400045), REAL(-0.489778), REAL(-0.422640),
	REAL(0.406817), REAL(-0.314349), REAL(-0.371230),
	REAL(0.300588), REAL(-0.281718), REAL(-0.170549),
	REAL(0.290866), REAL(-0.277304), REAL(-0.061905),
	REAL(0.187735), REAL(-0.241545), REAL(0.509437),
	REAL(0.188032), REAL(-0.287569), REAL(0.424234),
	REAL(0.227520), REAL(-0.373262), REAL(0.293102),
	REAL(0.266526), REAL(-0.273650), REAL(0.039597),
	REAL(0.291592), REAL(-0.291676), REAL(0.111386),
	REAL(0.291914), REAL(-0.122741), REAL(0.422683),
	REAL(0.297574), REAL(-0.156119), REAL(0.373368),
	REAL(0.286603), REAL(-0.232731), REAL(0.027162),
	REAL(0.364663), REAL(-0.201399), REAL(0.206850),
	REAL(0.353855), REAL(-0.132408), REAL(0.149228),
	REAL(0.282208), REAL(-0.019715), REAL(0.314960),
	REAL(0.331187), REAL(-0.099266), REAL(0.092701),
	REAL(0.375463), REAL(-0.093120), REAL(-0.006467),
	REAL(0.375917), REAL(-0.101236), REAL(-0.154882),
	REAL(0.466635), REAL(-0.094416), REAL(-0.305669),
	REAL(0.455805), REAL(-0.119881), REAL(-0.460632),
	REAL(0.277465), REAL(-0.604242), REAL(-0.651871),
	REAL(0.261022), REAL(-0.551176), REAL(-0.554667),
	REAL(0.093627), REAL(0.258494), REAL(-0.920589),
	REAL(0.114248), REAL(0.310608), REAL(-0.798070),
	REAL(0.144232), REAL(0.211434), REAL(-0.835001),
	REAL(0.119916), REAL(0.176940), REAL(-0.951159),
	REAL(0.184061), REAL(0.101854), REAL(-0.918220),
	REAL(0.092431), REAL(0.276521), REAL(-0.738231),
	REAL(0.133504), REAL(0.218403), REAL(-0.758602),
	REAL(0.194987), REAL(0.097655), REAL(-0.812476),
	REAL(0.185542), REAL(0.011005), REAL(-0.879202),
	REAL(0.230315), REAL(-0.127450), REAL(-0.884202),
	REAL(0.260471), REAL(0.255056), REAL(-0.624378),
	REAL(0.351567), REAL(-0.042194), REAL(-0.663976),
	REAL(0.253742), REAL(0.323524), REAL(-0.433716),
	REAL(0.411612), REAL(0.132299), REAL(-0.438264),
	REAL(0.270513), REAL(0.356530), REAL(-0.289984),
	REAL(0.422146), REAL(0.162819), REAL(-0.273130),
	REAL(0.164724), REAL(0.237490), REAL(0.208912),
	REAL(0.253806), REAL(0.092900), REAL(0.240640),
	REAL(0.203608), REAL(0.284597), REAL(0.096223),
	REAL(0.241006), REAL(0.343093), REAL(-0.171396),
	REAL(0.356076), REAL(0.149288), REAL(-0.143443),
	REAL(0.337656), REAL(0.131992), REAL(0.066374)
};

int gIndices[NUM_TRIANGLES][3] = {
	{126,134,133},
	{342,138,134},
	{133,134,138},
	{126,342,134},
	{312,316,317},
	{169,163,162},
	{312,317,319},
	{312,319,318},
	{169,162,164},
	{169,168,163},
	{312,314,315},
	{169,164,165},
	{169,167,168},
	{312,315,316},
	{312,313,314},
	{169,165,166},
	{169,166,167},
	{312,318,313},
	{308,304,305},
	{308,305,306},
	{179,181,188},
	{177,173,175},
	{177,175,176},
	{302,293,300},
	{322,294,304},
	{188,176,175},
	{188,175,179},
	{158,177,187},
	{305,293,302},
	{305,302,306},
	{322,304,308},
	{188,181,183},
	{158,173,177},
	{293,298,300},
	{304,294,296},
	{304,296,305},
	{185,176,188},
	{185,188,183},
	{187,177,176},
	{187,176,185},
	{305,296,298},
	{305,298,293},
	{436,432, 28},
	{436, 28, 23},
	{434,278,431},
	{ 30,208,209},
	{ 30,209, 29},
	{ 19, 20, 24},
	{208,207,211},
	{208,211,209},
	{ 19,210,212},
	{433,434,431},
	{433,431,432},
	{433,432,436},
	{436,437,433},
	{277,275,276},
	{277,276,278},
	{209,210, 25},
	{ 21, 26, 24},
	{ 21, 24, 20},
	{ 25, 26, 27},
	{ 25, 27, 29},
	{435,439,277},
	{439,275,277},
	{432,431, 30},
	{432, 30, 28},
	{433,437,438},
	{433,438,435},
	{434,277,278},
	{ 24, 25,210},
	{ 24, 26, 25},
	{ 29, 27, 28},
	{ 29, 28, 30},
	{ 19, 24,210},
	{208, 30,431},
	{208,431,278},
	{435,434,433},
	{435,277,434},
	{ 25, 29,209},
	{ 27, 22, 23},
	{ 27, 23, 28},
	{ 26, 22, 27},
	{ 26, 21, 22},
	{212,210,209},
	{212,209,211},
	{207,208,278},
	{207,278,276},
	{439,435,438},
	{ 12,  9, 10},
	{ 12, 10, 13},
	{  2,  3,  5},
	{  2,  5,  4},
	{ 16, 13, 14},
	{ 16, 14, 17},
	{ 22, 21, 16},
	{ 13, 10, 11},
	{ 13, 11, 14},
	{  1,  0,  3},
	{  1,  3,  2},
	{ 15, 12, 16},
	{ 19, 18, 15},
	{ 19, 15, 16},
	{ 19, 16, 20},
	{  9,  1,  2},
	{  9,  2, 10},
	{  3,  7,  8},
	{  3,  8,  5},
	{ 16, 17, 23},
	{ 16, 23, 22},
	{ 21, 20, 16},
	{ 10,  2,  4},
	{ 10,  4, 11},
	{  0,  6,  7},
	{  0,  7,  3},
	{ 12, 13, 16},
	{451,446,445},
	{451,445,450},
	{442,440,439},
	{442,439,438},
	{442,438,441},
	{421,420,422},
	{412,411,426},
	{412,426,425},
	{408,405,407},
	{413, 67, 68},
	{413, 68,414},
	{391,390,412},
	{ 80,384,386},
	{404,406,378},
	{390,391,377},
	{390,377, 88},
	{400,415,375},
	{398,396,395},
	{398,395,371},
	{398,371,370},
	{112,359,358},
	{112,358,113},
	{351,352,369},
	{125,349,348},
	{345,343,342},
	{342,340,339},
	{341,335,337},
	{328,341,327},
	{331,323,333},
	{331,322,323},
	{327,318,319},
	{327,319,328},
	{315,314,324},
	{302,300,301},
	{302,301,303},
	{320,311,292},
	{285,284,289},
	{310,307,288},
	{310,288,290},
	{321,350,281},
	{321,281,282},
	{423,448,367},
	{272,273,384},
	{272,384,274},
	{264,265,382},
	{264,382,383},
	{440,442,261},
	{440,261,263},
	{252,253,254},
	{252,254,251},
	{262,256,249},
	{262,249,248},
	{228,243,242},
	{228, 31,243},
	{213,215,238},
	{213,238,237},
	{ 19,212,230},
	{224,225,233},
	{224,233,231},
	{217,218, 56},
	{217, 56, 54},
	{217,216,239},
	{217,239,238},
	{217,238,215},
	{218,217,215},
	{218,215,214},
	{  6,102,206},
	{186,199,200},
	{197,182,180},
	{170,171,157},
	{201,200,189},
	{170,190,191},
	{170,191,192},
	{175,174,178},
	{175,178,179},
	{168,167,155},
	{122,149,158},
	{122,158,159},
	{135,153,154},
	{135,154,118},
	{143,140,141},
	{143,141,144},
	{132,133,136},
	{130,126,133},
	{124,125,127},
	{122,101,100},
	{122,100,121},
	{110,108,107},
	{110,107,109},
	{ 98, 99, 97},
	{ 98, 97, 64},
	{ 98, 64, 66},
	{ 87, 55, 57},
	{ 83, 82, 79},
	{ 83, 79, 84},
	{ 78, 74, 50},
	{ 49, 71, 41},
	{ 49, 41, 37},
	{ 49, 37, 36},
	{ 58, 44, 60},
	{ 60, 59, 58},
	{ 51, 34, 33},
	{ 39, 40, 42},
	{ 39, 42, 38},
	{243,240, 33},
	{243, 33,229},
	{ 39, 38,  6},
	{ 44, 46, 40},
	{ 55, 56, 57},
	{ 64, 62, 65},
	{ 64, 65, 66},
	{ 41, 71, 45},
	{ 75, 50, 51},
	{ 81, 79, 82},
	{ 77, 88, 73},
	{ 93, 92, 94},
	{ 68, 47, 46},
	{ 96, 97, 99},
	{ 96, 99, 95},
	{110,109,111},
	{111,112,110},
	{114,113,123},
	{114,123,124},
	{132,131,129},
	{133,137,136},
	{135,142,145},
	{145,152,135},
	{149,147,157},
	{157,158,149},
	{164,150,151},
	{153,163,168},
	{153,168,154},
	{185,183,182},
	{185,182,184},
	{161,189,190},
	{200,199,191},
	{200,191,190},
	{180,178,195},
	{180,195,196},
	{102,101,204},
	{102,204,206},
	{ 43, 48,104},
	{ 43,104,103},
	{216,217, 54},
	{216, 54, 32},
	{207,224,231},
	{230,212,211},
	{230,211,231},
	{227,232,241},
	{227,241,242},
	{235,234,241},
	{235,241,244},
	{430,248,247},
	{272,274,253},
	{272,253,252},
	{439,260,275},
	{225,224,259},
	{225,259,257},
	{269,270,407},
	{269,407,405},
	{270,269,273},
	{270,273,272},
	{273,269,268},
	{273,268,267},
	{273,267,266},
	{273,266,265},
	{273,265,264},
	{448,279,367},
	{281,350,368},
	{285,286,301},
	{290,323,310},
	{290,311,323},
	{282,281,189},
	{292,311,290},
	{292,290,291},
	{307,306,302},
	{307,302,303},
	{316,315,324},
	{316,324,329},
	{331,351,350},
	{330,334,335},
	{330,335,328},
	{341,337,338},
	{344,355,354},
	{346,345,348},
	{346,348,347},
	{364,369,352},
	{364,352,353},
	{365,363,361},
	{365,361,362},
	{376,401,402},
	{373,372,397},
	{373,397,400},
	{376, 92,377},
	{381,378,387},
	{381,387,385},
	{386, 77, 80},
	{390,389,412},
	{416,417,401},
	{403,417,415},
	{408,429,430},
	{419,423,418},
	{427,428,444},
	{427,444,446},
	{437,436,441},
	{450,445, 11},
	{450, 11,  4},
	{447,449,  5},
	{447,  5,  8},
	{441,438,437},
	{425,426,451},
	{425,451,452},
	{417,421,415},
	{408,407,429},
	{399,403,400},
	{399,400,397},
	{394,393,416},
	{389,411,412},
	{386,383,385},
	{408,387,378},
	{408,378,406},
	{377,391,376},
	{ 94,375,415},
	{372,373,374},
	{372,374,370},
	{359,111,360},
	{359,112,111},
	{113,358,349},
	{113,349,123},
	{346,343,345},
	{343,340,342},
	{338,336,144},
	{338,144,141},
	{327,341,354},
	{327,354,326},
	{331,350,321},
	{331,321,322},
	{314,313,326},
	{314,326,325},
	{300,298,299},
	{300,299,301},
	{288,287,289},
	{189,292,282},
	{287,288,303},
	{284,285,297},
	{368,280,281},
	{448,447,279},
	{274,226,255},
	{267,268,404},
	{267,404,379},
	{429,262,430},
	{439,440,260},
	{257,258,249},
	{257,249,246},
	{430,262,248},
	{234,228,242},
	{234,242,241},
	{237,238,239},
	{237,239,236},
	{ 15, 18,227},
	{ 15,227,229},
	{222,223, 82},
	{222, 82, 83},
	{214,215,213},
	{214,213, 81},
	{ 38,102,  6},
	{122,159,200},
	{122,200,201},
	{174,171,192},
	{174,192,194},
	{197,193,198},
	{190,170,161},
	{181,179,178},
	{181,178,180},
	{166,156,155},
	{163,153,152},
	{163,152,162},
	{120,156,149},
	{120,149,121},
	{152,153,135},
	{140,143,142},
	{135,131,132},
	{135,132,136},
	{130,129,128},
	{130,128,127},
	{100,105,119},
	{100,119,120},
	{106,104,107},
	{106,107,108},
	{ 91, 95, 59},
	{ 93, 94, 68},
	{ 91, 89, 92},
	{ 76, 53, 55},
	{ 76, 55, 87},
	{ 81, 78, 79},
	{ 74, 73, 49},
	{ 69, 60, 45},
	{ 58, 62, 64},
	{ 58, 64, 61},
	{ 53, 31, 32},
	{ 32, 54, 53},
	{ 42, 43, 38},
	{ 35, 36,  0},
	{ 35,  0,  1},
	{ 34, 35,  1},
	{ 34,  1,  9},
	{ 44, 40, 41},
	{ 44, 41, 45},
	{ 33,240, 51},
	{ 63, 62, 58},
	{ 63, 58, 59},
	{ 45, 71, 70},
	{ 76, 75, 51},
	{ 76, 51, 52},
	{ 86, 85, 84},
	{ 86, 84, 87},
	{ 89, 72, 73},
	{ 89, 73, 88},
	{ 91, 92, 96},
	{ 91, 96, 95},
	{ 72, 91, 60},
	{ 72, 60, 69},
	{104,106,105},
	{119,105,117},
	{119,117,118},
	{124,127,128},
	{117,116,129},
	{117,129,131},
	{118,117,131},
	{135,140,142},
	{146,150,152},
	{146,152,145},
	{149,122,121},
	{166,165,151},
	{166,151,156},
	{158,172,173},
	{161,160,189},
	{199,198,193},
	{199,193,191},
	{204,201,202},
	{178,174,194},
	{200,159,186},
	{109, 48, 67},
	{ 48,107,104},
	{216, 32,236},
	{216,236,239},
	{223,214, 81},
	{223, 81, 82},
	{ 33, 12, 15},
	{ 32,228,234},
	{ 32,234,236},
	{240, 31, 52},
	{256,255,246},
	{256,246,249},
	{258,263,248},
	{258,248,249},
	{275,260,259},
	{275,259,276},
	{207,276,259},
	{270,271,429},
	{270,429,407},
	{413,418,366},
	{413,366,365},
	{368,367,279},
	{368,279,280},
	{303,301,286},
	{303,286,287},
	{283,282,292},
	{283,292,291},
	{320,292,189},
	{298,296,297},
	{298,297,299},
	{318,327,326},
	{318,326,313},
	{329,330,317},
	{336,333,320},
	{326,354,353},
	{334,332,333},
	{334,333,336},
	{342,339,139},
	{342,139,138},
	{345,342,126},
	{347,357,356},
	{369,368,351},
	{363,356,357},
	{363,357,361},
	{366,367,368},
	{366,368,369},
	{375,373,400},
	{ 92, 90,377},
	{409,387,408},
	{386,385,387},
	{386,387,388},
	{412,394,391},
	{396,398,399},
	{408,406,405},
	{415,421,419},
	{415,419,414},
	{425,452,448},
	{425,448,424},
	{444,441,443},
	{448,452,449},
	{448,449,447},
	{446,444,443},
	{446,443,445},
	{250,247,261},
	{250,261,428},
	{421,422,423},
	{421,423,419},
	{427,410,250},
	{417,403,401},
	{403,402,401},
	{420,392,412},
	{420,412,425},
	{420,425,424},
	{386,411,389},
	{383,382,381},
	{383,381,385},
	{378,379,404},
	{372,371,395},
	{372,395,397},
	{371,372,370},
	{361,359,360},
	{361,360,362},
	{368,350,351},
	{349,347,348},
	{356,355,344},
	{356,344,346},
	{344,341,340},
	{344,340,343},
	{338,337,336},
	{328,335,341},
	{324,352,351},
	{324,351,331},
	{320,144,336},
	{314,325,324},
	{322,308,309},
	{310,309,307},
	{287,286,289},
	{203,280,279},
	{203,279,205},
	{297,295,283},
	{297,283,284},
	{447,205,279},
	{274,384, 80},
	{274, 80,226},
	{266,267,379},
	{266,379,380},
	{225,257,246},
	{225,246,245},
	{256,254,253},
	{256,253,255},
	{430,247,250},
	{226,235,244},
	{226,244,245},
	{232,233,244},
	{232,244,241},
	{230, 18, 19},
	{ 32, 31,228},
	{219,220, 86},
	{219, 86, 57},
	{226,213,235},
	{206,  7,  6},
	{122,201,101},
	{201,204,101},
	{180,196,197},
	{170,192,171},
	{200,190,189},
	{194,193,195},
	{183,181,180},
	{183,180,182},
	{155,154,168},
	{149,156,151},
	{149,151,148},
	{155,156,120},
	{145,142,143},
	{145,143,146},
	{136,137,140},
	{133,132,130},
	{128,129,116},
	{100,120,121},
	{110,112,113},
	{110,113,114},
	{ 66, 65, 63},
	{ 66, 63, 99},
	{ 66, 99, 98},
	{ 96, 46, 61},
	{ 89, 88, 90},
	{ 86, 87, 57},
	{ 80, 78, 81},
	{ 72, 69, 49},
	{ 67, 48, 47},
	{ 67, 47, 68},
	{ 56, 55, 53},
	{ 50, 49, 36},
	{ 50, 36, 35},
	{ 40, 39, 41},
	{242,243,229},
	{242,229,227},
	{  6, 37, 39},
	{ 42, 47, 48},
	{ 42, 48, 43},
	{ 61, 46, 44},
	{ 45, 70, 69},
	{ 69, 70, 71},
	{ 69, 71, 49},
	{ 74, 78, 77},
	{ 83, 84, 85},
	{ 73, 74, 77},
	{ 93, 96, 92},
	{ 68, 46, 93},
	{ 95, 99, 63},
	{ 95, 63, 59},
	{115,108,110},
	{115,110,114},
	{125,126,127},
	{129,130,132},
	{137,133,138},
	{137,138,139},
	{148,146,143},
	{148,143,147},
	{119,118,154},
	{161,147,143},
	{165,164,151},
	{158,157,171},
	{158,171,172},
	{159,158,187},
	{159,187,186},
	{194,192,191},
	{194,191,193},
	{189,202,201},
	{182,197,184},
	{205,  8,  7},
	{ 48,109,107},
	{218,219, 57},
	{218, 57, 56},
	{207,231,211},
	{232,230,231},
	{232,231,233},
	{ 53, 52, 31},
	{388,411,386},
	{409,430,250},
	{262,429,254},
	{262,254,256},
	{442,444,428},
	{273,264,383},
	{273,383,384},
	{429,271,251},
	{429,251,254},
	{413,365,362},
	{ 67,413,360},
	{282,283,295},
	{285,301,299},
	{202,281,280},
	{284,283,291},
	{284,291,289},
	{320,189,160},
	{308,306,307},
	{307,309,308},
	{319,317,330},
	{319,330,328},
	{353,352,324},
	{332,331,333},
	{340,341,338},
	{354,341,344},
	{349,358,357},
	{349,357,347},
	{364,355,356},
	{364,356,363},
	{364,365,366},
	{364,366,369},
	{374,376,402},
	{375, 92,373},
	{ 77,389,390},
	{382,380,381},
	{389, 77,386},
	{393,394,412},
	{393,412,392},
	{401,394,416},
	{415,400,403},
	{411,410,427},
	{411,427,426},
	{422,420,424},
	{247,248,263},
	{247,263,261},
	{445,443, 14},
	{445, 14, 11},
	{449,450,  4},
	{449,  4,  5},
	{443,441, 17},
	{443, 17, 14},
	{436, 23, 17},
	{436, 17,441},
	{424,448,422},
	{448,423,422},
	{414,419,418},
	{414,418,413},
	{406,404,405},
	{399,397,395},
	{399,395,396},
	{420,416,392},
	{388,410,411},
	{386,384,383},
	{390, 88, 77},
	{375, 94, 92},
	{415,414, 68},
	{415, 68, 94},
	{370,374,402},
	{370,402,398},
	{361,357,358},
	{361,358,359},
	{125,348,126},
	{346,344,343},
	{340,338,339},
	{337,335,334},
	{337,334,336},
	{325,353,324},
	{324,331,332},
	{324,332,329},
	{323,322,309},
	{323,309,310},
	{294,295,297},
	{294,297,296},
	{289,286,285},
	{202,280,203},
	{288,307,303},
	{282,295,321},
	{ 67,360,111},
	{418,423,367},
	{418,367,366},
	{272,252,251},
	{272,251,271},
	{272,271,270},
	{255,253,274},
	{265,266,380},
	{265,380,382},
	{442,428,261},
	{440,263,258},
	{440,258,260},
	{409,250,410},
	{255,226,245},
	{255,245,246},
	{ 31,240,243},
	{236,234,235},
	{236,235,237},
	{233,225,245},
	{233,245,244},
	{220,221, 85},
	{220, 85, 86},
	{ 81,213,226},
	{ 81,226, 80},
	{  7,206,205},
	{186,184,198},
	{186,198,199},
	{204,203,205},
	{204,205,206},
	{195,193,196},
	{171,174,172},
	{173,174,175},
	{173,172,174},
	{155,167,166},
	{160,161,143},
	{160,143,144},
	{119,154,155},
	{148,151,150},
	{148,150,146},
	{140,137,139},
	{140,139,141},
	{127,126,130},
	{114,124,128},
	{114,128,115},
	{117,105,106},
	{117,106,116},
	{104,105,100},
	{104,100,103},
	{ 59, 60, 91},
	{ 97, 96, 61},
	{ 97, 61, 64},
	{ 91, 72, 89},
	{ 87, 84, 79},
	{ 87, 79, 76},
	{ 78, 80, 77},
	{ 49, 50, 74},
	{ 60, 44, 45},
	{ 61, 44, 58},
	{ 51, 50, 35},
	{ 51, 35, 34},
	{ 39, 37, 41},
	{ 33, 34,  9},
	{ 33,  9, 12},
	{  0, 36, 37},
	{  0, 37,  6},
	{ 40, 46, 47},
	{ 40, 47, 42},
	{ 53, 54, 56},
	{ 65, 62, 63},
	{ 72, 49, 73},
	{ 79, 78, 75},
	{ 79, 75, 76},
	{ 52, 53, 76},
	{ 92, 89, 90},
	{ 96, 93, 46},
	{102,103,100},
	{102,100,101},
	{116,106,108},
	{116,108,115},
	{123,125,124},
	{116,115,128},
	{118,131,135},
	{140,135,136},
	{148,147,149},
	{120,119,155},
	{164,162,152},
	{164,152,150},
	{157,147,161},
	{157,161,170},
	{186,187,185},
	{186,185,184},
	{193,197,196},
	{202,203,204},
	{194,195,178},
	{198,184,197},
	{ 67,111,109},
	{ 38, 43,103},
	{ 38,103,102},
	{214,223,222},
	{214,222,221},
	{214,221,220},
	{214,220,219},
	{214,219,218},
	{213,237,235},
	{221,222, 83},
	{221, 83, 85},
	{ 15,229, 33},
	{227, 18,230},
	{227,230,232},
	{ 52, 51,240},
	{ 75, 78, 50},
	{408,430,409},
	{260,258,257},
	{260,257,259},
	{224,207,259},
	{268,269,405},
	{268,405,404},
	{413,362,360},
	{447,  8,205},
	{299,297,285},
	{189,281,202},
	{290,288,289},
	{290,289,291},
	{322,321,295},
	{322,295,294},
	{333,323,311},
	{333,311,320},
	{317,316,329},
	{320,160,144},
	{353,325,326},
	{329,332,334},
	{329,334,330},
	{339,338,141},
	{339,141,139},
	{348,345,126},
	{347,356,346},
	{123,349,125},
	{364,353,354},
	{364,354,355},
	{365,364,363},
	{376,391,394},
	{376,394,401},
	{ 92,376,374},
	{ 92,374,373},
	{377, 90, 88},
	{380,379,378},
	{380,378,381},
	{388,387,409},
	{388,409,410},
	{416,393,392},
	{399,398,402},
	{399,402,403},
	{250,428,427},
	{421,417,416},
	{421,416,420},
	{426,427,446},
	{426,446,451},
	{444,442,441},
	{452,451,450},
	{452,450,449}
};

#pragma endregion

extern ContactAddedCallback             gContactAddedCallback;
static bool CustomMaterialCombinerCallback(btManifoldPoint& cp,	const btCollisionObjectWrapper* colObj0Wrap,int partId0,int index0,const btCollisionObjectWrapper* colObj1Wrap,int partId1,int index1)
{
	contactInfo ctinfo( cp  ,partId0, index0 ,partId1,index1 ) ;
	contactPoints.push_back(ctinfo);
//	if index == -1 ,ignore it;
// index0 or index1 is between 0 and numofface -1
//	cout<<"contact point: "<< cp.getPositionWorldOnA().getX() << " "<< cp.getPositionWorldOnA().getY() <<" "<< cp.getPositionWorldOnA().getZ()<<std::endl;
//	cout<<" partId0 " <<partId0<<"index0 "<< index0<<" partId1 " <<partId1<<"index1 "<< index1<<endl;
	if( /*index1 == 0 || index1 ==1 ||*/ index1== 6104 || index1== 6103 )
	{
		//cout<<"contact point: "<< cp.getPositionWorldOnA().getX() << " "<< cp.getPositionWorldOnA().getY() <<" "<< cp.getPositionWorldOnA().getZ()<<std::endl;
		//cout<<" partId0 " <<partId0<<"index0 "<< index0<<" partId1 " <<partId1<<"index1 "<< index1<<endl;
	}
	//glPointSize(10);
	//glColor3f(1.0f, 0.0f ,0.0f);
	//glBegin(GL_POINTS);
	//glVertex3f(cp.getPositionWorldOnA().getX(),cp.getPositionWorldOnA().getY() ,cp.getPositionWorldOnA().getZ() );
	//glEnd();

	return true;
}


btVector3	centroid=btVector3(0,0,0);
btVector3   convexDecompositionObjectOffset(10,0,0);

static void myPreTickCallback(btDynamicsWorld *world, btScalar timeStep)
{
	contactPoints.clear();
	int i= 0;
	int numManifolds = world->getDispatcher()->getNumManifolds();
	for (i=0;i<numManifolds;i++)
	{
		btPersistentManifold* contactManifold = world->getDispatcher()->getManifoldByIndexInternal(i);
		btCollisionObject* obA = (btCollisionObject*)(contactManifold->getBody0());
		btCollisionObject* obB = (btCollisionObject*)(contactManifold->getBody1());
		btRigidBody* bodyA = btRigidBody::upcast(obA);
		btRigidBody* bodyB = btRigidBody::upcast(obB);
		btVector3 gravityOfA,gravityOfB;
		if(bodyA)
		{	
			gravityOfA = bodyA->getGravity();
			//bodyA->applyGravity();
		}
		if(bodyB)
		{
			gravityOfB = bodyB->getGravity();
			//bodyB->applyGravity();
		}
		//printf("gravityOfA: %d %d %d  \n",gravityOfA.x() , gravityOfA.y(), gravityOfA.z());
		//printf("gravityOfB: %d %d %d  \n",gravityOfB.x() , gravityOfB.y(), gravityOfB.z());
		int numContacts = contactManifold->getNumContacts();
		for (int j=0;j<numContacts;j++)
		{
			btManifoldPoint& contactPoint = contactManifold->getContactPoint(j);
			btVector3 normal = contactPoint.m_normalWorldOnB;
			//btScalar angleX = normal.angle(btVector3(1,0,0));
			//btScalar angleY = normal.angle(btVector3(0,1,0));
			//btScalar angleZ = normal.angle(btVector3(0,0,1));
			//btScalar impulseX = contactPoint.m_appliedImpulse*cos(angleX);
			//btScalar impulseY = contactPoint.m_appliedImpulse*cos(angleY);
			//btScalar impulseZ = contactPoint.m_appliedImpulse*cos(angleZ);
			//btScalar forceX = impulseX/(timeStep);
			//btScalar forceY = impulseY/(timeStep);
			////btScalar forceZ = impulseZ/(timeStep);
			//printf("Force: %8.6f %8.6f %8.6f %8.6f \n",(float)timeStep,forceX,forceY,forceZ);
			if( abs(contactPoint.m_appliedImpulse) > 0.001)
			{
				//printf("pre numManifolds: %d  numContacts: %d contactid: %d impluse %8.6f \n",numManifolds,numContacts, j, contactPoint.m_appliedImpulse);

			}
		}
	}




}

static void myPosTickCallback(btDynamicsWorld *world, btScalar timeStep)
{

	//cout<< "contact points size  "<<contactPoints.size()<<endl;
	int i= 0;
	int numManifolds = world->getDispatcher()->getNumManifolds();
	for (i=0;i<numManifolds;i++)
	{
		btPersistentManifold* contactManifold = world->getDispatcher()->getManifoldByIndexInternal(i);
		btCollisionObject* obA = (btCollisionObject*)(contactManifold->getBody0());
		btCollisionObject* obB = (btCollisionObject*)(contactManifold->getBody1());
		btRigidBody* bodyA = btRigidBody::upcast(obA);
		btRigidBody* bodyB = btRigidBody::upcast(obB);
		btVector3 gravityOfA,gravityOfB;
		int indexOfA ,indexOfB;
		if(bodyA)
		{	
			gravityOfA = bodyA->getGravity();
			//bodyA->applyGravity();
			indexOfA = bodyA->getUserIndex();
		}
		if(bodyB)
		{
			gravityOfB = bodyB->getGravity();
			//bodyB->applyGravity();
			indexOfB = bodyB->getUserIndex();
		}
		if( !(indexOfA == 1 || indexOfB ==1))
			continue;

		//printf("index a: %d  index b: %d \n",indexOfA ,indexOfB);
		//printf("gravityOfA: %d %d %d  \n",gravityOfA.x() , gravityOfA.y(), gravityOfA.z());
		//printf("gravityOfB: %d %d %d  \n",gravityOfB.x() , gravityOfB.y(), gravityOfB.z());
		int numContacts = contactManifold->getNumContacts();
		std::vector<int>  triangle_indexs(numContacts,0);
		g_vertex_indexs.clear();
		g_impluses.clear();
		for (int j=0;j<numContacts;j++)
		{
			btManifoldPoint& contactPoint = contactManifold->getContactPoint(j);
			btVector3 normal = contactPoint.m_normalWorldOnB;
			normal.normalize();
			btVector3& impuluseVector = -contactPoint.m_appliedImpulse * normal;
			//btScalar angleX = normal.angle(btVector3(1,0,0));
			//btScalar angleY = normal.angle(btVector3(0,1,0));
			//btScalar angleZ = normal.angle(btVector3(0,0,1));
			//btScalar impulseX = contactPoint.m_appliedImpulse*cos(angleX);
			//btScalar impulseY = contactPoint.m_appliedImpulse*cos(angleY);
			//btScalar impulseZ = contactPoint.m_appliedImpulse*cos(angleZ);
			//btScalar forceX = impulseX/(timeStep);
			//btScalar forceY = impulseY/(timeStep);
			////btScalar forceZ = impulseZ/(timeStep);
			//printf("Force: %8.6f %8.6f %8.6f %8.6f \n",(float)timeStep,forceX,forceY,forceZ);
			if( abs(contactPoint.m_appliedImpulse) > 0.001)
			{
				//printf("pos numManifolds: %d  numContacts: %d contactid: %d impluse %8.6f \n",numManifolds,numContacts, j, contactPoint.m_appliedImpulse);

			}
			const btVector3& forceOrigin = contactPoint.getPositionWorldOnA();
			const btVector3& forceEnd = forceOrigin + impuluseVector;
			glLineWidth(10);
			glBegin( GL_LINES);
			glVertex3f(forceOrigin.x(),forceOrigin.y() ,forceOrigin.z() );
			glVertex3f(forceEnd.x(),forceEnd.y() ,forceEnd.z() );
			glEnd();

			//find the nearest triangle
			float distance = 1000.0f;
			float best_index = 0 ;
			if (!contactPoints.size())
				return;
			for (int v_j = 0; v_j < contactPoints.size(); ++v_j)
			{
				btVector3 vb = contactPoints[v_j].contactPoints.getPositionWorldOnA()- contactPoint.getPositionWorldOnA();
				if( vb.length() < distance)
				{
					distance = vb.length();
					best_index = v_j;
				}
			}
			triangle_indexs[j] = best_index;
			int target_index = best_index;
			btManifoldPoint& cp = contactPoints[ best_index ].contactPoints;
			int target_tri_index =0;
			int idex = 0;
			if( contactPoints[target_index].index0 >= 0 && Glmesh_Ori)
			{
				target_tri_index = 3* contactPoints[target_index].index0;
				idex = (*Glmesh_Ori->m_indices)[  target_tri_index];
				g_vertex_indexs.push_back(idex);
				g_impluses.push_back(impuluseVector.getX());
				g_impluses.push_back(impuluseVector.getY());
				g_impluses.push_back(impuluseVector.getZ());
			}
			if( contactPoints[target_index].index1 >= 1 && Glmesh_Ori)
			{
				target_tri_index = 3* contactPoints[target_index].index1;
				idex = (*Glmesh_Ori->m_indices)[  target_tri_index];
				g_vertex_indexs.push_back(idex);
				g_impluses.push_back(impuluseVector.getX());
				g_impluses.push_back(impuluseVector.getY());
				g_impluses.push_back(impuluseVector.getZ());
			}
			 
		}

		return ;

		//cout<<"triangle index size  "<< triangle_indexs.size() <<endl;
		g_vertex_indexs.clear();
		g_impluses.clear();

//#ifndef use_4_point
		for( int i_trangle = 0 ; i_trangle < triangle_indexs.size() ; ++i_trangle) 
		{
			if(contactPoints.size()> triangle_indexs.size())
			{
				int target_index = triangle_indexs[i_trangle];
//#else
//		for( int i_trangle = 0 ; i_trangle < contactPoints.size() ; ++i_trangle) 
//		{
//
//#endif
				//int target_index = i_trangle;
				btManifoldPoint& cp = contactPoints[ target_index ].contactPoints;

				glPointSize(10);
				glColor3f(1.0f, 0.0f ,0.0f);
				glBegin(GL_POINTS);
				glVertex3f(cp.getPositionWorldOnA().getX(),cp.getPositionWorldOnA().getY() ,cp.getPositionWorldOnA().getZ() );
				glEnd();

				if( contactPoints[target_index].index0 >= 0 && Glmesh_Ori)
				{
					int target_tri_index = 3* contactPoints[target_index].index0;
					int idex = (*Glmesh_Ori->m_indices)[  target_tri_index];
//					btScalar impulse = cp.m_appliedImpulse > 0.1 ? cp.m_appliedImpulse > 0.9f ?0.9f :cp.m_appliedImpulse :0.1;
					btScalar impulse = cp.m_appliedImpulse;
					btVector3 normal = -cp.m_normalWorldOnB;
					normal.normalize();
					btVector3 impulse2 = impulse*normal;
//					Tbx::Vec3 impulse_vec = Tbx::Vec3( impulse2.getX(),impulse2.getX(),impulse2.getX()); 
					g_vertex_indexs.push_back(idex);
					g_impluses.push_back(impulse2.getX());
					g_impluses.push_back(impulse2.getY());
					g_impluses.push_back(impulse2.getZ());
				}


				if( contactPoints[target_index].index1 >= 0 && Glmesh_Ori)
				{
					int target_tri_index = 3* contactPoints[target_index].index1;
					int idex = (*Glmesh_Ori->m_indices)[  target_tri_index];
//					btScalar impulse = cp.m_appliedImpulse > 0.1 ? cp.m_appliedImpulse > 0.9f ?0.9f :cp.m_appliedImpulse :0.1;
					btScalar impulse = cp.m_appliedImpulse;
					btVector3 normal = -cp.m_normalWorldOnB;
					normal.normalize();
					btVector3 impulse2 = impulse*normal;
					//					Tbx::Vec3 impulse_vec = Tbx::Vec3( impulse2.getX(),impulse2.getX(),impulse2.getX()); 
					g_vertex_indexs.push_back(idex);
					g_impluses.push_back(impulse2.getX());
					g_impluses.push_back(impulse2.getY());
					g_impluses.push_back(impulse2.getZ());
				}

#pragma region
				if( contactPoints[target_index].index0 >= 0 && Glmesh_Ori &&0)
				{


					int target_tri_index = 3* contactPoints[target_index].index0;
					//					cout<<"triangle index1 "<< contactPoints[target_index].index1<<endl;
					btVector3 vertex1,vertex2,vertex3;
					float vertex[3][3];
					for( int i_vertex = 0; i_vertex < 3 ;++i_vertex)
					{
						for( int j_vertex = 0; j_vertex < 3 ;++j_vertex)
						{
							int idex = (*Glmesh_Ori->m_indices)[  target_tri_index];
							vertex[i_vertex][j_vertex] = (*Glmesh_Ori->m_vertices)[ idex +i ].xyzw[j_vertex];
						}
					}

					vertex1.setValue(vertex[0][0],vertex[0][1],vertex[0][2]);
					vertex2.setValue(vertex[1][0],vertex[1][0],vertex[1][0]);
					vertex3.setValue(vertex[2][0],vertex[2][0],vertex[2][0]);

					//					cout<<"impulse before clamp "<<cp.m_appliedImpulse<<endl;
					btScalar impulse = cp.m_appliedImpulse > 0.1 ? cp.m_appliedImpulse > 0.9f ?0.9f :cp.m_appliedImpulse :0.1;

					//  				cout<<"impulse clamp "<<impulse<<endl;

					btVector3 normal = -cp.m_normalWorldOnB;
					normal.normalize();
					vertex1 += impulse*normal;
					vertex2 += impulse*normal;
					vertex3 += impulse*normal;
					float after_vertex[3][3];
					after_vertex[0][0] = vertex1.getX();
					after_vertex[0][1] = vertex1.getY();
					after_vertex[0][2] = vertex1.getZ();
					after_vertex[1][0] = vertex2.getX();
					after_vertex[1][1] = vertex2.getY();
					after_vertex[1][2] = vertex2.getZ();
					after_vertex[2][0] = vertex3.getX();
					after_vertex[2][1] = vertex3.getY();
					after_vertex[2][2] = vertex3.getZ();
					for( int i_vertex = 0; i_vertex < 3 ;++i_vertex)
					{
						for( int j_vertex = 0; j_vertex < 3 ;++j_vertex)
						{
							int idex = (*Glmesh_Ori->m_indices)[  target_tri_index];
							(*Glmesh_Ori->m_vertices)[ idex +i ].xyzw[j_vertex] = after_vertex[i_vertex][j_vertex];
						}
					}

					glBegin(GL_TRIANGLES);
					glVertex3f(vertex[0][0],vertex[0][1],vertex[0][2]);
					glVertex3f(vertex[1][0],vertex[1][0],vertex[1][0]);
					glVertex3f(vertex[2][0],vertex[2][0],vertex[2][0]);
					glEnd();
					//					cout<<"triangle index1 positon  "<< x1 <<" "<< y1<<" "<< z1 <<" "<< x2 <<" "<< y2 <<" "<<z2 <<" "<<x3<<" "<<y3 <<" "<<z3 <<endl;
				}
				if( contactPoints[target_index].index1 >= 0 && Glmesh_Ori && 0)
				{
					int target_tri_index = 3* contactPoints[target_index].index1;
//					cout<<"triangle index1 "<< contactPoints[target_index].index1<<endl;
					btVector3 vertex1,vertex2,vertex3;
					float vertex[3][3];
					for( int i_vertex = 0; i_vertex < 3 ;++i_vertex)
					{
						for( int j_vertex = 0; j_vertex < 3 ;++j_vertex)
						{
							int idex = (*Glmesh_Ori->m_indices)[  target_tri_index];
							vertex[i_vertex][j_vertex] = (*Glmesh_Ori->m_vertices)[ idex +i ].xyzw[j_vertex];
						}
					}

					vertex1.setValue(vertex[0][0],vertex[0][1],vertex[0][2]);
					vertex2.setValue(vertex[1][0],vertex[1][0],vertex[1][0]);
					vertex3.setValue(vertex[2][0],vertex[2][0],vertex[2][0]);

//					cout<<"impulse before clamp "<<cp.m_appliedImpulse<<endl;
					btScalar impulse = cp.m_appliedImpulse > 0.1 ? cp.m_appliedImpulse > 0.9f ?0.9f :cp.m_appliedImpulse :0.1;
//  				cout<<"impulse clamp "<<impulse<<endl;
	
					btVector3 normal = -cp.m_normalWorldOnB;
					normal.normalize();
					vertex1 += impulse*normal;
					vertex2 += impulse*normal;
					vertex3 += impulse*normal;
					float after_vertex[3][3];
					after_vertex[0][0] = vertex1.getX();
					after_vertex[0][1] = vertex1.getY();
					after_vertex[0][2] = vertex1.getZ();
					after_vertex[1][0] = vertex2.getX();
					after_vertex[1][1] = vertex2.getY();
					after_vertex[1][2] = vertex2.getZ();
					after_vertex[2][0] = vertex3.getX();
					after_vertex[2][1] = vertex3.getY();
					after_vertex[2][2] = vertex3.getZ();
					for( int i_vertex = 0; i_vertex < 3 ;++i_vertex)
					{
						for( int j_vertex = 0; j_vertex < 3 ;++j_vertex)
						{
							int idex = (*Glmesh_Ori->m_indices)[  target_tri_index];
							(*Glmesh_Ori->m_vertices)[ idex +i ].xyzw[j_vertex] = after_vertex[i_vertex][j_vertex];
						}
					}

					glBegin(GL_TRIANGLES);
					glVertex3f(vertex[0][0],vertex[0][1],vertex[0][2]);
					glVertex3f(vertex[1][0],vertex[1][0],vertex[1][0]);
					glVertex3f(vertex[2][0],vertex[2][0],vertex[2][0]);
					glEnd();
//					cout<<"triangle index1 positon  "<< x1 <<" "<< y1<<" "<< z1 <<" "<< x2 <<" "<< y2 <<" "<<z2 <<" "<<x3<<" "<<y3 <<" "<<z3 <<endl;
				}
#pragma  endregion
				//Glmesh->m_numIndices;
//#ifndef use_4_point
			}
		}
//#else
//		}
//#endif
//		}
		//convertOriObjForRender( Glmesh ,*convert_Glmesh);
		//exportObj( Glmesh ,"./resource/bulletoutput/");

	}
}
/*
enum ObjToRigidBodyOptionsEnum
{
	ObjUseConvexHullForRendering=1,
	OptimizeConvexObj=2,
	ComputePolyhedralFeatures=4,
};
*/
class MyConvexDecomposition : public ConvexDecomposition::ConvexDecompInterface
{
	RigidbodyDemo*	m_convexDemo;

public:

	btAlignedObjectArray<btConvexHullShape*> m_convexShapes;
	btAlignedObjectArray<btVector3> m_convexCentroids;

	MyConvexDecomposition (FILE* outputFile, RigidbodyDemo* demo)
		:m_convexDemo(demo),
		mBaseCount(0),
		mHullCount(0),
		mOutputFile(outputFile)

	{
	}

	virtual void ConvexDecompResult(ConvexDecomposition::ConvexResult &result)
	{

		btTriangleMesh* trimesh = new btTriangleMesh();
		m_convexDemo->m_trimeshes.push_back(trimesh);

		btVector3 localScaling(1.1f,1.1f,1.1f);

		//export data to .obj
		printf("ConvexResult. ");
		if (mOutputFile)
		{
			fprintf(mOutputFile,"## Hull Piece %d with %d vertices and %d triangles.\r\n", mHullCount, result.mHullVcount, result.mHullTcount );

			fprintf(mOutputFile,"usemtl Material%i\r\n",mBaseCount);
			fprintf(mOutputFile,"o Object%i\r\n",mBaseCount);

			for (unsigned int i=0; i<result.mHullVcount; i++)
			{
				const float *p = &result.mHullVertices[i*3];
				fprintf(mOutputFile,"v %0.9f %0.9f %0.9f\r\n", p[0], p[1], p[2] );
			}

			//calc centroid, to shift vertices around center of mass
			centroid.setValue(0,0,0);

			btAlignedObjectArray<btVector3> vertices;
			if ( 1 )
			{
				//const unsigned int *src = result.mHullIndices;
				for (unsigned int i=0; i<result.mHullVcount; i++)
				{
					btVector3 vertex(result.mHullVertices[i*3],result.mHullVertices[i*3+1],result.mHullVertices[i*3+2]);
					vertex *= localScaling;
					centroid += vertex;

				}
			}

			centroid *= 1.f/(float(result.mHullVcount) );

			if ( 1 )
			{
				//const unsigned int *src = result.mHullIndices;
				for (unsigned int i=0; i<result.mHullVcount; i++)
				{
					btVector3 vertex(result.mHullVertices[i*3],result.mHullVertices[i*3+1],result.mHullVertices[i*3+2]);
					vertex *= localScaling;
					vertex -= centroid ;
					vertices.push_back(vertex);
				}
			}



			if ( 1 )
			{
				const unsigned int *src = result.mHullIndices;
				for (unsigned int i=0; i<result.mHullTcount; i++)
				{
					unsigned int index0 = *src++;
					unsigned int index1 = *src++;
					unsigned int index2 = *src++;


					btVector3 vertex0(result.mHullVertices[index0*3], result.mHullVertices[index0*3+1],result.mHullVertices[index0*3+2]);
					btVector3 vertex1(result.mHullVertices[index1*3], result.mHullVertices[index1*3+1],result.mHullVertices[index1*3+2]);
					btVector3 vertex2(result.mHullVertices[index2*3], result.mHullVertices[index2*3+1],result.mHullVertices[index2*3+2]);
					vertex0 *= localScaling;
					vertex1 *= localScaling;
					vertex2 *= localScaling;

					vertex0 -= centroid;
					vertex1 -= centroid;
					vertex2 -= centroid;


					trimesh->addTriangle(vertex0,vertex1,vertex2);

					index0+=mBaseCount;
					index1+=mBaseCount;
					index2+=mBaseCount;

					fprintf(mOutputFile,"f %d %d %d\r\n", index0+1, index1+1, index2+1 );
				}
			}

			//	float mass = 1.f;


			//this is a tools issue: due to collision margin, convex objects overlap, compensate for it here:
			//#define SHRINK_OBJECT_INWARDS 1
#ifdef SHRINK_OBJECT_INWARDS

			float collisionMargin = 0.01f;

			btAlignedObjectArray<btVector3> planeEquations;
			btGeometryUtil::getPlaneEquationsFromVertices(vertices,planeEquations);

			btAlignedObjectArray<btVector3> shiftedPlaneEquations;
			for (int p=0;p<planeEquations.size();p++)
			{
				btVector3 plane = planeEquations[p];
				plane[3] += collisionMargin;
				shiftedPlaneEquations.push_back(plane);
			}
			btAlignedObjectArray<btVector3> shiftedVertices;
			btGeometryUtil::getVerticesFromPlaneEquations(shiftedPlaneEquations,shiftedVertices);


			btConvexHullShape* convexShape = new btConvexHullShape(&(shiftedVertices[0].getX()),shiftedVertices.size());

#else //SHRINK_OBJECT_INWARDS

			btConvexHullShape* convexShape = new btConvexHullShape(&(vertices[0].getX()),vertices.size());
#endif 
			if (0)
				convexShape->initializePolyhedralFeatures();
			convexShape->setMargin(0.01f);
			m_convexShapes.push_back(convexShape);
			m_convexCentroids.push_back(centroid);
			m_convexDemo->m_collisionShapes.push_back(convexShape);
			mBaseCount+=result.mHullVcount; // advance the 'base index' counter.


		}
	}

	int   	mBaseCount;
	int		mHullCount;
	FILE*	mOutputFile;

};






RigidbodyDemo::RigidbodyDemo(PaintCanvas* qgl,const std::string _filename)
	:BasicDemo(qgl)
{
	m_fileName = _filename;
	m_options = 4;
}

RigidbodyDemo::~RigidbodyDemo()
{

}


void RigidbodyDemo::InitializePhysics()
{
	createEmptyDynamicsWorld();
	s_pWorld->setInternalTickCallback(myPreTickCallback,0,true);
	s_pWorld->setInternalTickCallback(myPosTickCallback,0,false);
	//CreateObjObjectConvexDecomp();
//	CreateObjObject2();
	CreateGimpactObject();
}

void RigidbodyDemo::ShutdownPhysics()
{

}

void RigidbodyDemo::UpdateScene(float dt)
{
	BulletOpenGLApplication::UpdateScene(dt);

	return;
	int shapeId = 0;
	//registerGraphicsShape( (btScalar*)( &glmesh->m_vertices->at(0).xyzw[0] ), 
	//	glmesh->m_numvertices, 
	//	&glmesh->m_indices->at(0), 
	//	glmesh->m_numIndices,
	//	B3_GL_TRIANGLES, -1);
//	int num_instance = m_renderer->getTotalNumInstances();
	std::vector<float> defromed_vertices;
	std::vector<int> vertex_indexs = g_vertex_indexs;
	std::vector<float> impluses_objectSpace =g_impluses;
	btRigidBody* pkeg = (btRigidBody*)m_RigidBodys[1];
	btTransform pTransForm = pkeg->getWorldTransform();
	btQuaternion quanterion = pTransForm.getRotation();
	btTransform rotation(quanterion);
	for (int i = 0; i < impluses_objectSpace.size()/3; i++)
	{
		btVector3 ip(impluses_objectSpace[3*i] ,impluses_objectSpace[3*i+1],impluses_objectSpace[3*i+2]);
		ip = rotation.inverse() * ip;
		impluses_objectSpace[3*i] = ip.getX();
		impluses_objectSpace[3*i+1] = ip.getY();
		impluses_objectSpace[3*i+2] = ip.getZ();
	}

	
	float invermass = 1.0f/pkeg->getInvMass();
	float mass = 100.0f;
	if(invermass < 1e-3)
	{
		mass = 100.0f;
	}else
	{
		mass = 1./invermass;
	}
	
	float belta = 0.1f;
	Cuda_ctrl::_example_mesh.genertateVertices(defromed_vertices,g_faces,vertex_indexs,impluses_objectSpace,mass,dt,belta );
	convertVecForRencer( defromed_vertices ,g_faces, *GlmeshForRender );
	static int count = 0;
	++count;
	if( count >60)
	{
		count = 0;
		exportObj( defromed_vertices ,g_faces,"resource/bulletoutput/keg_out.obj");
	}
	
//	convertOriObjForRender( Glmesh ,*convert_Glmesh);
//	exportObj( Glmesh ,"./resource/bulletoutput/");

	//for (int i = 0; i < defromed_vertices.size()/3; i++)
	//{
	//	convert_Glmesh->m_vertices->at(i).xyzw[0] = defromed_vertices[3*i+0];
	//	convert_Glmesh->m_vertices->at(i).xyzw[1] = defromed_vertices[3*i+1];
	//	convert_Glmesh->m_vertices->at(i).xyzw[2] = defromed_vertices[3*i+2];
	//}

//	if(Glmesh)
//		m_renderer->updateShape( shapeId,&convert_Glmesh->m_vertices->at(0).xyzw[0] );

#pragma region
	if( m_RigidBodys.size()>1 && Glmesh_Ori && 0)
	{
		int obj_user_index1 = m_RigidBodys[0]->getUserIndex();
		int shape_user_index1 = m_RigidBodys[0]->getCollisionShape()->getUserIndex();
		int obj_user_index2 = m_RigidBodys[1]->getUserIndex();
		int shape_user_index2 = m_RigidBodys[1]->getCollisionShape()->getUserIndex();
//		updateGraphyObj( m_RigidBodys[0]);
//		updateGraphyObj( m_RigidBodys[1]);
		
		btRigidBody* prev_rigidbody = (btRigidBody*)(m_RigidBodys[1]);
		float inversmass = prev_rigidbody->getInvMass();
		btCollisionShape* pre_collisionshape = prev_rigidbody->getCollisionShape();
		btTransform pre_transdform = prev_rigidbody->getWorldTransform();
		int pre_user_idex = prev_rigidbody->getUserIndex();
		btMotionState* prevmotionstate = prev_rigidbody->getMotionState();

//		s_pWorld->removeRigidBody( (btRigidBody*)(m_RigidBodys[1]) );
//		delete prev_rigidbody;

			// create trimesh
		btScalar* pvertices = new btScalar[Glmesh_Ori->m_numvertices*3];
		for ( int i = 0 ; i < Glmesh_Ori->m_numvertices ;++i)
		{
			GLInstanceVertex& v0 = Glmesh_Ori->m_vertices->at(i);
			//printf("sizeof  GLInstanceVertex %d",sizeof(v0));
			pvertices[3*i +0]  = v0.xyzw[0];
			pvertices[3*i +1]  = v0.xyzw[1];
			pvertices[3*i +2]  = v0.xyzw[2];
			//		printf("%d vetex indices %8.6f %8.6f %8.6f %\n" ,i , pvertices[3*i +0],pvertices[3*i +1],pvertices[3*i +2]);
		}
		int* indices = new int[Glmesh_Ori->m_numIndices];
		for (int i = 0; i < Glmesh_Ori->m_numIndices; i++)
		{
			indices[i] = Glmesh_Ori->m_indices->at(i);
			//		printf("%d indices %d\n" ,i , indices[i]);
		}

		btTriangleIndexVertexArray* m_indexVertexArrays = new btTriangleIndexVertexArray( Glmesh_Ori->m_numIndices/3,
			//&glmesh->m_indices->at(0),
			indices,
			3*sizeof(int),
			//946,
			Glmesh_Ori->m_numvertices,
			(REAL*) pvertices,sizeof(REAL)*3);

		btCollisionObjectArray &obarr= s_pWorld->getCollisionObjectArray();
		btCollisionObject *ob;
		btBroadphaseProxy* proxy;

		for(int i= 0; i < obarr.size(); i++) {
			ob= obarr[i];
			if (ob->getCollisionShape() == pre_collisionshape){
				proxy = obarr[i]->getBroadphaseHandle();

				if(proxy)
					s_pWorld->getPairCache()->cleanProxyFromPairs(proxy,s_pWorld->getDispatcher());
			}
		}



		btGImpactMeshShape * trimesh = new btGImpactMeshShape(m_indexVertexArrays);
		btVector4 color(1.0f,0.0f,0.0f,0.0f);		
//		m_RigidBodys[1]->setCollisionShape(trimesh);
		trimesh->postUpdate();
		btVector3 inertial;
		trimesh->calculateLocalInertia( 1.0f/inversmass,inertial);
		((btRigidBody*)m_RigidBodys[1])->setMassProps(1.0f/inversmass,inertial);
		((btRigidBody*)m_RigidBodys[1])->updateInertiaTensor();
//		s_pWorld->addRigidBody((btRigidBody*)m_RigidBodys[1]);
//		btRigidBody* new_rigidbody = createRigidBody(1.0f/inversmass , pre_transdform ,trimesh,color,pre_user_idex,prevmotionstate );
//		new_rigidbody->setCollisionFlags(new_rigidbody->getCollisionFlags()  | btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK);  //add collison callback
//		m_RigidBodys[1] = new_rigidbody;
//		m_RigidBodys[1]->setCollisionShape(trimesh);
		//delete pvertices;
		//delete indices;
		//delete m_indexVertexArrays;



	}
#pragma  endregion




	//if(m_trimeshShape)
	//	((btGImpactMeshShape*)m_trimeshShape)->postUpdate();
}
#define  _gim_pact_
#ifndef  _gim_pact_
void RigidbodyDemo::CreateObjObject()
{
	btBoxShape* groundShape = createBoxShape(btVector3(btScalar(50.),btScalar(50.),btScalar(50.)));
	m_collisionShapes.push_back(groundShape);
	btTransform groundTransform;
	groundTransform.setIdentity();
	groundTransform.setOrigin(btVector3(0,-50,0));

	{
		btScalar mass(0.);
		createRigidBody(mass,groundTransform,groundShape, btVector4(0,0,1,1));
	}




// 	s_pWorld->getDebugDrawer()->setDebugMode((btIDebugDraw::DBG_DrawAabb));
	btTransform trans;
	trans.setIdentity();
	trans.setRotation(btQuaternion(btVector3(1,0,0),SIMD_HALF_PI));
	btVector3 position = trans.getOrigin();
	btQuaternion orn = trans.getRotation();

	btVector3 scaling(1,1,1);
	btVector3 color(1,1,1);


	btBoxShape* colShape = createBoxShape(btVector3(.1,.1,.1));
	btScalar	mass(1.f);
	m_collisionShapes.push_back(colShape);

	//rigidbody is dynamic if and only if mass is non zero, otherwise static
	bool isDynamic = (mass != 0.f);

	btVector3 localInertia(0,0,0);
	if (isDynamic)
		colShape->calculateLocalInertia(mass,localInertia);

	int shapeId = loadAndRegisterMeshFromFile2(m_fileName);    
	if (shapeId>=0)
	{
		//int id = 
		m_renderer->registerGraphicsInstance(shapeId,position,orn,color,scaling);
	} 

	autogenerateGraphicsObjects( s_pWorld);
}

void RigidbodyDemo::CreateObjObject2()
{

	btBoxShape* groundShape = createBoxShape(btVector3(btScalar(5.),btScalar(0.2),btScalar(5.)));
	m_collisionShapes.push_back(groundShape);

	btTransform groundTransform;
	groundTransform.setIdentity();
	groundTransform.setOrigin(btVector3(0,0.0,0)); 
	{
		btScalar mass(0.);
		createRigidBody(mass,groundTransform,groundShape, btVector4(0,0,1,1));
	}



	GLInstanceGraphicsShape* glmesh = LoadMeshFromObj(m_fileName.c_str(), "");
	Glmesh_Ori = glmesh;
	printf("[INFO] Obj loaded: Extracted %d verticed from obj file [%s]\n", glmesh->m_numvertices, m_fileName.c_str());

	const GLInstanceVertex& v = glmesh->m_vertices->at(0);
	btConvexHullShape* shape = new btConvexHullShape((const btScalar*)(&(v.xyzw[0])), glmesh->m_numvertices, sizeof(GLInstanceVertex));
	btTransform origintransform;
	origintransform.setIdentity();
	btVector3  aabbmin;
	btVector3 aabbmax;
	shape->getAabb(origintransform, aabbmin,aabbmax);
	btScalar maxScale = (aabbmax.x() -aabbmin.x()) > (aabbmax.y() -aabbmin.y()) ? ( (aabbmax.x() -aabbmin.x())>(aabbmax.z() -aabbmin.z())?(aabbmax.x() -aabbmin.x()):(aabbmax.z() -aabbmin.z()) ) :
		(aabbmax.y() -aabbmin.y())>(aabbmax.z() -aabbmin.z())?(aabbmax.y() -aabbmin.y()):(aabbmax.z() -aabbmin.z());

	shape->getAabb( origintransform , aabbmin ,aabbmax);
	float scaling[4] = {0.5/maxScale,0.5/maxScale,0.5/maxScale,1};

	btVector3 localScaling(scaling[0],scaling[1],scaling[2]);
	shape->setLocalScaling(localScaling);

	if (m_options & OptimizeConvexObj)
	{
		shape->optimizeConvexHull();
	}

	if (m_options & ComputePolyhedralFeatures)
	{
		shape->initializePolyhedralFeatures();    
	}



	//shape->setMargin(0.001);
	m_collisionShapes.push_back(shape);

	btTransform startTransform;
	startTransform.setIdentity();

	btScalar	mass(1.f);
	bool isDynamic = (mass != 0.f);
	btVector3 localInertia(0,0,0);
	if (isDynamic)
		shape->calculateLocalInertia(mass,localInertia);

	float color[4] = {1,1,1,1};
	float orn[4] = {0,0,0,1};
	float pos[4] = {0,1,0,0};
	btVector3 position(pos[0],pos[1],pos[2]);
	startTransform.setOrigin(position);
	btRigidBody* body = createRigidBody(mass,startTransform,shape);



	bool useConvexHullForRendering = ((m_options & ObjUseConvexHullForRendering)!=0);


	if (!useConvexHullForRendering)
	{
		int shapeId = registerGraphicsShape(&glmesh->m_vertices->at(0).xyzw[0], 
			glmesh->m_numvertices, 
			&glmesh->m_indices->at(0), 
			glmesh->m_numIndices,
			B3_GL_TRIANGLES, -1);
		shape->setUserIndex(shapeId);
		int renderInstance = m_renderer->registerGraphicsInstance(shapeId,pos,orn,color,scaling);
		body->setUserIndex(renderInstance);
	}

if(0)
{
	btBoxShape* colShape = createBoxShape(btVector3(.1,.1,.1));


	//btCollisionShape* colShape = new btSphereShape(btScalar(1.));
	m_collisionShapes.push_back(colShape);

	for (int k=0;k<ARRAY_SIZE_Y;k++)
	{
		for (int i=0;i<ARRAY_SIZE_X;i++)
		{
			for(int j = 0;j<ARRAY_SIZE_Z;j++)
			{
				startTransform.setOrigin(btVector3(
					btScalar(0.2*i),
					btScalar(2+.2*k),
					btScalar(0.2*j)));


				createRigidBody(mass,startTransform,colShape);


			}
		}
	}

}

	autogenerateGraphicsObjects(s_pWorld);

}

void RigidbodyDemo::CreateObjObjectConvexDecomp()
{

	ConvexDecomposition::WavefrontObj wo;
	wo.loadObj(m_fileName.c_str());


	btTransform startTransform;
	startTransform.setIdentity();
	startTransform.setOrigin(btVector3(0,-4.5,0));

	btCollisionShape* boxShape = new btBoxShape(btVector3(30,2,30));
	m_collisionShapes.push_back(boxShape);
	btScalar	mass(0.f);
	createRigidBody(0.f,startTransform,boxShape);
	{
		btTriangleMesh* trimesh = new btTriangleMesh();
		m_trimeshes.push_back(trimesh);

		btVector3 localScaling(1.1f,1.1f,1.1f);

		int i;
		for ( i=0;i<wo.mTriCount;i++)
		{
			int index0 = wo.mIndices[i*3];
			int index1 = wo.mIndices[i*3+1];
			int index2 = wo.mIndices[i*3+2];

			btVector3 vertex0(wo.mVertices[index0*3], wo.mVertices[index0*3+1],wo.mVertices[index0*3+2]);
			btVector3 vertex1(wo.mVertices[index1*3], wo.mVertices[index1*3+1],wo.mVertices[index1*3+2]);
			btVector3 vertex2(wo.mVertices[index2*3], wo.mVertices[index2*3+1],wo.mVertices[index2*3+2]);

			vertex0 *= localScaling;
			vertex1 *= localScaling;
			vertex2 *= localScaling;

			trimesh->addTriangle(vertex0,vertex1,vertex2);
		}


		btConvexShape* tmpConvexShape = new btConvexTriangleMeshShape(trimesh);

		printf("old numTriangles= %d\n",wo.mTriCount);
		printf("old numIndices = %d\n",wo.mTriCount*3);
		printf("old numVertices = %d\n",wo.mVertexCount);

		printf("reducing vertices by creating a convex hull\n");

		//create a hull approximation
		btShapeHull* hull = new btShapeHull(tmpConvexShape);
		btScalar margin = tmpConvexShape->getMargin();
		hull->buildHull(margin);
		tmpConvexShape->setUserPointer(hull);


		printf("new numTriangles = %d\n", hull->numTriangles ());
		printf("new numIndices = %d\n", hull->numIndices ());
		printf("new numVertices = %d\n", hull->numVertices ());

		btConvexHullShape* convexShape = new btConvexHullShape();
		bool updateLocalAabb = false;

		for (i=0;i<hull->numVertices();i++)
		{
			convexShape->addPoint(hull->getVertexPointer()[i],updateLocalAabb);	
		}
		convexShape->recalcLocalAabb();

		if (m_options & ComputePolyhedralFeatures)
			convexShape->initializePolyhedralFeatures();
		delete tmpConvexShape;
		delete hull;



		m_collisionShapes.push_back(convexShape);

		float mass = 1.f;

		btTransform startTransform;
		startTransform.setIdentity();
		startTransform.setOrigin(btVector3(0,2,14));

		createRigidBody(mass, startTransform,convexShape);

		bool useQuantization = true;
		btCollisionShape* concaveShape = new btBvhTriangleMeshShape(trimesh,useQuantization);
		startTransform.setOrigin(convexDecompositionObjectOffset);
		createRigidBody(0.f,startTransform,concaveShape);

		m_collisionShapes.push_back (concaveShape);
	}

	{
				//-----------------------------------
		// Bullet Convex Decomposition
		//-----------------------------------

		char outputFileName[512];
  		strcpy(outputFileName,m_fileName.c_str());
  		char *dot = strstr(outputFileName,".");
  		if ( dot ) 
			*dot = 0;
		strcat(outputFileName,"_convex.obj");
  		FILE* outputFile = fopen(outputFileName,"wb");
				
		unsigned int depth = 5;
		float cpercent     = 5;
		float ppercent     = 15;
		unsigned int maxv  = 16;
		float skinWidth    = 0.0;

//		printf("WavefrontObj num triangles read %i\n",tcount);
		ConvexDecomposition::DecompDesc desc;
		desc.mVcount       = wo.mVertexCount;
		desc.mVertices     = wo.mVertices;
		desc.mTcount       = wo.mTriCount;
		desc.mIndices      = (unsigned int *)wo.mIndices;
		desc.mDepth        = depth;
		desc.mCpercent     = cpercent;
		desc.mPpercent     = ppercent;
		desc.mMaxVertices  = maxv;
		desc.mSkinWidth    = skinWidth;

		MyConvexDecomposition	convexDecomposition(outputFile,this);
		desc.mCallback = &convexDecomposition;


		//-----------------------------------------------
		// HACD
		//-----------------------------------------------

		std::vector< HACD::Vec3<HACD::Real> > points;
		std::vector< HACD::Vec3<long> > triangles;

		for(int i=0; i<wo.mVertexCount; i++ ) 
		{
			int index = i*3;
			HACD::Vec3<HACD::Real> vertex(wo.mVertices[index], wo.mVertices[index+1],wo.mVertices[index+2]);
			points.push_back(vertex);
		}

		for(int i=0;i<wo.mTriCount;i++)
		{
			int index = i*3;
			HACD::Vec3<long> triangle(wo.mIndices[index], wo.mIndices[index+1], wo.mIndices[index+2]);
			triangles.push_back(triangle);
		}


		HACD::HACD myHACD;
		myHACD.SetPoints(&points[0]);
		myHACD.SetNPoints(points.size());
		myHACD.SetTriangles(&triangles[0]);
		myHACD.SetNTriangles(triangles.size());
		myHACD.SetCompacityWeight(0.1);
		myHACD.SetVolumeWeight(0.0);

		// HACD parameters
		// Recommended parameters: 2 100 0 0 0 0
		size_t nClusters = 2;
		double concavity = 100;
		bool invert = false;
		bool addExtraDistPoints = false;
		bool addNeighboursDistPoints = false;
		bool addFacesPoints = false;       

		myHACD.SetNClusters(nClusters);                     // minimum number of clusters
		myHACD.SetNVerticesPerCH(100);                      // max of 100 vertices per convex-hull
		myHACD.SetConcavity(concavity);                     // maximum concavity
		myHACD.SetAddExtraDistPoints(addExtraDistPoints);   
		myHACD.SetAddNeighboursDistPoints(addNeighboursDistPoints);   
		myHACD.SetAddFacesPoints(addFacesPoints); 

		myHACD.Compute();
		nClusters = myHACD.GetNClusters();	

		myHACD.Save("output.wrl", false);


		//convexDecomposition.performConvexDecomposition(desc);

//		ConvexBuilder cb(desc.mCallback);
//		cb.process(desc);
		//now create some bodies
		
		if (1)
		{
			btCompoundShape* compound = new btCompoundShape();
			m_collisionShapes.push_back (compound);

			btTransform trans;
			trans.setIdentity();

			for (int c=0;c<nClusters;c++)
			{
				//generate convex result
				size_t nPoints = myHACD.GetNPointsCH(c);
				size_t nTriangles = myHACD.GetNTrianglesCH(c);

				float* vertices = new float[nPoints*3];
				unsigned int* triangles = new unsigned int[nTriangles*3];
				
				HACD::Vec3<HACD::Real> * pointsCH = new HACD::Vec3<HACD::Real>[nPoints];
				HACD::Vec3<long> * trianglesCH = new HACD::Vec3<long>[nTriangles];
				myHACD.GetCH(c, pointsCH, trianglesCH);

				// points
				for(size_t v = 0; v < nPoints; v++)
				{
					vertices[3*v] = pointsCH[v].X();
					vertices[3*v+1] = pointsCH[v].Y();
					vertices[3*v+2] = pointsCH[v].Z();
				}
				// triangles
				for(size_t f = 0; f < nTriangles; f++)
				{
					triangles[3*f] = trianglesCH[f].X();
					triangles[3*f+1] = trianglesCH[f].Y();
					triangles[3*f+2] = trianglesCH[f].Z();
				}

				delete [] pointsCH;
				delete [] trianglesCH;

				ConvexDecomposition::ConvexResult r(nPoints, vertices, nTriangles, triangles);
				convexDecomposition.ConvexDecompResult(r);
			}

			for (int i=0;i<convexDecomposition.m_convexShapes.size();i++)
			{
				btVector3 centroid = convexDecomposition.m_convexCentroids[i];
				trans.setOrigin(centroid);
				btConvexHullShape* convexShape = convexDecomposition.m_convexShapes[i];
				compound->addChildShape(trans,convexShape);

				btRigidBody* body;
				body = createRigidBody( 1.0, trans,convexShape);
			}
/*			for (int i=0;i<convexDecomposition.m_convexShapes.size();i++)
			{
				
				btVector3 centroid = convexDecomposition.m_convexCentroids[i];
				trans.setOrigin(centroid);
				btConvexHullShape* convexShape = convexDecomposition.m_convexShapes[i];
				compound->addChildShape(trans,convexShape);

				btRigidBody* body;
				body = localCreateRigidBody( 1.0, trans,convexShape);
			}*/

#if 1
			btScalar mass=10.f;
			trans.setOrigin(-convexDecompositionObjectOffset);
			btRigidBody* body = createRigidBody( mass, trans,compound);
			body->setCollisionFlags(body->getCollisionFlags() |   btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK);

			convexDecompositionObjectOffset.setZ(6);
			trans.setOrigin(-convexDecompositionObjectOffset);
			body = createRigidBody( mass, trans,compound);
			body->setCollisionFlags(body->getCollisionFlags() |   btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK);

			convexDecompositionObjectOffset.setZ(-6);
			trans.setOrigin(-convexDecompositionObjectOffset);
			body = createRigidBody( mass, trans,compound);
			body->setCollisionFlags(body->getCollisionFlags() |   btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK);
#endif
		}

		
		if (outputFile)
			fclose(outputFile);
	}

	autogenerateGraphicsObjects(s_pWorld);
}
#endif
void RigidbodyDemo::CreateGimpactObject()
{

	gContactAddedCallback = CustomMaterialCombinerCallback;


	btBoxShape* groundShape = createBoxShape(btVector3(btScalar(50.),btScalar(0.2),btScalar(50.)));
	m_collisionShapes.push_back(groundShape);
	btRigidBody* groundObject;
	btTransform groundTransform;
	groundTransform.setIdentity();
	groundTransform.setOrigin(btVector3(0,0.0,0)); 
	{
		btScalar mass(0.);
		groundObject = createRigidBody(mass,groundTransform,groundShape, btVector4(0,0,1,1));
		m_RigidBodys.push_back(groundObject);
	}

	groundObject->setCollisionFlags(groundObject->getCollisionFlags()|btCollisionObject::CF_STATIC_OBJECT);

	//enable custom material callback
 	groundObject->setCollisionFlags(groundObject->getCollisionFlags()|btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK);


	GLInstanceGraphicsShape* glmesh = NULL;
	if(0)
	{
		glmesh= LoadMeshFromObj(m_fileName.c_str(), "");
		printf("[INFO] Obj loaded: Extracted %d verticed from obj file [%s]\n", glmesh->m_numvertices, m_fileName.c_str());
	}
	else
	{
		if(1)
		{
			importObj(  glmesh ,  m_fileName );  //this is the right way to import
		}else
		{
			//this is for test,not use
			glmesh = new GLInstanceGraphicsShape;
			b3AlignedObjectArray<GLInstanceVertex>* vertices = new b3AlignedObjectArray<GLInstanceVertex>;
			b3AlignedObjectArray<int>* indicesPtr = new b3AlignedObjectArray<int>;

			for (int i = 0; i < NUM_VERTICES; i++)
			{
				GLInstanceVertex v0;
				v0.xyzw[0] =	gVertices[ i*3 +0];
				v0.xyzw[1] =	gVertices[ i*3 +1];
				v0.xyzw[2] =	gVertices[ i*3 +2];
				v0.xyzw[3] = 0.f;
				vertices->push_back(v0);

			}

			for (int i = 0; i < NUM_TRIANGLES; i++)
			{
				indicesPtr->push_back(gIndices[i][0]) ;
				indicesPtr->push_back(gIndices[i][1]);
				indicesPtr->push_back(gIndices[i][2]);
			}

			glmesh->m_vertices = vertices;
			glmesh->m_numvertices = vertices->size();
			glmesh->m_indices = indicesPtr;
			glmesh->m_numIndices = indicesPtr->size();
			for (int i=0;i<4;i++)
				glmesh->m_scaling[i] = 1;//bake the scaling into the vertices
		}
	}

	Cuda_ctrl::_example_mesh.setupExample("D:/mprojects/EBPD/ebpd/PCM/resource/meshes/keg3/","keg_skinning" );


	Glmesh_Ori = glmesh;
	exportObj(glmesh ,"./resource/bulletoutput/");
	const GLInstanceVertex& v = glmesh->m_vertices->at(0);
	//btConvexHullShape* shape = new btConvexHullShape((const btScalar*)(&(v.xyzw[0])), glmesh->m_numvertices, sizeof(GLInstanceVertex));

	GLInstanceGraphicsShape* convert_glmesh = new GLInstanceGraphicsShape();
	convertOriObjForRender(glmesh ,*convert_glmesh);
	exportObj(convert_glmesh ,"./resource/bulletoutput/");
	glmesh = convert_glmesh;
	GlmeshForRender = convert_glmesh;
	// create trimesh
	btScalar* pvertices = new btScalar[glmesh->m_numvertices*3];
	for ( int i = 0 ; i < glmesh->m_numvertices ;++i)
	{
		GLInstanceVertex& v0 = glmesh->m_vertices->at(i);
		//printf("sizeof  GLInstanceVertex %d",sizeof(v0));
		pvertices[3*i +0]  = v0.xyzw[0];
		pvertices[3*i +1]  = v0.xyzw[1];
		pvertices[3*i +2]  = v0.xyzw[2];
//		printf("%d vetex indices %8.6f %8.6f %8.6f %\n" ,i , pvertices[3*i +0],pvertices[3*i +1],pvertices[3*i +2]);
	}
	int* indices = new int[glmesh->m_numIndices];
	for (int i = 0; i < glmesh->m_numIndices; i++)
	{
		indices[i] = glmesh->m_indices->at(i);
//		printf("%d indices %d\n" ,i , indices[i]);
	}

	//btTriangleIndexVertexArray* m_indexVertexArrays = new btTriangleIndexVertexArray(NUM_TRIANGLES,
	//	&gIndices[0][0],
	//	3*sizeof(int),
	//NUM_VERTICES,(REAL*) &gVertices[0],sizeof(REAL)*3);
	//btTriangleIndexVertexArray* m_indexVertexArrays = new btTriangleIndexVertexArray(glmesh->m_numIndices/3,
	//	indices,
	//	3*sizeof(int),
	//	NUM_VERTICES,(REAL*) &gVertices[0],sizeof(REAL)*3);
	btTriangleIndexVertexArray* m_indexVertexArrays = new btTriangleIndexVertexArray( glmesh->m_numIndices/3,
		//&glmesh->m_indices->at(0),
		indices,
		3*sizeof(int),
		//946,
		glmesh->m_numvertices,
		(REAL*) pvertices,sizeof(REAL)*3);


	btGImpactMeshShape * trimesh = new btGImpactMeshShape(m_indexVertexArrays);
//	float scaling[4] =  {1.04f,1.04f,1.04f ,1.0f};
	float scaling[4] =  {1.00f,1.00f,1.00f ,1.0f};
	btVector3 localScaling(scaling[0],scaling[1],scaling[2]);
	trimesh->setLocalScaling(localScaling);

	
	trimesh->updateBound();
if(0)
{	btTransform origintransform;
	origintransform.setIdentity();
	btVector3  aabbmin;
	btVector3 aabbmax;
	trimesh->getAabb(origintransform, aabbmin,aabbmax);
	btScalar maxScale = (aabbmax.x() -aabbmin.x()) > (aabbmax.y() -aabbmin.y()) ? ( (aabbmax.x() -aabbmin.x())>(aabbmax.z() -aabbmin.z())?(aabbmax.x() -aabbmin.x()):(aabbmax.z() -aabbmin.z()) ) :
		(aabbmax.y() -aabbmin.y())>(aabbmax.z() -aabbmin.z())?(aabbmax.y() -aabbmin.y()):(aabbmax.z() -aabbmin.z());
	float scaling[4] = {2.0/maxScale,2.0/maxScale,2.0/maxScale,1};

	btVector3 localScaling(scaling[0],scaling[1],scaling[2]);
	trimesh->setLocalScaling(localScaling);
	trimesh->updateBound();
}

	if(0)
	{	m_trimeshShape= btCreateCompoundFromGimpactShape(trimesh,1);
	    delete trimesh;
		trimesh = 0;
	}else
	{
		m_trimeshShape = trimesh;
	}


	m_collisionShapes.push_back(m_trimeshShape);


	//register algorithm

	btCollisionDispatcher * dispatcher = static_cast<btCollisionDispatcher *>( s_pWorld ->getDispatcher());
	btGImpactCollisionAlgorithm::registerAlgorithm(dispatcher);

	btTransform startTransform;
	startTransform.setIdentity();

	btScalar	mass(1.f);
	bool isDynamic = (mass != 0.f);
	btVector3 localInertia(0,0,0);
	//if (isDynamic)
	//	trimesh->calculateLocalInertia(mass,localInertia);


	float color[4] = {1,1,1,1};
	//float quaterion[4] = {0,0,0,1};
	float quaterion[4] = {0,0,1,-0.8};
	float pos[4] = {0,4,0,0};
	btVector3 position(pos[0],pos[1],pos[2]);
	btVector3 axis(0,0,1);
	btQuaternion rotation(axis,-0.8);
	quaterion[0] = rotation.getX();
	quaterion[1] = rotation.getY();
	quaterion[2] = rotation.getZ();
	quaterion[3] = rotation.getW();
	startTransform.setOrigin(position);
	startTransform.setRotation(rotation);
	btRigidBody* body = createRigidBody(mass,startTransform,m_trimeshShape);
	m_RigidBodys.push_back(body);

	body->setCollisionFlags(body->getCollisionFlags()  | btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK);  //add collison callback

	if (0)
	{
		//int shapeId = registerGraphicsShape(&glmesh->m_vertices->at(0).xyzw[0], 
		//	glmesh->m_numvertices, 
		//	&glmesh->m_indices->at(0), 
		//	glmesh->m_numIndices,
		//	B3_GL_TRIANGLES, -1);
		//int& a = glmesh->m_indices->at(0);
		//btTriangleIndexVertexArray* m_indexVertexArrays = new btTriangleIndexVertexArray( glmesh->m_numIndices,&a,3* sizeof(int),
		//									glmesh->m_numvertices,(btScalar*)(&(v.xyzw[0])),sizeof(GLInstanceVertex));
		//btTriangleIndexVertexArray* m_indexVertexArrays = new btTriangleIndexVertexArray(NUM_TRIANGLES,
		//	&gIndices[0][0],
		//	3*sizeof(int),
		//	NUM_VERTICES,(REAL*) &gVertices[0],sizeof(REAL)*3);
		b3AlignedObjectArray<GLInstanceVertex>* vertices = new b3AlignedObjectArray<GLInstanceVertex>;
		for ( int i = 0 ; i < NUM_VERTICES ;++i)
		{
			GLInstanceVertex v0 ;
			//printf("sizeof  GLInstanceVertex %d",sizeof(v0));
			v0.xyzw[0] = gVertices[3*i +0];
			v0.xyzw[1] = gVertices[3*i +1];
			v0.xyzw[2] = gVertices[3*i +2];
			v0.xyzw[3] = 0.0f;
			v0.normal[0] = 0.0f;
			v0.normal[1] = 0.0f;
			v0.normal[2] = 0.0f;
			vertices->push_back(v0);
		}
		int* indeces = new int[ 3* NUM_TRIANGLES];
		for (int i = 0; i <NUM_TRIANGLES; i++)
		{
			indeces[ 3*i+0] = gIndices[i][0];
			indeces[ 3*i+1] = gIndices[i][1];
			indeces[ 3*i+2] = gIndices[i][2];
		}
		//int shapeId = registerGraphicsShape( (btScalar*)(& (vertices[0][0].xyzw[0]) ), 
		//	NUM_VERTICES, 
		//	&gIndices[0][0], 
		//	NUM_INDICES,
		//	B3_GL_TRIANGLES, -1);

		int shapeId = registerGraphicsShape( (btScalar*)( &glmesh->m_vertices->at(0).xyzw[0] ), 
			glmesh->m_numvertices, 
			&glmesh->m_indices->at(0), 
			glmesh->m_numIndices,
			B3_GL_TRIANGLES, -1);
		m_trimeshShape->setUserIndex(shapeId);
		int renderInstance = m_renderer->registerGraphicsInstance(shapeId,pos,quaterion,color,scaling);
		body->setUserIndex(renderInstance);
	}


	autogenerateGraphicsObjects(s_pWorld);

}

int RigidbodyDemo::loadAndRegisterMeshFromFile2(const std::string& fileName)
{
	int shapeId = -1;

	b3ImportMeshData meshData;
	if (b3ImportMeshUtility::loadAndRegisterMeshFromFileInternal(fileName, meshData))
	{
		int textureIndex = -1;

		if (meshData.m_textureImage)
		{
			textureIndex = m_renderer->registerTexture(meshData.m_textureImage,meshData.m_textureWidth,meshData.m_textureHeight);
		}

		shapeId = m_renderer->registerShape(&meshData.m_gfxShape->m_vertices->at(0).xyzw[0], 
			meshData.m_gfxShape->m_numvertices, 
			&meshData.m_gfxShape->m_indices->at(0), 
			meshData.m_gfxShape->m_numIndices,
			B3_GL_TRIANGLES,
			textureIndex);
		delete meshData.m_gfxShape;
		delete meshData.m_textureImage;
	}
	return shapeId;
}



