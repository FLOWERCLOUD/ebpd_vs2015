#include "ExampleWeightSover.h"
#include "toolbox/maths/quat_cu.hpp"
#include "solver/fadiff.h"
#include "solver/badiff.h"
#include <iostream>

using std::cout;
using std::endl;
using namespace  fadbad;
std::vector< B<F<float>> > ExampleSover::Skinning_function( std::vector<B<F<float>> >& ori_exampleWeights ,int num_example ,int vertex_idex , bool isQlerp)
{
	//std::vector<float> example_weights(num_example,0.0f);
	//for (int i = 0; i < num_example; i++)
	//{
	//	example_weights[i] = ori_exampleWeights[i].x().val();
	//}
//	generateSkinningVetex(vertex_idex,vtx,example_weights,isQlerp);
	std::vector< B<F<float>> > result(3);
	std::vector< B<F<float>> > result2(3);
	result2 = generateSkinningVetex(vertex_idex,result,ori_exampleWeights,isQlerp);


	return result;

}

B<F<float>> sq(const B<F<float>> &x) { return x * x; }
std::vector< B<F<float>> > Skinning_function2( std::vector<B<F<float>> >& ori_exampleWeights, int k )
{
	std::vector<float> example_weights(ori_exampleWeights.size(),0.0f);

	std::vector< B<F<float>> > result(3,0.0f);
	for (int i = 0; i <3; i++)
	{
		for (int j = 0; j < ori_exampleWeights.size(); j++)
		{
			result[i] += (i+1)*(j+1)*sq(ori_exampleWeights[j]+1)*k;
		}
		
	}
	double vtx[3];
	vtx[0] = result[0].x().val();
	vtx[1] = result[1].x().val();
	vtx[2] = result[2].x().val();
	result[0].diff(0,3);
	result[1].diff(1,3);
	result[2].diff(2,3);
	std::vector< B<F<float>> > result2(3,0.0f);
	result2[0] = vtx[0];
	result2[1] = vtx[1];
	result2[2] = vtx[2];

	return result2;

}





bool ExampleSover::SolveVertices(const std::map<int,Tbx::Vec3>& delta_xi, std::map<int, std::vector<float> >& delta_exampleWeightsOfVertex, std::map<int, std::vector<float> >& ori_exampleWeights)
{
	bool isQlerp = true;
	auto iter = delta_xi.begin();
	for (int i = 0; iter != delta_xi.end(); i++, ++iter)
	{
		int i_vertex = iter->first;
		Tbx::Vec3 delta_xi = iter->second;

		float x = m_inputVertices[3*i_vertex];
		float y = m_inputVertices[3*i_vertex+1];
		float z = m_inputVertices[3*i_vertex+2];
		Tbx::Point3 cur_point(x,y,z);
		Tbx::Point3 acc_point;
		const std::vector<float> exmaple_weights =  ori_exampleWeights[i_vertex];
		generateSkinningVetex( i_vertex ,acc_point,exmaple_weights,isQlerp);
		std::vector<float> delata_example;
		solver(i_vertex,acc_point,delta_xi,exmaple_weights,delata_example ,isQlerp);
		delta_exampleWeightsOfVertex[i_vertex] = delata_example;

	}
	return true;

}

static void  quat_to_matrix3(B<F<float>> quat[4] , B<F<float>> mat3[9]  ) 
{
	B<F<float>>* coeff = quat;
	B<F<float>> W = coeff[0], X = -coeff[1], Y = -coeff[2], Z = -coeff[3];
	B<F<float>> xx = X * X, xy = X * Y, xz = X * Z, xw = X * W;
	B<F<float>> yy = Y * Y, yz = Y * Z, yw = Y * W, zz = Z * Z;
	B<F<float>> zw = Z * W;
	mat3[0]=1.f - 2.f * (yy + zz);
	mat3[1]=2.f * (xy + zw);
	mat3[2]=2.f * (xz - yw);
	mat3[3]=2.f * (xy - zw);
	mat3[4]=1.f - 2.f * (xx + zz);
	mat3[5]=2.f * (yz + xw);
	mat3[6]=2.f * (xz + yw);
	mat3[7]=2.f * (yz - xw);
	mat3[8]=1.f - 2.f * (xx + yy);
	//Mat3 mat = Mat3(
	//	1.f - 2.f * (yy + zz),      2.f * (xy + zw),       2.f * (xz - yw),
	//	2.f * (xy - zw),1.f - 2.f * (xx + zz),       2.f * (yz + xw),
	//	2.f * (xz + yw),      2.f * (yz - xw), 1.f - 2.f * (xx + yy)
	//	);
	//return mat;
}
static void  matrix3_to_quat(B<F<float>> mat3[9]  ,  B<F<float>> quat[4] ) 
{
	B<F<float>>* coeff = quat;
	// Compute trace of matrix 't'
	B<F<float>> T = 1 + mat3[0] + mat3[4] + mat3[8];

	B<F<float>> S, X, Y, Z, W;

	if ( T > 0.00000001f ) // to avoid large distortions!
	{
		S = sqrt(T) * 2.f;
		X = ( mat3[5] - mat3[7] ) / S;
		Y = ( mat3[6] - mat3[2] ) / S;
		Z = ( mat3[1] - mat3[3] ) / S;
		W = 0.25f * S;
	}
	else
	{
		if ( mat3[0] > mat3[4] && mat3[0] > mat3[8] )
		{
			// Column 0 :
			S  = sqrt( 1.0f + mat3[0] - mat3[4] - mat3[8] ) * 2.f;
			X = 0.25f * S;
			Y = (mat3[1] + mat3[3] ) / S;
			Z = (mat3[6] + mat3[2] ) / S;
			W = (mat3[5] - mat3[7] ) / S;
		}
		else if ( mat3[4] > mat3[8] )
		{
			// Column 1 :
			S  = sqrt( 1.0f + mat3[4] - mat3[0] - mat3[8] ) * 2.f;
			X = (mat3[1] + mat3[3] ) / S;
			Y = 0.25f * S;
			Z = (mat3[5] + mat3[7] ) / S;
			W = (mat3[6] - mat3[2] ) / S;
		}
		else
		{   // Column 2 :
			S  = sqrt( 1.0f + mat3[8] - mat3[0] - mat3[4] ) * 2.f;
			X = (mat3[6] + mat3[2] ) / S;
			Y = (mat3[5] + mat3[7] ) / S;
			Z = 0.25f * S;
			W = (mat3[1] - mat3[3] ) / S;
		}
	}

	coeff[0] = W; coeff[1] = -X; coeff[2] = -Y; coeff[3] = -Z;

}
static void  matrix3_copy_from_a_to_b( const B<F<float>> mat3a[9] , B<F<float>> mat3b[9]  ) 
{
	for (int i = 0; i < 9; i++)
	{
		mat3b[i] = mat3a[i];
	}
}
static void transformPoint(const B<F<float>> rotate[9],const B<F<float>> translate[3],
							const B<F<float>> oripoint[3],B<F<float>> transformedPoint[3] )
{
	transformedPoint[0] = rotate[0]*oripoint[0] + rotate[1]*oripoint[1]+rotate[2]*oripoint[2]+ translate[0];
	transformedPoint[1] = rotate[3]*oripoint[0] + rotate[4]*oripoint[1]+rotate[5]*oripoint[2]+ translate[1];
	transformedPoint[2] = rotate[6]*oripoint[0] + rotate[7]*oripoint[1]+rotate[8]*oripoint[2]+ translate[2];
}
//inline Point3 operator*(const Point3& v) const {
//	return Point3(
//		m[0] * v.x + m[1] * v.y + m[ 2] * v.z + m[ 3],
//		m[4] * v.x + m[5] * v.y + m[ 6] * v.z + m[ 7],
//		m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11]);
//}


std::vector<B<F<float>> > ExampleSover::generateSkinningVetex(int vertex_idex ,std::vector<B<F<float>> >& vtx, std::vector<B<F<float>> >& ori_exampleWeights , bool isQlerp)
{
	int i_vertex = vertex_idex;
	float x = m_inputVertices[3*i_vertex];
	float y = m_inputVertices[3*i_vertex+1];
	float z = m_inputVertices[3*i_vertex+2];
	//Tbx::Point3 cur_point0(x,y,z);
	B<F<float>> cur_point[3]; cur_point[0] = x;cur_point[1] = y;cur_point[2] = z;        //B<F<Tbx::Point3>> cur_point =cur_point0 ;
	B<F<float>> acc_point[3]; //B<F<Tbx::Point3>> acc_point;


	for (int i_indice= 0; i_indice < m_numbIndices ;++i_indice)
	{
		int i_bone = m_boneWightIdx[i_vertex*m_numbIndices+i_indice];

		B<F<float>> acc_translate[3];//B<F<Tbx::Vec3>> acc_translate;
		B<F<float>> acc_rotate[9]; //B<F<Tbx::Mat3>> acc_rotate;
		B<F<float>> acc_rotate_quat[4];//B<F<Tbx::Quat_cu>> acc_rotate_quat;
		for (int i_example = 0 ;i_example< m_numExample ;++i_example)
		{

			const Tbx::Transfo& transfo = m_transfosOfExamples[i_example*m_numBone+i_bone];

			B<F<float>> translate[3]; translate[0] = transfo.m[3];translate[1] = transfo.m[7];translate[2] = transfo.m[11];//B<F<Tbx::Vec3>> translate = transfo.get_translation();
			B<F<float>> rotate[9]; 
			rotate[0] = transfo.m[0];rotate[1] = transfo.m[1];rotate[2] = transfo.m[2];
			rotate[3] = transfo.m[4];rotate[4] = transfo.m[5];rotate[5] = transfo.m[6];
			rotate[6] = transfo.m[8];rotate[7] = transfo.m[9];rotate[8] = transfo.m[10];//B<F<Tbx::Mat3>> rotate = transfo.get_mat3();

			B<F<float>> rotate_quat[4]; matrix3_to_quat(rotate,rotate_quat);//;rotate_quat[0] =   B<F<Tbx::Quat_cu>> rotate_quat(transfo);
			B<F<float>> exmaple_weight = ori_exampleWeights[i_example];
			for (int i = 0; i < 3; i++)
			{
				acc_translate[i] += translate[i]* exmaple_weight;
			}
			
			if(0==i_example)
			{
				//let it identity
				for (int i = 0; i < 4; i++)
				{
					acc_rotate_quat[i] = rotate_quat[i]*exmaple_weight;
				}
				//acc_rotate_quat = rotate_quat.x().val()*exmaple_weight.x().val();
			}
			else
			{
				for (int i = 0; i < 4; i++)
				{
					acc_rotate_quat[i] = acc_rotate_quat[i]+ rotate_quat[i]*exmaple_weight;
				}

				//B<F<Tbx::Quat_cu>> temp = rotate_quat.x().val()*exmaple_weight.x().val();
				//acc_rotate_quat = acc_rotate_quat + temp;
			}
		}
		/*if(isQlerp)
			acc_rotate_quat = acc_rotate_quat.x().val()/acc_rotate_quat.x().val().norm(); */
		if(isQlerp)
		{
			B<F<float>> norm = sqrt(acc_rotate_quat[0]*acc_rotate_quat[0] +
				acc_rotate_quat[1]*acc_rotate_quat[1] +
				acc_rotate_quat[2]*acc_rotate_quat[2] +
				acc_rotate_quat[3]*acc_rotate_quat[3]);
			for (int i = 0; i < 4; i++)
			{
				acc_rotate_quat[i] = acc_rotate_quat[i]/ norm;
			}
		}
		 
		//acc_rotate = acc_rotate_quat.x().val().to_matrix3();
		quat_to_matrix3(acc_rotate_quat , acc_rotate  ) ;

		//Tbx::Transfo final_transfo( acc_rotate.x().val() ,acc_translate.x().val() 

		float bone_weight = m_boneWeights[i_vertex*m_numbIndices+i_indice];
		B<F<float>> transformed_point[3];
		transformPoint(acc_rotate,acc_translate,
			cur_point,transformed_point );
		for (int i = 0; i < 3; i++)
		{
			acc_point[i] += bone_weight* transformed_point[i];
		}
		//B<F<Tbx::Point3>> temp = (bone_weight * (final_transfo*cur_point.x().val())).to_point3();
		//acc_point += temp;
	}
	std::vector<B<F<float>> > result(3);
	for (int i = 0; i < 3; i++)
	{
		vtx[i] = acc_point[i];
		result[i] = acc_point[i];
	}
	return result;
}


bool ExampleSover::generateSkinningVetex(int vertex_idex ,Tbx::Point3& vtx, const std::vector<float>& ori_exampleWeights , bool isQlerp)
{
	int i_vertex = vertex_idex;
	float x = m_inputVertices[3*i_vertex];
	float y = m_inputVertices[3*i_vertex+1];
	float z = m_inputVertices[3*i_vertex+2];
	Tbx::Point3 cur_point(x,y,z);
	Tbx::Point3 acc_point;


	for (int i_indice= 0; i_indice < m_numbIndices ;++i_indice)
	{
		int i_bone = m_boneWightIdx[i_vertex*m_numbIndices+i_indice];

		Tbx::Vec3 acc_translate;
		Tbx::Mat3 acc_rotate;
		Tbx::Quat_cu acc_rotate_quat;
		for (int i_example = 0 ;i_example< m_numExample ;++i_example)
		{

			const Tbx::Transfo& transfo = m_transfosOfExamples[i_example*m_numBone+i_bone];

			Tbx::Vec3& translate = transfo.get_translation();
			Tbx::Mat3& rotate = transfo.get_mat3();
			Tbx::Quat_cu rotate_quat(transfo);
			const float exmaple_weight = ori_exampleWeights[i_example];
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
		float bone_weight = m_boneWeights[i_vertex*m_numbIndices+i_indice];
		acc_point = acc_point + bone_weight * (final_transfo*cur_point);
	}
	vtx = acc_point;
	return true;
}

bool ExampleSover::solver(int vertex_idex ,Tbx::Point3& vtx, Tbx::Vec3& delta , const std::vector<float>& ori_example ,std::vector<float>& delata_example,bool isQlerp)
{

	float final_resiual = 100;
	int max_iter_count = 1;
	int iter_count = 0;
	int num_example = ori_example.size();
	Tbx::Point3 target_point = vtx + delta;
	Tbx::Vec3 new_delta = delta;
	std::vector<float> inter_oriexample(ori_example);
	std::vector<float> new_oriexample(ori_example);
	while( iter_count < max_iter_count && final_resiual > 0.01f)
	{
		++iter_count;
		cout<<"itercount "<<iter_count<<endl;
		std::vector<B<F<float>> > fad__exampleWeights( num_example);
		for (int i = 0; i < m_numExample; i++)
		{
			fad__exampleWeights[i] = inter_oriexample[i];
			cout<<"fad__exampleWeights"<<"["<<i<<"].x().x()"<< fad__exampleWeights[i].x().x()<< endl;
			fad__exampleWeights[i].x().diff(i,m_numExample);
		}
		int k = 100;
		std::vector<B<F<float>> > result = Skinning_function( fad__exampleWeights ,m_numExample ,vertex_idex  , isQlerp);
		//std::vector<B<F<float>> > result = Skinning_function2( fad__exampleWeights,k);
		for (int i = 0; i < 3; i++)
		{
			result[i].diff(i,3);
			/*result[i].diff(0,1);*/
			cout<< "result["<<i<<"]: "<<result[i].x().x()<<endl;
		}
		std::vector<float> jacobian_x(m_numExample);
		std::vector<float> jacobian_y(m_numExample);
		std::vector<float> jacobian_z(m_numExample);
		for (int i = 0; i < m_numExample; i++)
		{
			jacobian_x[i] = fad__exampleWeights[i].d(0).x();
			jacobian_y[i] = fad__exampleWeights[i].d(1).x();
			jacobian_z[i] = fad__exampleWeights[i].d(2).x();
		}

		delata_example.resize(m_numExample,0.0f);
		float alpha = 50.1f;   //alpha decide the step size
		float square_norm_delta_example = 0.0f;
		float square_norm_delta = 0.0f;

		for (int i = 0; i < m_numExample; i++)
		{
			
			delata_example[i] = jacobian_x[i]* new_delta[0] + jacobian_y[i]* new_delta[1] +jacobian_z[i]* new_delta[2];
			square_norm_delta_example += delata_example[i]*delata_example[i];

		}
		square_norm_delta = new_delta[0]*new_delta[0] +new_delta[1]*new_delta[1]+new_delta[2]*new_delta[2];
		square_norm_delta = sqrt(square_norm_delta);
		square_norm_delta_example = sqrt( square_norm_delta_example);
		float epsilon = 1e-8;//avoid devide zero
		for (int i = 0; i < m_numExample; i++)
		{
			
			if(square_norm_delta_example > epsilon ) 
			{
				delata_example[i] = alpha* delata_example[i]*square_norm_delta/square_norm_delta_example;
			}else
			{
				delata_example[i] = 0.0f;
			}
		}


		float weight_sum = 0.0f;
		float max_weight = -100.0f;
		float min_weight = 100.0f;
		for (int i = 0; i < m_numExample; i++)
		{
			new_oriexample[i] = inter_oriexample[i] + delata_example[i];
			if( new_oriexample[i] < min_weight)
				min_weight = new_oriexample[i];
			if( new_oriexample[i] > max_weight)
				max_weight = new_oriexample[i];
			weight_sum += new_oriexample[i];

		}
		//clamp weight sum to 1
		for (int i = 0; i < m_numExample; i++)
		{
			//new_oriexample[i] += -min_weight;
			//new_oriexample[i] /= (max_weight-min_weight);
			new_oriexample[i] /= weight_sum;
			inter_oriexample[i] = new_oriexample[i];
		}
		
		
		Tbx::Point3 new_point;
		generateSkinningVetex( vertex_idex ,new_point,new_oriexample,isQlerp);
		new_delta = target_point - new_point;
		//Tbx::Vec3 residual = new_point - vtx -delta; // try to minimize this residual
		Tbx::Vec3 residual = new_delta;
		float residual_value = residual.norm();
		if (residual_value < final_resiual)
		{
			 cout<<"residual_value "<<residual_value<<"< "<<"final_resiual"<< final_resiual<<endl;
		}
		final_resiual = residual_value;
	}
	cout<<"final example result "<<endl;
	cout<<"new ori example ";
	for (int i = 0; i < m_numExample; i++)
	{
		delata_example[i] = new_oriexample[i] - ori_example[i];
		cout<<new_oriexample[i];
	}
	cout<<endl;

	return true;
}


