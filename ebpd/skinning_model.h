#pragma once
#include "basic_types.h"
#include "print_util.h"
#include "quaternion.h"
#include "solver.h"



 /*
   xi =  WibTbui
       [1 X B]         [B X 3]
   [ w1,w2,....wB] *[ x1, y1 ,z1]          [T1 ,T2, T3,....TB]     [x1]
                    [ x2, y2 ,z2]      <=      [4 X 4B]           *[y1]  
                      ..........                                   [z1]  [4B X 1]
					[xB , yB ,zB]                                  [1]
					                                               [x1]
					                                               [y1]
																   [z1]															   																   
																 ...
[T1 ]               [x1 x2 x3 ....xn]       [x11 x21 x31 ,... xn1]
[T2 ]     *         [y1 y2 y3 ....yn]    =  [y11 y21 y31 ,... yn1]
[T3 ]               [z1 z2 z3 ....zn]       [z11 z21 z31 .... zn1]   [4 B* N]
[...]               [1  1  1  .... 1]       [ 1   1   1  ....  1 ]
[TB ]                                       [x1B ,x2B , .....xnB ]
                                            [y1B ,y2B , .....ynB ]
											[z1B ,z2B,  .....znb ]
											[ 1   1   1  ....  1 ]

[w11 w12 .. w1B  ,         0    ,0 .... 0            ]          [x11  y11 z11  ]      [X1  Y1  Z1]
[        0   ,    w21 w22  w2B                       ]     *    [x12  y12 z12  ]      [X2  Y2  Z2]
[...                                                            [x1B  y1B z1B  ]      [...       ]
[                                       wn1 wn2 ..wnB]          [x21 y21 z21   ]
                                                                [x22 y22 z22   ]   =  [Xn  Yn ,Zn]
																[x2B y2B z2B   ] 
																[ ...          ]
																[xn1 yn1 zn1   ]
																[xn2 yn2 zn2   ]
																[xnB ynB znB   ]

 */                                                               
class Skinning
{

public:
	//默认为列向量
	MatrixXX caculateTraditionalSkinng( MatrixXX& weight , MatrixXX& tf ,MatrixXX& original,int num_vertex , int num_bone)
	{
		Logger<<"weight"<<endl;
		printMatrix( weight);
		Logger<<"tf"<<endl;
		printMatrix( tf);
		Logger<<"original"<<endl;
		printMatrix( original);

//		weight.resize( num_vertex,num_bone);
//		tf.resize(4*num_bone,4);
//		original.resize( num_vertex,3);
		MatrixXX oriVerts;
		oriVerts.resize( 4 ,num_vertex);
		for( int i = 0 ;i <num_vertex; ++i)
	   {
			oriVerts( 0,i) = original(i,0);
			oriVerts( 1,i) = original(i,1);
			oriVerts( 2,i) = original(i,2);
			oriVerts( 3,i) = 1;
		}
		Logger<<"oriVerts"<<endl;
		printMatrix( oriVerts);

		MatrixXX oriTf = tf;
		Logger<<"oriTf"<<endl;
		printMatrix( oriTf);


		MatrixXX  temp_vtx = oriTf*oriVerts;
		Logger<<" temp_vtx"<<endl;
		printMatrix( temp_vtx);

		MatrixXX  temp_vtx2;
		temp_vtx2.resize(num_bone*num_vertex,3);
		for( int i = 0 ; i < num_vertex ; ++i)
		{
			for( int j = 0 ; j < num_bone ; ++j)
			{
				temp_vtx2(i*num_bone+j,0) = temp_vtx(j*4 ,i);
				temp_vtx2(i*num_bone+j,1) = temp_vtx(j*4+1 ,i);
				temp_vtx2(i*num_bone+j,2) = temp_vtx(j*4+2 ,i);
			}
			
		}
		Logger<<" temp_vtx2"<<endl;
		printMatrix( temp_vtx2);
		MatrixXX new_weight;
		new_weight.resize(num_vertex ,num_vertex *num_bone);
		new_weight.setZero();
		for( int i= 0 ; i < num_vertex ;++i)
		{
			for( int j= 0 ; j < num_bone ;++j)
			{
				new_weight( i ,i*num_bone+j) =  weight(i ,j);
			}
		}
		Logger<<" new weight"<<endl;
		printMatrix( new_weight);
		MatrixXX new_vtxs = new_weight* temp_vtx2;
		Logger<<"new_vtxs"<<endl;
		printMatrix( new_vtxs);
		return new_vtxs;
	}
/*

T  [7B X E]
[ q11 q12 ...q1E ]
[ p11 p12 ...q1E ]
[ q21 q22 ...q2E ] 
[ p21 p22 ...q2E ]
          ....
[ qB1 qB2 ...qBE ]
[ pB1 pB2 ...qBE ]

 [e11 e12 e13   e1E]      [p11 p21 ... pB1]
 [e21 e22 e123  e2E]    * [p12 p22 ... pB2]  
 [....             ]      [...............]
 [eN1 eN2 eN3   eNE]      [p1E p2E ... pBE]

 QLERP( Ei ,Tb)
 再转成矩阵形式


	[T11 T12 ... T1n]               [x1 0   0  ....0 ]       [x11 x21 x31 ,... xn1]
	[T21 T22 ... T2n]     *         [y1 0   0  ....0 ]    =  [y11 y21 y31 ,... yn1]
	[T31 T22 ... T2n]               [z1 0   0  ....0 ]       [z11 z21 z31 .... zn1]   [4 B* N]
	[...]                           [1  0   0  ....0 ]       [ 1   1   1  ....  1 ]
	[TB1 TB2 ... TBn]               [0  x2  0        ]       [x1B ,x2B , .....xnB ]
	                                [0  y2  0                [y1B ,y2B , .....ynB ]
	                                [0  z2  0                [z1B ,z2B,  .....znb ]
	                                [  .....                 [ 1   1   1  ....  1 ]
									[             xn ]
									[             yn ]
									[             zn ]
*/
	MatrixXX caculateQLEPSkinng( MatrixXX& weight , MatrixXX& tf ,MatrixXX& original ,MatrixXX& eweight ,int num_vertex , int num_bone)
	{
		Logger<<"weight"<<endl;
		printMatrix( weight);
		Logger<<"tf"<<endl;
		printMatrix( tf);
		Logger<<"original"<<endl;
		printMatrix( original);
		Logger<<"eweight"<<endl;
		printMatrix( eweight);

		//		weight.resize( num_vertex,num_bone);
		//		tf.resize(4*num_bone,4);
		//		original.resize( num_vertex,3);
		MatrixXX oriVerts;
		oriVerts.resize( 4 * num_vertex,num_vertex);
		oriVerts.setZero();
		for( int i = 0 ;i <num_vertex; ++i)
		{
			oriVerts( 4*i+0,i) = original(i,0);
			oriVerts( 4*i+1,i) = original(i,1);
			oriVerts( 4*i+2,i) = original(i,2);
			oriVerts( 4*i+3,i) = 1;
		}
		Logger<<"oriVerts"<<endl;
		printMatrix( oriVerts);

//tf 为7B X E 的矩阵

		//提取旋转的部分
		MatrixXX qm;
		int num_exam = eweight.cols();
		qm.resize( 4*num_bone ,num_exam); //4B X E
		for( int i  = 0 ; i < num_bone ; ++i)
		{
			for( int j = 0; j < num_exam ; ++j )
			{
				qm(4*i,j) = tf( 7* i ,j);
				qm(4*i+1,j) = tf( 7* i+1 ,j);
				qm(4*i+2,j) = tf( 7* i+2 ,j);
				qm(4*i+3,j) = tf( 7* i+3 ,j);
			}
		}
		Logger<<"qm"<<endl;
		printMatrix( qm);
		//3b*3N
		MatrixXX rom = qlerp(qm,eweight);

		//提取平移部分
		MatrixXX tm;  //3B XE
		tm.resize( 3*num_bone , num_exam); //我把平移矩阵依然放在仿射矩阵内，只是旋转部分要设为0
		tm.setZero();
/*
		[1 0 0 Tx]
		[0 1 0 Ty]
		[0 0 1 Ty]
		[0 0 0 1]
*/
		for( int i  = 0 ; i < num_bone ; ++i)
		{

			for( int j = 0; j < num_exam ; ++j )
			{
				tm(3*i,j) = tf( 7* i+4 ,j);
				tm(3*i+1,j) = tf( 7* i+5 ,j);
				tm(3*i+2,j) = tf( 7* i+6 ,j);
			}
		}
		Logger<<"tm"<<endl;
		printMatrix( tm);
		//nxE  * EX3B = NX3B
		MatrixXX traM = eweight * tm.transpose();
		Logger<<"traM"<<endl;
		printMatrix( traM);

		traM.transposeInPlace();  //3BXN

		Logger<<"traM"<<endl;
		printMatrix( traM);
		MatrixXX mergeM ;
		mergeM.resize( 4*num_bone ,4 *num_vertex); //4B x4N
		mergeM.setZero();
		for( int i = 0 ; i< num_bone ;++i)
		{
			for( int j = 0 ; j< num_vertex; ++j)
			{
				mergeM( 4*i ,4*j+3) = traM(3*i ,j);
				mergeM( 4*i+1 ,4*j+3)= traM(3*i+1 ,j);
				mergeM( 4*i+2 ,4*j+3)= traM(3*i+2 ,j);
				mergeM( 4*i+3 ,4*j+3)= 1;
			}
			for( int j = 0 ; j< num_vertex; ++j)
			{
				mergeM( 4*i ,4*j) = rom(3*i ,3*j);
				mergeM( 4*i ,4*j+1) = rom(3*i ,3*j+1);
				mergeM( 4*i ,4*j+2) = rom(3*i ,3*j+2);
				mergeM( 4*i+1 ,4*j) = rom(3*i+1 ,3*j);
				mergeM( 4*i+1 ,4*j+1) = rom(3*i+1 ,3*j+1);
				mergeM( 4*i+1 ,4*j+2) = rom(3*i+1 ,3*j+2);
				mergeM( 4*i+2 ,4*j) =  rom(3*i+2 ,3*j);
				mergeM( 4*i+2 ,4*j+1) = rom(3*i+2 ,3*j+1);
				mergeM( 4*i+2 ,4*j+2) = rom(3*i+2 ,3*j+2);
			}
		}

		Logger<<"mergeM"<<endl;
		printMatrix( mergeM);


		//MatrixXX  temp_vtx = oriTf*oriVerts;
		MatrixXX  temp_vtx =  mergeM*oriVerts;
		Logger<<" temp_vtx"<<endl;
		printMatrix( temp_vtx);

		MatrixXX  temp_vtx2;
		temp_vtx2.resize(num_bone*num_vertex,3);
		for( int i = 0 ; i < num_vertex ; ++i)
		{
			for( int j = 0 ; j < num_bone ; ++j)
			{
				temp_vtx2(i*num_bone+j,0) = temp_vtx(j*4 ,i);
				temp_vtx2(i*num_bone+j,1) = temp_vtx(j*4+1 ,i);
				temp_vtx2(i*num_bone+j,2) = temp_vtx(j*4+2 ,i);
			}

		}
		Logger<<" temp_vtx2"<<endl;
		printMatrix( temp_vtx2);
		MatrixXX new_weight;
		new_weight.resize(num_vertex ,num_vertex *num_bone);
		new_weight.setZero();
		for( int i= 0 ; i < num_vertex ;++i)
		{
			for( int j= 0 ; j < num_bone ;++j)
			{
				new_weight( i ,i*num_bone+j) =  weight(i ,j);
			}
		}
		Logger<<" new weight"<<endl;
		printMatrix( new_weight);
		MatrixXX new_vtxs = new_weight* temp_vtx2;
		Logger<<"new_vtxs"<<endl;
		printMatrix( new_vtxs);
		return new_vtxs;
	}
	//4B X E * NXE' = 4BXN
	//3B X3N
	MatrixXX qlerp( MatrixXX& tbe , MatrixXX& Eie)
	{
		MatrixXX result ;
		
		int num_bone = tbe.rows()/4;
		int num_sample = Eie.cols();
		int num_vtx = Eie.rows();
		result.resize( 4*num_bone,num_vtx);
		MatrixXX q1; q1.resize(4,1);q1.setZero();
		double a[4];
		for( int i = 0 ; i < num_bone ; ++i)
		{
			for( int j = 0 ; j < num_vtx; ++j)
			{
				for( int k = 0 ; k < num_sample;++k )
				{
					a[0] += tbe( 4*i,k)* Eie( j ,k);
					a[1] += tbe( 4*i+1,k)* Eie( j ,k);
					a[2] += tbe( 4*i+1,k)* Eie( j ,k);
					a[3] += tbe( 4*i+1,k)* Eie( j ,k);
				}
				result(4*i,j) = a[0]/ sqrt(a[0]*a[0]);
				result(4*i+1,j) = a[1]/ sqrt(a[1]*a[1]);
				result(4*i+2,j) = a[2]/ sqrt(a[10]*a[1]);
				result(4*i+3,j) = a[3]/ sqrt(a[1]*a[1]);
				a[0] = a[1] = a[2] = a[3] = 0;
			}
		}
		Logger<<"result"<<endl;
		printMatrix( result);
		MatrixXX resultM;
		resultM.resize( 3*num_bone , 3*num_vtx);
		for( int i = 0 ; i < num_bone ; ++i )
		{
			for( int j = 0 ; j < num_vtx; ++j)
			{
				tdviewer::Quaternion q( result( 4* i, j) ,result( 4* i+1, j),result( 4* i+2, j),result( 4* i+3, j));
				double RM[3][3];
				q.getRotationMatrix(RM);
				for( int i1 = 0 ; i1 < 3 ;++i1)
				{
					for( int i2 = 0; i2 < 3 ;++i2)
					{
						resultM( 3*i+i1 ,3*j+i2) = RM[i1][i2];
					}
				}

			}
		}
		Logger<<"resultM"<<endl;
		printMatrix( resultM);
		return resultM;
	}
	MatrixXXF qlerpF( MatrixXXF& tbe , MatrixXXF& Eie)
	{
		MatrixXXF result ;

		int num_bone = tbe.rows()/4;
		int num_sample = Eie.cols();
		int num_vtx = Eie.rows();
		result.resize( 4*num_bone,num_vtx);
		MatrixXXF q1; q1.resize(4,1);q1.setZero();
		fadbad::F<double> a[4];
		for( int i = 0 ; i < num_bone ; ++i)
		{
			for( int j = 0 ; j < num_vtx; ++j)
			{
				for( int k = 0 ; k < num_sample;++k )
				{
					a[0] += tbe( 4*i,k)* Eie( j ,k);
					a[1] += tbe( 4*i+1,k)* Eie( j ,k);
					a[2] += tbe( 4*i+1,k)* Eie( j ,k);
					a[3] += tbe( 4*i+1,k)* Eie( j ,k);
				}
				result(4*i,j) = a[0]/ sqrt(a[0]*a[0]);
				result(4*i+1,j) = a[1]/ sqrt(a[1]*a[1]);
				result(4*i+2,j) = a[2]/ sqrt(a[10]*a[1]);
				result(4*i+3,j) = a[3]/ sqrt(a[1]*a[1]);
				a[0] = a[1] = a[2] = a[3] = 0;
			}
		}
		Logger<<"result"<<endl;
		printMatrix( result);
		MatrixXXF resultM;
		resultM.resize( 3*num_bone , 3*num_vtx);
		for( int i = 0 ; i < num_bone ; ++i )
		{
			for( int j = 0 ; j < num_vtx; ++j)
			{
				tdviewer::Quaternion q( result( 4* i, j).val() ,result( 4* i+1, j).val(),result( 4* i+2, j).val(),result( 4* i+3, j).val());
				double RM[3][3];
				q.getRotationMatrix(RM);
				for( int i1 = 0 ; i1 < 3 ;++i1)
				{
					for( int i2 = 0; i2 < 3 ;++i2)
					{
						resultM( 3*i+i1 ,3*j+i2) = RM[i1][i2];
					}
				}

			}
		}
		Logger<<"resultM"<<endl;
		printMatrix( resultM);
		return resultM;
	}
 

	MatrixXXF func(const MatrixXXF& weight , const MatrixXXF& tf ,const MatrixXXF& original ,MatrixXXF& eweight ,int num_vertex , int num_bone)
	{
		Logger<<"weight"<<endl;
		printMatrix( weight);
		Logger<<"tf"<<endl;
		printMatrix( tf);
		Logger<<"original"<<endl;
		printMatrix( original);
		Logger<<"eweight"<<endl;
		printMatrix( eweight);

		//		weight.resize( num_vertex,num_bone);
		//		tf.resize(4*num_bone,4);
		//		original.resize( num_vertex,3);
		MatrixXXF oriVerts;
		oriVerts.resize( 4 * num_vertex,num_vertex);
		oriVerts.setZero();
		for( int i = 0 ;i <num_vertex; ++i)
		{
			oriVerts( 4*i+0,i) = original(i,0);
			oriVerts( 4*i+1,i) = original(i,1);
			oriVerts( 4*i+2,i) = original(i,2);
			oriVerts( 4*i+3,i) = 1;
		}
		Logger<<"oriVerts"<<endl;
		printMatrix( oriVerts);

//tf 为7B X E 的矩阵

		//提取旋转的部分
		MatrixXXF qm;
		int num_exam = eweight.cols();
		qm.resize( 4*num_bone ,num_exam); //4B X E
		for( int i  = 0 ; i < num_bone ; ++i)
		{
			for( int j = 0; j < num_exam ; ++j )
			{
				qm(4*i,j) = tf( 7* i ,j);
				qm(4*i+1,j) = tf( 7* i+1 ,j);
				qm(4*i+2,j) = tf( 7* i+2 ,j);
				qm(4*i+3,j) = tf( 7* i+3 ,j);
			}
		}
		Logger<<"qm"<<endl;
		printMatrix( qm);
		//3b*3N
		MatrixXXF rom = qlerpF(qm,eweight);

		//提取平移部分
		MatrixXXF tm;  //3B XE
		tm.resize( 3*num_bone , num_exam); //我把平移矩阵依然放在仿射矩阵内，只是旋转部分要设为0
		tm.setZero();
/*
		[1 0 0 Tx]
		[0 1 0 Ty]
		[0 0 1 Ty]
		[0 0 0 1]
*/
		for( int i  = 0 ; i < num_bone ; ++i)
		{

			for( int j = 0; j < num_exam ; ++j )
			{
				tm(3*i,j) = tf( 7* i+4 ,j);
				tm(3*i+1,j) = tf( 7* i+5 ,j);
				tm(3*i+2,j) = tf( 7* i+6 ,j);
			}
		}
		Logger<<"tm"<<endl;
		printMatrix( tm);
		//nxE  * EX3B = NX3B
		MatrixXXF traM = eweight * tm.transpose();
		Logger<<"traM"<<endl;
		printMatrix( traM);

		traM.transposeInPlace();  //3BXN

		Logger<<"traM"<<endl;
		printMatrix( traM);
		MatrixXXF mergeM ;
		mergeM.resize( 4*num_bone ,4 *num_vertex); //4B x4N
		mergeM.setZero();
		for( int i = 0 ; i< num_bone ;++i)
		{
			for( int j = 0 ; j< num_vertex; ++j)
			{
				mergeM( 4*i ,4*j+3) = traM(3*i ,j);
				mergeM( 4*i+1 ,4*j+3)= traM(3*i+1 ,j);
				mergeM( 4*i+2 ,4*j+3)= traM(3*i+2 ,j);
				mergeM( 4*i+3 ,4*j+3)= 1;
			}
			for( int j = 0 ; j< num_vertex; ++j)
			{
				mergeM( 4*i ,4*j) = rom(3*i ,3*j);
				mergeM( 4*i ,4*j+1) = rom(3*i ,3*j+1);
				mergeM( 4*i ,4*j+2) = rom(3*i ,3*j+2);
				mergeM( 4*i+1 ,4*j) = rom(3*i+1 ,3*j);
				mergeM( 4*i+1 ,4*j+1) = rom(3*i+1 ,3*j+1);
				mergeM( 4*i+1 ,4*j+2) = rom(3*i+1 ,3*j+2);
				mergeM( 4*i+2 ,4*j) =  rom(3*i+2 ,3*j);
				mergeM( 4*i+2 ,4*j+1) = rom(3*i+2 ,3*j+1);
				mergeM( 4*i+2 ,4*j+2) = rom(3*i+2 ,3*j+2);
			}
		}

		Logger<<"mergeM"<<endl;
		printMatrix( mergeM);


		//MatrixXX  temp_vtx = oriTf*oriVerts;
		MatrixXXF  temp_vtx =  mergeM*oriVerts;
		Logger<<" temp_vtx"<<endl;
		printMatrix( temp_vtx);

		MatrixXXF  temp_vtx2;
		temp_vtx2.resize(num_bone*num_vertex,3);
		for( int i = 0 ; i < num_vertex ; ++i)
		{
			for( int j = 0 ; j < num_bone ; ++j)
			{
				temp_vtx2(i*num_bone+j,0) = temp_vtx(j*4 ,i);
				temp_vtx2(i*num_bone+j,1) = temp_vtx(j*4+1 ,i);
				temp_vtx2(i*num_bone+j,2) = temp_vtx(j*4+2 ,i);
			}

		}
		Logger<<" temp_vtx2"<<endl;
		printMatrix( temp_vtx2);
		MatrixXXF new_weight;
		new_weight.resize(num_vertex ,num_vertex *num_bone);
		new_weight.setZero();
		for( int i= 0 ; i < num_vertex ;++i)
		{
			for( int j= 0 ; j < num_bone ;++j)
			{
				new_weight( i ,i*num_bone+j) =  weight(i ,j);
			}
		}
		Logger<<" new weight"<<endl;
		printMatrix( new_weight);
		MatrixXXF new_vtxs = new_weight* temp_vtx2;
		Logger<<"new_vtxs"<<endl;
		printMatrix( new_vtxs);
		return new_vtxs;
	}

	void getJacobian(MatrixXX& out_, const MatrixXXF& weight , const MatrixXXF& tf ,const MatrixXXF& original ,MatrixXXF& eweight,
		// std::function< MatrixXXF (const MatrixXXF&  , const MatrixXXF&  ,const MatrixXXF&  ,MatrixXXF&  ,int  , int )> func1 ,
		int num_vertex ,int num_bone,int num_example)
	{
		 std::function< fadbad::F<double> (const fadbad::F<double>& x, const fadbad::F<double>& y, const fadbad::F<double>& z)> funcdd;
		// std::function< MatrixXXF (MatrixXXF&  ,MatrixXXF& ,MatrixXXF& ,MatrixXXF&,MatrixXXF& ,MatrixXXF&)> fd;
		//std::function< MatrixXX (const MatrixXX&  , const MatrixXX&  ,const MatrixXX&  ,MatrixXX&  ,int  , int )> func1;
		const int num_variable = num_vertex * num_example;
		for( int i = 0 ; i< num_vertex ;++i)
		{
			for( int j = 0 ; i< num_example ;++i)
			{
				eweight(i,j).diff(i*num_vertex+j , num_variable);
			}
		}
		MatrixXXF result = func(weight , tf ,original ,eweight ,num_vertex , num_bone);
		MatrixXX jacobians;
		jacobians.resize( num_vertex*3 , num_vertex* num_example);
		for( int i = 0 ; i< num_vertex ;++i)
		{
			//x
			for( int k = 0 ; k< num_vertex ;++k)   
			{
				for( int j = 0 ; j< num_example ;++j)
				{
					jacobians( 3*i ,k*num_vertex+ j) = result(i,0).d(k*num_vertex+j);
				}
			}
			//y
			for( int k = 0 ; k< num_vertex ;++k)
			{
				for( int j = 0 ; j< num_example ;++j)
				{
					jacobians( 3*i+1 ,k*num_vertex+ j) = result(i,1).d(k*num_vertex+j);
				}
			}
			//z
			for( int k = 0 ; k< num_vertex ;++k)
			{
				for( int j = 0 ; j< num_example ;++j)
				{
					jacobians( 3*i+2 ,k*num_vertex+ j) = result(i,2).d(k*num_vertex+j);
				}
			}

		}
		out_ = jacobians;

	}
/*
ji at vertex i is mapped to a change in the barycentric coordinates in row i of the matrix, E. To ensure smooth deformations, we propagate
this change to nearby vertices using a smoothing kernel
*/
	void impulse2exampleWeight(MatrixXX& fi ,MatrixXX& delta_xi, double delta_time, double mass ,  double belta)
	{
		int num_vertex  = fi.rows();
		for( int i = 0 ; i < num_vertex ; ++i)
		{
			double fab_f = fabs(fi(i,0));
			if( fab_f > belta)
			{
				delta_xi(i,0) = delta_time/mass *( fab_f - belta ) * fi(i,0)/fab_f;
			}else
			{
				delta_xi(i,0) = 0.0;
			}
		}
	}
	void solveForTheBestWeight( MatrixXX& out_, const MatrixXXF& weight , const MatrixXXF& tf ,const MatrixXXF& original ,MatrixXXF& eweight ,int num_vertex ,int num_bone,int num_example)
	{
		MatrixXX delta_xi;
		MatrixXX jacobians;
		MatrixXX delta_ei;
		delta_ei.resize( num_vertex , num_example);
		jacobians.resize( num_vertex*3 , num_vertex* num_example);
		 auto func1 = &Skinning::func;
		getJacobian( jacobians ,weight ,tf , original,eweight , num_vertex ,num_bone ,num_example);

		Matrix3X jim; //3 X E
		Matrix3X xi;  
		jim.resize( 3 , num_example);
		xi.resize( 3 ,1);
		for( int i = 0 ; i < num_vertex ; ++i)
		{

			for( int i1= 0 ; i1 < 3 ; ++ i1)
			{
				for( int i2= 0 ; i2 < num_example ; ++ i2)
				{
					jim( i1 , i2) = jacobians( i*3+ i1 , i2);
				}
			}
			//3 X 1
			xi(0,0) = delta_xi( i ,0);
			xi(1,0) = delta_xi( i ,1);
			xi(2,0) = delta_xi( i ,2);

			//E X 1
			MatrixXX eM = jim.transpose() * xi;
			for( int j = 0 ; j < num_example ; ++j)
			{
				delta_ei( i , j) = eM(j ,0);
			}

		}
	}



private:
	


};