#pragma once
#include <QVector>
#include <QQuaternion>
class BoundingBox
{
public:
	static float delta;
	QVector3D pMin;
	QVector3D pMax;

	BoundingBox(void);
	BoundingBox(QVector3D ipMin,QVector3D ipMax);
	virtual ~BoundingBox(void);
	//���з�������bbox��ñ�ģ�ʹ�һ��
	void merge(const QVector3D&point,bool isBigger = true);
	void merge(const BoundingBox&iBox);
	//������һ��bbox���ཻ����
	bool intersect(const BoundingBox& iBox, BoundingBox& resultBox);
	//���ģ�͵�ʵ��bbox���꣬����������˵�bbox
	void getTightBound(QVector3D& tightMin, QVector3D& tightMax);
	void draw()const;
	//����һ������
	float halfArea();
	float area();
	//��ʾ����ֵ
	void displayCoords()const;
	//���һ�����ǲ����ڰ�Χ�����棬�����պ��ڰ�Χ�б����ϵĵ�
	bool isInBoxInclusive(const QVector3D& point);
	//���һ�����ǲ����ڰ�Χ�����棬�����պ��ڰ�Χ�б����ϵĵ�
	bool isInBoxExclusive(const QVector3D& point);
	//����һ���Ѿ�ȷ������bbox����ĵ�
	//ȷ�����Ƿ���һ�����ϣ�����ڣ�����һ��������
	//face��ʾ������
	// 0 Ϊx���棬 1 Ϊx����
	// 2 Ϊy���棬 3 Ϊy����
	// 4 Ϊz���棬 5 Ϊz����
	bool onFace(QVector3D& point, int& face);
	//�������Ƿ����Χ���ཻ
	//�ҳ����귶Χ��������,x����Ϊ0��y����Ϊ1��z����Ϊ2
	//tMin,tMaxΪ���߽���Ĳ���ֵ
	int maxAxis();	
	BoundingBox operator=(const BoundingBox&box);
	bool operator== (const BoundingBox& box)const;
private:
	inline void swap(float&t1,float&t2);
};
