#include "VideoEDMesh.h"
#include <QtOpenGL/QGLFunctions>

// �����������õ�����Ĵ�����ʼ���
#define SCENE_TEXTURE_REGISTER_OFFSET 0 
#define MESH_TEXTURE_REGISTER_OFFSET  5  

//qHash������Ϊ��qHash������أ����ܷ���namespace videoEditting ����
static uint qHash(const QVector3D& key)
{
	//float x = key.x();
	//float y = key.y();
	//float z = key.z();
	//float sum = x + y + z;
	//return *((uint*)(&sum));
	return qHash(QString("%1x%2x%3").arg(key.x()).arg(key.y()).arg(key.z()));
}

static uint qHash(const QVector2D& key)
{
	//float x = key.x();
	//float y = key.y();
	//float sum = x + y;
	//return *((uint*)(&sum));
	return qHash( QString("%1x%2x").arg(key.x()).arg(key.y()) );
}

//uint qHash(const QVector3D& v, uint seed = 0)
//{
//	return qHash( QString("%1x%2x%3").arg(v.x()).arg(v.y()).arg(v.z()) );
//}
//uint qHash(const QVector2D& v, uint seed = 0)
//{
//	return qHash( QString("%1x%2x").arg(v.x()).arg(v.y()) );
//}

namespace videoEditting
{
	QSharedPointer<QGLShaderProgram> Mesh::geometryProgram;
	QSharedPointer<QGLShaderProgram> Mesh::appearProgram;

	bool Mesh::isWireFrameEnabled = true;

	Mesh::Mesh()
	{
		type = OBJ_MESH;

		if (!geometryProgram)
		{
			//ʹ��QGLShaderProgram ��Ĭ�Ϲ��캯���Ļ�����ǰshaderprogram ���ڵ�ǰ��QGLContext
			//���ھ��ж��QGLContext ������£��Ͳ�Ӧ����ͬһ��QGLShaderProgram ����
			geometryProgram = QSharedPointer<QGLShaderProgram>(new QGLShaderProgram);
			geometryProgram->addShaderFromSourceFile(QGLShader::Vertex, "./rendering/myshaders/fboShaderVS.glsl");
			geometryProgram->addShaderFromSourceFile(QGLShader::Fragment, "./rendering/myshaders/fboShaderFS.glsl");
			geometryProgram->bindAttributeLocation("vertexV", PROGRAM_VERTEX_ATTRIBUTE);
			geometryProgram->bindAttributeLocation("normalV", PROGRAM_NORMAL_ATTRIBUTE);
			geometryProgram->bindAttributeLocation("texCoordV", PROGRAM_TEXCOORD_ATTRIBUTE);
			geometryProgram->link();
			geometryProgram->bind();
		}
		if (!appearProgram)
		{
			appearProgram = QSharedPointer<QGLShaderProgram>(new QGLShaderProgram);
			appearProgram->addShaderFromSourceFile(QGLShader::Vertex, "./rendering/myshaders/sceneShaderVS.glsl");
			appearProgram->addShaderFromSourceFile(QGLShader::Fragment, "./rendering/myshaders/sceneShaderFS.glsl");
			appearProgram->bindAttributeLocation("vertexV", PROGRAM_VERTEX_ATTRIBUTE);
			appearProgram->bindAttributeLocation("normalV", PROGRAM_NORMAL_ATTRIBUTE);
			appearProgram->bindAttributeLocation("tangentV", PROGRAM_TANGENT_ATTRIBUTE);
			appearProgram->bindAttributeLocation("bitangentV", PROGRAM_BITANGENT_ATTRIBUTE);
			appearProgram->bindAttributeLocation("texCoordV", PROGRAM_TEXCOORD_ATTRIBUTE);
			appearProgram->link();
			appearProgram->bind();
		}
	}


	void Mesh::drawGeometry()
	{
		if (!isObjVisible)
			return;
		QGLFunctions glFuncs(QGLContext::currentContext());
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glMultMatrixf(transform.getTransformMatrix().constData());

		geometryProgram->enableAttributeArray(PROGRAM_VERTEX_ATTRIBUTE);
		geometryProgram->enableAttributeArray(PROGRAM_NORMAL_ATTRIBUTE);
		geometryProgram->enableAttributeArray(PROGRAM_TEXCOORD_ATTRIBUTE);

		glFuncs.glEnableVertexAttribArray(PROGRAM_VERTEX_ATTRIBUTE);
		glFuncs.glEnableVertexAttribArray(PROGRAM_NORMAL_ATTRIBUTE);
		glFuncs.glEnableVertexAttribArray(PROGRAM_TEXCOORD_ATTRIBUTE);

		// ��ʾ��ǰ�����ȵ�����
		int texRegBase = GL_TEXTURE0_ARB + MESH_TEXTURE_REGISTER_OFFSET;
		int texRegOffset = MESH_TEXTURE_REGISTER_OFFSET;
		glActiveTextureARB(texRegBase + 3);								// ����һ������Ĵ���
		glBindTexture(GL_TEXTURE_2D, canvas.getGLBaseThicknessTexObj());		// ����ͼ����󶨵��Ĵ���
		geometryProgram->setUniformValue("baseThickTex", texRegOffset + 3);	// �ѼĴ����󶨵���ɫ������

		bool isBindError = false;																	// �������������
		if (!glBuffer[0].bind())
		{
			std::cout << "drawGeometry() bind 0 error" << std::endl;
			isBindError = true;
		}
		glFuncs.glVertexAttribPointer(PROGRAM_VERTEX_ATTRIBUTE, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		if (!glBuffer[1].bind())
		{
			std::cout << "drawGeometry() bind 1 error" << std::endl;
			isBindError = true;
		}
		glFuncs.glVertexAttribPointer(PROGRAM_NORMAL_ATTRIBUTE, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		if (!glBuffer[4].bind())
		{
			std::cout << "drawGeometry() bind 4 error" << std::endl;
			isBindError = true;
		}
		glFuncs.glVertexAttribPointer(PROGRAM_TEXCOORD_ATTRIBUTE, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		if(!isBindError)
			glDrawArrays(GL_TRIANGLES, 0, glVertexBuffer.size());

		glFuncs.glDisableVertexAttribArray(PROGRAM_VERTEX_ATTRIBUTE);
		glFuncs.glDisableVertexAttribArray(PROGRAM_NORMAL_ATTRIBUTE);
		glFuncs.glDisableVertexAttribArray(PROGRAM_TEXCOORD_ATTRIBUTE);

		glPopMatrix();
	}

	void Mesh::buildGLArrays()
	{
		int nElements = faces.size() * 3;
		glVertexBuffer.reserve(nElements);
		glNormalBuffer.reserve(nElements);
		glTangentBuffer.reserve(nElements);
		glBitangentBuffer.reserve(nElements);
		glTexcoordBuffer.reserve(nElements);

		for (int ithTri = 0; ithTri < faces.size(); ++ithTri)
		{
			ObjTriangle& f = faces[ithTri];

			QVector3D& v0 = vertices[f.vertIndex[0]];
			QVector3D& v1 = vertices[f.vertIndex[1]];
			QVector3D& v2 = vertices[f.vertIndex[2]];
			QVector2D& t0 = texcoords[f.texcoordIndex[0]];
			QVector2D& t1 = texcoords[f.texcoordIndex[1]];
			QVector2D& t2 = texcoords[f.texcoordIndex[2]];

			QVector2D dt1 = t1 - t0;
			QVector2D dt2 = t2 - t0;
			QVector3D dv1 = v1 - v0;
			QVector3D dv2 = v2 - v0;

			qreal det = dt1.x() * dt2.y() - dt2.x() * dt1.y();
			QVector3D tangent = (dt2.y() * dv1 - dt1.y() * dv2) / det;
			QVector3D bitangent = (dt1.x() * dv2 - dt2.x() * dv1) / det;

			for (int ithPoint = 0; ithPoint < 3; ++ithPoint)
			{
				glVertexBuffer.push_back(vertices[f.vertIndex[ithPoint]]);
				glNormalBuffer.push_back(normals[f.norIndex[ithPoint]]);
				glTangentBuffer.push_back(tangent);
				glBitangentBuffer.push_back(bitangent);
				QVector2D& uv = texcoords[f.texcoordIndex[ithPoint]];
				glTexcoordBuffer.push_back(QVector4D(uv.x(), uv.y(), float(objectID), float(ithTri)));
			}
		}
	}

	void Mesh::releaseGLArrays()
	{
		glVertexBuffer.clear();
		glNormalBuffer.clear();
		glTangentBuffer.clear();
		glBitangentBuffer.clear();
		glTexcoordBuffer.clear();
	}
	void assertTrue(bool b)
	{
		if (!b)
		{
			std::cout << "Mesh::init assertTrue error " << std::endl;
		}
		
	}
	void Mesh::init()
	{
		optimizeArrays();
		if (!glVertexBuffer.size())
			buildGLArrays();

		QVector<QVector3D>* buf[] = {
			&glVertexBuffer,
			&glNormalBuffer,
			&glTangentBuffer,
			&glBitangentBuffer
		};

		for (int i = 0; i < NUM_GL_VALUE_BUFFERS - 1; ++i)
		{
			assertTrue(glBuffer[i].create());
			assertTrue(glBuffer[i].bind());
			glBuffer[i].setUsagePattern(QGLBuffer::StaticDraw);
			glBuffer[i].allocate(buf[i]->constData(), sizeof(QVector3D) * buf[i]->size());
		}

		assertTrue(glBuffer[4].create());
		assertTrue(glBuffer[4].bind());
		glBuffer[4].setUsagePattern(QGLBuffer::StaticDraw);
		glBuffer[4].allocate(glTexcoordBuffer.constData(), sizeof(QVector4D) * glTexcoordBuffer.size());

		glBuffer[5] = QGLBuffer(QGLBuffer::IndexBuffer);
		glBuffer[5].setUsagePattern(QGLBuffer::DynamicDraw);
		assertTrue(glBuffer[5].create());
		assertTrue(glBuffer[5].bind());

		buildLocalBBox();
		canvas.init(vertices, normals, texcoords, faces);
	}

	void Mesh::releaseGLBuffer()
	{
		if (vertices.size())
		{
			for (int i = 0; i < NUM_GL_VALUE_BUFFERS; ++i)
			{
				if(glBuffer[i].isCreated())
					glBuffer[i].release();
			}
			releaseGLArrays();
		}
	}

	Mesh::~Mesh()
	{
		qDebug() << "delete mesh " << objectID << endl;
		releaseGLBuffer();
		canvas.release();
	}


	void Mesh::buildLocalBBox()
	{
		if (vertices.size())
		{
			localBBox.pMax = localBBox.pMin = vertices[0];
			for (int i = 1; i < vertices.size(); i++)
			{
				localBBox.merge(vertices[i]);
			}
		}
	}

	void Mesh::drawAppearance()
	{
		QGLFunctions glFuncs(QGLContext::currentContext());
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glMultMatrixf(transform.getTransformMatrix().constData());

		appearProgram->enableAttributeArray(PROGRAM_VERTEX_ATTRIBUTE);
		appearProgram->enableAttributeArray(PROGRAM_NORMAL_ATTRIBUTE);
		appearProgram->enableAttributeArray(PROGRAM_TANGENT_ATTRIBUTE);
		appearProgram->enableAttributeArray(PROGRAM_BITANGENT_ATTRIBUTE);
		appearProgram->enableAttributeArray(PROGRAM_TEXCOORD_ATTRIBUTE);

		glFuncs.glEnableVertexAttribArray(PROGRAM_VERTEX_ATTRIBUTE);
		glFuncs.glEnableVertexAttribArray(PROGRAM_NORMAL_ATTRIBUTE);
		glFuncs.glEnableVertexAttribArray(PROGRAM_BITANGENT_ATTRIBUTE);
		glFuncs.glEnableVertexAttribArray(PROGRAM_TANGENT_ATTRIBUTE);
		glFuncs.glEnableVertexAttribArray(PROGRAM_TEXCOORD_ATTRIBUTE);

		int texRegBase = GL_TEXTURE0_ARB + MESH_TEXTURE_REGISTER_OFFSET;
		int texRegOffset = MESH_TEXTURE_REGISTER_OFFSET;
		glActiveTextureARB(texRegBase + 0);								// ����һ������Ĵ���
		glBindTexture(GL_TEXTURE_2D, canvas.getGLColorTexObj());		// ����ͼ����󶨵��Ĵ���
		appearProgram->setUniformValue("colorTex", texRegOffset + 0);	// �ѼĴ����󶨵���ɫ������

		glActiveTextureARB(texRegBase + 1);
		glBindTexture(GL_TEXTURE_2D, canvas.getGLSurfTexObj());
		appearProgram->setUniformValue("surfTex", texRegOffset + 1);

		glActiveTextureARB(texRegBase + 2);
		glBindTexture(GL_TEXTURE_2D, canvas.getGLThicknessTexObj());
		appearProgram->setUniformValue("thickTex", texRegOffset + 2);

		appearProgram->setUniformValue("finalAlpha", alphaForAppearance);

		bool isBindError = false;
		if (!glBuffer[0].bind())
		{
			std::cout << "drawAppearance() bind 0 error" << std::endl;
			isBindError = true;
		}
		glFuncs.glVertexAttribPointer(PROGRAM_VERTEX_ATTRIBUTE, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		if (!glBuffer[1].bind())
		{
			std::cout << "drawAppearance() bind 1 error" << std::endl;
			isBindError = true;
		}
		glFuncs.glVertexAttribPointer(PROGRAM_NORMAL_ATTRIBUTE, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		if (!glBuffer[2].bind())
		{
			std::cout << "drawAppearance() bind 2 error" << std::endl;
			isBindError = true;
		}
		glFuncs.glVertexAttribPointer(PROGRAM_TANGENT_ATTRIBUTE, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		if(!glBuffer[3].bind())
		{
			std::cout << "drawAppearance() bind 3 error" << std::endl;
			isBindError = true;
		}
		glFuncs.glVertexAttribPointer(PROGRAM_BITANGENT_ATTRIBUTE, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		if (!glBuffer[4].bind())
		{
			std::cout << "drawAppearance() bind 4 error" << std::endl;
			isBindError = true;
		}
		glFuncs.glVertexAttribPointer(PROGRAM_TEXCOORD_ATTRIBUTE, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		if(!isBindError)
			glDrawArrays(GL_TRIANGLES, 0, glVertexBuffer.size());

		glFuncs.glDisableVertexAttribArray(PROGRAM_VERTEX_ATTRIBUTE);
		glFuncs.glDisableVertexAttribArray(PROGRAM_NORMAL_ATTRIBUTE);
		glFuncs.glDisableVertexAttribArray(PROGRAM_BITANGENT_ATTRIBUTE);
		glFuncs.glDisableVertexAttribArray(PROGRAM_TANGENT_ATTRIBUTE);
		glFuncs.glDisableVertexAttribArray(PROGRAM_TEXCOORD_ATTRIBUTE);

		drawSelectedFaces();
		if (isWireFrameEnabled && isObjSelected)
		{
			drawWireFrame();
		}
		glBuffer[0].release();
		glBuffer[1].release();
		glBuffer[2].release();
		glBuffer[3].release();
		glBuffer[4].release();


		glPopMatrix();

	}
	void Mesh::drawSelectedFaces()
	{
		if (!selectedFaceIDSet.size())
			return;

		appearProgram->release();
		QGLFunctions glFuncs(QGLContext::currentContext());
		glLineWidth(3.0f);
		glColorMaterial(GL_FRONT, GL_AMBIENT);    //�ʼ��ɫ���ʶ�Ӧ����ambient�ġ�����Ҫ��Ϊdiffuse
		glColor3f(255.0 / 255.0f, 159.0 / 255.0f, 0.0 / 255.0f);
		glEnable(GL_COLOR_MATERIAL);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		glFuncs.glEnableVertexAttribArray(PROGRAM_VERTEX_ATTRIBUTE);

		bool isBindError = false;
		if (!glBuffer[5].bind())
		{
			std::cout << "drawSelectedFaces bind 5 error" << std::endl;
			isBindError = true;
		}
		if(!isBindError)
			glDrawElements(GL_TRIANGLES, selectedVertexIDArray.size(), GL_UNSIGNED_INT, 0);

		glFuncs.glDisableVertexAttribArray(PROGRAM_VERTEX_ATTRIBUTE);

		glDisable(GL_COLOR_MATERIAL);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		appearProgram->bind();
	}
	void Mesh::updateGLTextures()
	{
		canvas.updateGLTextures();
	}

	bool Mesh::getTriangleData(const int faceID, TriangleData& triData)
	{/*
	 if (faceID < 0 || faceID >= faces.size())
	 return false;*/
		ObjTriangle& f = faces[faceID];

		for (char i = 0; i < 3; ++i)
		{
			triData.vertex[i] = vertices[f.vertIndex[i]];
			triData.normal[i] = normals[f.norIndex[i]];
			triData.texCoord[i] = texcoords[f.texcoordIndex[i]];
		}
		return true;
	}



	void Mesh::drawWireFrame()
	{

		appearProgram->release();
		QGLFunctions glFuncs(QGLContext::currentContext());
		glDisable(GL_LIGHTING);  //��Ҫ�ص�����
		glColor3f(0.8, 0.8, 0.8);
		glLineWidth(1.0f);

		glFuncs.glEnableVertexAttribArray(PROGRAM_VERTEX_ATTRIBUTE);
		glColorMaterial(GL_FRONT, GL_AMBIENT);    //�ʼ��ɫ���ʶ�Ӧ����ambient�ġ�����Ҫ��Ϊdiffuse
		glColor3f(116.0 / 255.0f, 190.0 / 255.0f, 160.0 / 255.0f);
		glEnable(GL_COLOR_MATERIAL);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		//�����Ⱦ������buffer�ȳ��������ֿ��ܳ���
		glDrawArrays(GL_TRIANGLES, 0, glVertexBuffer.size());

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDisable(GL_COLOR_MATERIAL);

		glFuncs.glDisableVertexAttribArray(PROGRAM_VERTEX_ATTRIBUTE);

		appearProgram->bind();
	}






	void Mesh::optimizeArrays()
	{
		// �Ѹ��ֶ������ݼ��뵽һ��hashmap���ϲ���ͬ����
		QMultiHash<QVector3D, int>vertexMap, normalMap;
		QMultiHash<QVector2D, int>texcoordMap;

		for (int ithVert = 0; ithVert < vertices.size(); ++ithVert)
			vertexMap.insertMulti(vertices[ithVert], ithVert);

		for (int ithNor = 0; ithNor < normals.size(); ++ithNor)
			normalMap.insertMulti(normals[ithNor], ithNor);

		for (int ithTex = 0; ithTex < texcoords.size(); ++ithTex)
			texcoordMap.insertMulti(texcoords[ithTex], ithTex);

		QVector<int> vertexIDMap(vertices.size()),
			normalIDMap(normals.size()),
			texcoordIDMap(texcoords.size());

		QVector<QVector3D> newVertexBuf, newNormalBuf;
		QVector<QVector2D> newTexcoordBuf;

		// �����µĶ��㻺��
		QVector3D keyVertex(FLT_MAX, FLT_MAX, FLT_MAX);
		for (QMultiHash<QVector3D, int>::const_iterator iter = vertexMap.constBegin();
			iter != vertexMap.constEnd(); ++iter)
		{
			if (iter.key() != keyVertex)
			{	// �����µĶ��㣬��������µ�����
				keyVertex = iter.key();
				newVertexBuf.push_back(iter.key());
			}
			vertexIDMap[iter.value()] = newVertexBuf.size() - 1;
		}

		// �����µķ��߻���
		QVector3D keyNormal(FLT_MAX, FLT_MAX, FLT_MAX);
		for (QMultiHash<QVector3D, int>::const_iterator iter = normalMap.constBegin();
			iter != normalMap.constEnd(); ++iter)
		{
			if (iter.key() != keyNormal)
			{	// �����µĶ��㣬��������µ�����
				keyNormal = iter.key();
				newNormalBuf.push_back(iter.key());
			}
			normalIDMap[iter.value()] = newNormalBuf.size() - 1;
		}


		// �����µ���ͼ���껺��
		QVector2D keyTexcoord(FLT_MAX, FLT_MAX);
		for (QMultiHash<QVector2D, int>::const_iterator iter = texcoordMap.constBegin();
			iter != texcoordMap.constEnd(); ++iter)
		{
			if (iter.key() != keyTexcoord)
			{	// �����µĶ��㣬��������µ�����
				keyTexcoord = iter.key();
				newTexcoordBuf.push_back(iter.key());
			}
			texcoordIDMap[iter.value()] = newTexcoordBuf.size() - 1;
		}

		// ��������������
		//�������һ����෨��ȿ��ܻ����
		for (int ithTri = 0; ithTri < faces.size(); ++ithTri)
		{
			ObjTriangle& tri = faces[ithTri];
			for (int ithPoint = 0; ithPoint < 3; ++ithPoint)
			{
				tri.vertIndex[ithPoint] = vertexIDMap[tri.vertIndex[ithPoint]];
				tri.norIndex[ithPoint] = normalIDMap[tri.norIndex[ithPoint]];
				tri.texcoordIndex[ithPoint] = texcoordIDMap[tri.texcoordIndex[ithPoint]];
			}
		}
		// ���¶�������
		vertices = newVertexBuf;
		normals = newNormalBuf;
		texcoords = newTexcoordBuf;
	}

	void Mesh::buildSelectVtxIDArray()
	{
		selectedVertexIDArray.clear();
		foreach(const int& faceID, selectedFaceIDSet)
		{
			ObjTriangle& face = faces[faceID];
			selectedVertexIDArray.push_back(faceID * 3);
			selectedVertexIDArray.push_back(faceID * 3 + 1);
			selectedVertexIDArray.push_back(faceID * 3 + 2);
		}
		glBuffer[5].bind();
		glBuffer[5].allocate(selectedVertexIDArray.constData(), sizeof(int) * selectedVertexIDArray.size());
	}

	void Mesh::clearSelectedFaceID()
	{
		selectedFaceIDSet.clear();
		buildSelectVtxIDArray();
	}

	void Mesh::addSelectedFaceID(const QSet<int>& faceIDSet)
	{
		selectedFaceIDSet += faceIDSet;
		buildSelectVtxIDArray();
	}

	void Mesh::removeSelectedFaceID(const QSet<int>& faceIDSet)
	{
		selectedFaceIDSet -= faceIDSet;
		buildSelectVtxIDArray();
	}

	void Mesh::setSelectedFaceID(const QSet<int>& faceIDSet)
	{
		selectedFaceIDSet = faceIDSet;
		buildSelectVtxIDArray();
	}
	QDataStream& operator<<(QDataStream& out, const ObjTriangle&tri)
	{
		out << quint32(tri.vertIndex[0])
			<< quint32(tri.vertIndex[1])
			<< quint32(tri.vertIndex[2])
			<< quint32(tri.texcoordIndex[0])
			<< quint32(tri.texcoordIndex[1])
			<< quint32(tri.texcoordIndex[2])
			<< quint32(tri.norIndex[0])
			<< quint32(tri.norIndex[1])
			<< quint32(tri.norIndex[2])
			<< quint32(tri.mtlIndex);
		return out;
	}

	QDataStream& operator >> (QDataStream& in, ObjTriangle&tri)
	{
		in >> (tri.vertIndex[0])
			>> (tri.vertIndex[1])
			>> (tri.vertIndex[2])
			>> (tri.texcoordIndex[0])
			>> (tri.texcoordIndex[1])
			>> (tri.texcoordIndex[2])
			>> (tri.norIndex[0])
			>> (tri.norIndex[1])
			>> (tri.norIndex[2])
			>> (tri.mtlIndex);
		return in;
	}

	QDataStream& operator<<(QDataStream& out, const Mesh&mesh)
	{
		out << mesh.vertices << mesh.normals << mesh.texcoords << mesh.faces
			<< mesh.canvas;
		return out;
	}

	QDataStream& operator >> (QDataStream& in, Mesh&mesh)
	{
		in >> mesh.vertices >> mesh.normals >> mesh.texcoords >> mesh.faces
			>> mesh.canvas;
		mesh.init();
		return in;
	}
}
