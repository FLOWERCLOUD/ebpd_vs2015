#pragma once

#include <QTextCodec>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QTime>
#include <QSqlError>
#include <QtDebug>
#include <QSqlDriver>
#include <QSqlRecord>

class ViewDataBase
{
public:
	ViewDataBase();
	~ViewDataBase();
	bool createConnection();  //����һ������
	bool createTable();       //�������ݿ��
	bool insert();            //��������
	bool queryAll();          //��ѯ������Ϣ
	bool updateById(int id);  //����
	bool deleteById(int id);  //ɾ��
	bool sortById();          //����
};