#include "Viewdatabase.h"


ViewDataBase::ViewDataBase()
{

}
ViewDataBase::~ViewDataBase()
{


}



//����һ�����ݿ�����
bool ViewDataBase::createConnection()
{
	//�Ժ�Ϳ�����"sqlite1"�����ݿ����������
	QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", "sqlite1");
	db.setDatabaseName(".//qtDb.db");
	if (!db.open())
	{
		qDebug() << "�޷��������ݿ�����";
		return false;
	}
	return true;
}

//�������ݿ��
bool ViewDataBase::createTable()
{
	QSqlDatabase db = QSqlDatabase::database("sqlite1"); //�������ݿ�����
	QSqlQuery query(db);
	bool success = query.exec("create table automobil(id int primary key,attribute varchar,"
		"type varchar,kind varchar,nation int,carnumber int,elevaltor int,"
		"distance int,oil int,temperature int)");
	if (success)
	{
		qDebug() << QObject::tr("���ݿ�����ɹ���\n");
		return true;
	}
	else
	{
		qDebug() << QObject::tr("���ݿ����ʧ�ܣ�\n");
		return false;
	}
}

//�����ݿ��в����¼
bool ViewDataBase::insert()
{
	QSqlDatabase db = QSqlDatabase::database("sqlite1"); //�������ݿ�����
	QSqlQuery query(db);
	query.prepare("insert into automobil values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");

	long records = 10;
	for (int i = 0; i < records; i++)
	{
		query.bindValue(0, i);
		query.bindValue(1, "����");
		query.bindValue(2, "�γ�");
		query.bindValue(3, "����");
		query.bindValue(4, rand() % 100);
		query.bindValue(5, rand() % 10000);
		query.bindValue(6, rand() % 300);
		query.bindValue(7, rand() % 200000);
		query.bindValue(8, rand() % 52);
		query.bindValue(9, rand() % 100);

		bool success = query.exec();
		if (!success)
		{
			QSqlError lastError = query.lastError();
			qDebug() << lastError.driverText() << QString(QObject::tr("����ʧ��"));
			return false;
		}
	}
	return true;
}

//��ѯ������Ϣ
bool ViewDataBase::queryAll()
{
	QSqlDatabase db = QSqlDatabase::database("sqlite1"); //�������ݿ�����
	QSqlQuery query(db);
	query.exec("select * from automobil");
	QSqlRecord rec = query.record();
	qDebug() << QObject::tr("automobil���ֶ�����") << rec.count();

	while (query.next())
	{
		for (int index = 0; index < 10; index++)
			qDebug() << query.value(index) << " ";
		qDebug() << "\n";
	}
	return true;
}

//����IDɾ����¼
bool ViewDataBase::deleteById(int id)
{
	QSqlDatabase db = QSqlDatabase::database("sqlite1"); //�������ݿ�����
	QSqlQuery query(db);
	query.prepare(QString("delete from automobil where id = %1").arg(id));
	if (!query.exec())
	{
		qDebug() << "ɾ����¼ʧ�ܣ�";
		return false;
	}
	return true;
}

//����ID���¼�¼
bool ViewDataBase::updateById(int id)
{
	QSqlDatabase db = QSqlDatabase::database("sqlite1"); //�������ݿ�����
	QSqlQuery query(db);
	query.prepare(QString("update automobil set attribute=?,type=?,"
		"kind=?, nation=?,"
		"carnumber=?, elevaltor=?,"
		"distance=?, oil=?,"
		"temperature=? where id=%1").arg(id));

	query.bindValue(0, "����");
	query.bindValue(1, "�γ�");
	query.bindValue(2, "����");
	query.bindValue(3, rand() % 100);
	query.bindValue(4, rand() % 10000);
	query.bindValue(5, rand() % 300);
	query.bindValue(6, rand() % 200000);
	query.bindValue(7, rand() % 52);
	query.bindValue(8, rand() % 100);

	bool success = query.exec();
	if (!success)
	{
		QSqlError lastError = query.lastError();
		qDebug() << lastError.driverText() << QString(QObject::tr("����ʧ��"));
	}
	return true;
}

//����
bool ViewDataBase::sortById()
{
	QSqlDatabase db = QSqlDatabase::database("sqlite1"); //�������ݿ�����
	QSqlQuery query(db);
	bool success = query.exec("select * from automobil order by id desc");
	if (success)
	{
		qDebug() << QObject::tr("����ɹ�");
		return true;
	}
	else
	{
		qDebug() << QObject::tr("����ʧ�ܣ�");
		return false;
	}
}