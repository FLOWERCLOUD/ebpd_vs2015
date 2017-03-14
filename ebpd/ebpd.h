#ifndef EBPD_H
#define EBPD_H

#include <QtWidgets/QMainWindow>
#include "ui_ebpd.h"

class ebpd : public QMainWindow
{
	Q_OBJECT

public:
	ebpd(QWidget *parent = 0);
	~ebpd();

private:
	Ui::ebpdClass ui;
};

#endif // EBPD_H
