#include <QDialog>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>

class Diag_ok_cancel : public QDialog {
    Q_OBJECT

public:
    Diag_ok_cancel( const QString& name, const QString& txt, QWidget *parent) :
        QDialog( parent )
    {
        this->setWindowTitle(name);

        // Setup layouts
        QVBoxLayout* vlayout = new QVBoxLayout(this);
        QWidget* widget      = new QWidget(this);
        QHBoxLayout* hlayout = new QHBoxLayout(widget);

        // Add label
        QLabel* lbl;
        lbl = new QLabel(txt, this);
        vlayout->addWidget(lbl);
        vlayout->addWidget(widget);

        // Add buttons
        QPushButton *ok, *cancel;
        ok = new QPushButton( "OK", this );
        ok->setGeometry( 10,10, 100,30 );
        connect( ok, SIGNAL(clicked()), SLOT(accept()) );
        cancel = new QPushButton( "Cancel", this );
        cancel->setGeometry( 10,60, 100,30 );
        connect( cancel, SIGNAL(clicked()), SLOT(reject()) );

        // Add widgets to layout
        hlayout->addWidget(ok);
        hlayout->addWidget(cancel);
    }
};
