#include "parsers/tex_loader.hpp"

#include <QImage>
#include <QGLWidget>
#include <QFileInfo>
#include <iostream>

// =============================================================================
namespace Loader {
// =============================================================================

// =============================================================================
namespace Tex_loader{
// =============================================================================
using namespace Tbx;
GlTex2D* load(const std::string& file_path)
{
    QFileInfo qt_file(QString(file_path.c_str()));
    if( !qt_file.exists() || qt_file.isDir() )
    {
        std::cerr << "WARNING: can't load this texture. The file: ";
        std::cerr << file_path << " does not exists\n";

        return 0;
    }


    QImage img(file_path.c_str());

    if(img.isNull()){
        std::cerr << "WARNING: file type isn't supported for reading. ";
        std::cerr << "Can't load this texture: " << file_path << "\n";
        return 0;
    }

    QImage gl_img = QGLWidget::convertToGLFormat( img );

    GlTex2D* tex = new GlTex2D(gl_img.width(), gl_img.height(), 0,
                               GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT,
                               GL_RGBA);

    GLTextureBinding2DSave save_tex_binding;
    tex->bind();
    tex->allocate(GL_UNSIGNED_BYTE, GL_RGBA, gl_img.bits());

    return tex;
}


}// END TEX_LOADER NAMESPACE ===================================================

}// END LOADER NAMESPACE =======================================================
