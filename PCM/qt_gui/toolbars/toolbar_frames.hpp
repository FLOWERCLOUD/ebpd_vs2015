#ifndef TOOLBAR_FRAMES_HPP__
#define TOOLBAR_FRAMES_HPP__

#include <QToolBar>
#include <QToolButton>
#include <QPushButton>
#include <QLineEdit>
#include <QSpinBox>
#include <QSlider>
#include <QLayout>
#include <QComboBox>
#include <QTimer>

#include <vector>
#include <string>

namespace Loader
{

	class Base_anim_eval;
}
//#include "loader.hpp"
class Widget_frame;
/**
 * @name Toolbar_frames
 * @brief GUI Handling animation timeline and recording
 *
*/
class Toolbar_frames : public QToolBar {
    Q_OBJECT

public:
    Toolbar_frames(QWidget* parent);
    ~Toolbar_frames();

    /// Update the scene knowing the current animation and frame
    void update_scene();

    /// Sets the combo_box to display the animation list.
    void set_anim_list(const std::vector<Loader::Base_anim_eval*>& list);

    /// Concatenate animation list
    /// @note animation names are renamed if they already exists in the list
    void add_anims(const std::vector<Loader::Base_anim_eval*>& list);

public slots:
    void img_button_toogle(bool t);
    void rec_button_released();
    void path_button_released();
    void set_frame_number(int n);
    void anim_box_index_changed(int idx);

    /// Goto previous frame
    void prev();
    /// Goto next frame
    void next();
    /// play the whole animation
    void play();
    /// stop animation
    void stop();

    /// Toogled when dspinbox of fps changes
    void fps_changed(double fps);
signals:
    /// Signal emited when the frame number changes
    void update_gl();

private:
    /// Toggle play/pause button to pause
    void set_pause();
    /// Toggle play/pause button to play
    void set_play();


    /// File formats to register the mesh position
    enum Anim_t {MDD, PC2};
    void export_anim(Anim_t t);

    Widget_frame* _buttons;     ///< play/stop/etc. buttons
    QIcon         _ico_rec_off; ///< record icon off
    QIcon         _ico_rec_on;  ///< record icon on
    QIcon         _ico_play;    ///< play icon
    QIcon         _ico_pause;   ///< play icon

    QComboBox*      _anim_box;
    QSlider*        _slider;        ///< Slider representing timelines
    QSpinBox*       _spinB_frames;  ///< Frame countdown
    QDoubleSpinBox* _dSpinB_fps;    ///< Frame rate animation
    QToolButton*    _rec_button;    ///< toogle recording
    QToolButton*    _img_button;    ///< toogle screenshots
    QPushButton*    _path_button;
    QLineEdit*      _line_edit;     ///< Holds recorded files path
    QTimer          _timer_frame;   ///< Timer to play the animation

    bool _rec_state;             ///< wether we record the animation or not
    bool _shot_state;            ///< wether screenshots are on or off
    bool _play;                  ///< wether we are in pause or playing

    /// List of animations names for the combo box
    std::vector<std::string> _anim_names;
    /// List of animations evaluators
    std::vector<Loader::Base_anim_eval*> _anim_list;

    int _frame;      ///< number of the current frame
    int _curr_anim;  ///< index of the current animation in '_anim_list'
};




#endif // TOOLBAR_FRAMES_HPP__
