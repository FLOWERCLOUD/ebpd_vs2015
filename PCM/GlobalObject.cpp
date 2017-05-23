#include "global_datas/g_vbo_primitives.hpp"
#include "GlobalObject.h"
#include "qt_gui/main_window.h"
#include "qt_gui/paint_canvas.h"
#include "VideoEdittingParameter.h"
#include "VideoEditingWindow.h"
#include "sample_set.h"
#include <QSemaphore>
#include <QMutex>

main_window* Global_Window;

PaintCanvas* Global_Canvas;
SampleSet*   Global_SampleSet;
VideoEditingWindow* Global_WideoEditing_Window;
void globalObjectsInit()
{

	Global_SampleSet = new SampleSet();
}
void globalObjectsDelete()
{
	if(Global_SampleSet)delete Global_SampleSet;
	
}
void setGlobalWindow(main_window* _window)
{
	Global_Window = _window;
}

void setGlobalCanvas(PaintCanvas* _canvas)
{
	Global_Canvas = _canvas;
}

void setGlobalSampleSet(SampleSet* _sampleSet)
{
	Global_SampleSet = _sampleSet;
}
void setGlobalWideoEditingWindow(VideoEditingWindow* _videoEditingWindow)
{
	Global_WideoEditing_Window = _videoEditingWindow;
}
QSemaphore GlobalData::trimapSemaphore(1);
QSemaphore GlobalData::mattingSemaphore(0);
QMutex     GlobalData::mutex;