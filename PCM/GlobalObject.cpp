#include "global_datas/g_vbo_primitives.hpp"
#include "GlobalObject.h"
#include "qt_gui/main_window.h"
#include "qt_gui/paint_canvas.h"
#include "sample_set.h"

main_window* Global_Window;

PaintCanvas* Global_Canvas;
SampleSet*   Global_SampleSet;
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
