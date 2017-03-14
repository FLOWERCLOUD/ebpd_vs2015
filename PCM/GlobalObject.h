#pragma  once

class main_window;
class PaintCanvas;
class SampleSet;
extern main_window* Global_Window;
extern PaintCanvas* Global_Canvas;
extern SampleSet*   Global_SampleSet;
void globalObjectsInit();
void globalObjectsDelete();
void setGlobalWindow( main_window* _window);
void setGlobalCanvas( PaintCanvas* _canvas);
void setGlobalSampleSet(SampleSet* _sampleSet);