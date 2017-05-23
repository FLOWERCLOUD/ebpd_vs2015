#pragma  once

class main_window;
class PaintCanvas;
class SampleSet;
class VideoEditingWindow;
extern main_window* Global_Window;
extern PaintCanvas* Global_Canvas;
extern SampleSet*   Global_SampleSet;
extern VideoEditingWindow* Global_WideoEditing_Window;
void globalObjectsInit();
void globalObjectsDelete();
void setGlobalWindow( main_window* _window);
void setGlobalCanvas( PaintCanvas* _canvas);
void setGlobalSampleSet(SampleSet* _sampleSet);
void setGlobalWideoEditingWindow(VideoEditingWindow* _videoEditingWindow);