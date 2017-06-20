#pragma once
#include "VideoEdittingParameter.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <map>
#include <set>
#include <fstream>

using namespace std;
using namespace cv;

class FrameDif
{
public:
	FrameDif(void);
	void initKeyFrame(std::set<int> &initKeyFrameNo, VideoCapture &capture, int totalFrameNumber);
	int computeMatArea(Mat frame);
	~FrameDif(void);

};