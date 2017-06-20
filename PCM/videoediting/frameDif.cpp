#include "frameDif.h"
using namespace std;

FrameDif::FrameDif(void)
{

}
void FrameDif::initKeyFrame(set<int> &initKeyFrameNo, VideoCapture &capture, int totalFrameNumber)
{
	multimap<float, int> ratiomap;	//将每一帧的面积占比保存在multimap中
	int keyframeNum = totalFrameNumber / perKF;	//关键帧数量
	Mat preSrc, preFrame;
	capture.set(CV_CAP_PROP_POS_FRAMES, 0);	//从头开始获取原始帧
	capture >> preSrc;

	////统计时间	
	//string TimeFile ="initKeyTime.txt";
	//ofstream ftimeout(TimeFile,ios::app);
	//double initkeyT = (double)cvGetTickCount();

	if (!preSrc.empty())
	{
		cvtColor(preSrc, preFrame, CV_BGR2GRAY);

		if (!initKeyFrameNo.empty())
		{
			initKeyFrameNo.clear();
		}
		initKeyFrameNo.insert(0);	//默认第一帧为关键帧
		initKeyFrameNo.insert(totalFrameNumber - 1);	//默认最后一帧也为关键帧

		for (int frameNum = 1; frameNum < totalFrameNumber; frameNum++)		//从第二帧开始判断
		{
			Mat imgSrc;
			capture >> imgSrc;
			if (!imgSrc.data)
				break;
			Mat curframe;
			cvtColor(imgSrc, curframe, CV_BGR2GRAY);

			Mat difFrame;
			absdiff(curframe, preFrame, difFrame);	//帧差法
													/*imshow("difFrame",difFrame);
													waitKey(10);*/

			threshold(difFrame, difFrame, 30, 255, THRESH_BINARY);	//二值化
																	/*imshow("threshold",difFrame);
																	waitKey(50);*/
			int matArea = computeMatArea(difFrame);
			float ratio = (matArea*1.0) / (difFrame.rows*difFrame.cols);
			//if (ratio > 0.1) // 面积比例大于10% 
			//	initKeyFrameNo.push_back(frameNum);
			ratiomap.insert(pair<float, int>(ratio, frameNum));		//将每一帧的面积占比保存在multimap中,默认从小到大排序
			preFrame = curframe;
		}

		multimap<float, int>::reverse_iterator rit = ratiomap.rbegin();
		for (int i = 0; i<keyframeNum; i++, rit++)
		{
			//为了使关键帧分配的较为均匀，如果set中已有与插入元素a相隔较近的元素，此处取区间[a-3,a+3]，则rit++，不插入该元素
			while (initKeyFrameNo.lower_bound(rit->second - 3) != initKeyFrameNo.upper_bound(rit->second + 3))		//第一个大于等于a-3的元素与第一个大于a+3的元素相同，则说明set中没有属于区间[a-3, a+3]的元素
			{
				rit++;
			}
			initKeyFrameNo.insert(rit->second);
		}

		//////////////////////////////////////////////////////////////////////////写入文件
		/*ofstream ratiomapOut("ratiomap.txt", ios::app);
		for (multimap<float, int>::reverse_iterator rit=ratiomap.rbegin(); rit != ratiomap.rend();rit++)
		{
		ratiomapOut<<rit->first<<" , "<<rit->second<<endl;
		}

		ofstream initKeyFrameOut("initkeyframeNo.txt", ios::app);
		for (set<long>::iterator it=initKeyFrameNo.begin(); it != initKeyFrameNo.end(); it++)
		{
		initKeyFrameOut<<*it<<" , ";
		}*/
		//////////////////////////////////////////////////////////////////////////
	}

	////统计时间
	//initkeyT=(double)cvGetTickCount()-initkeyT;
	//float initkeyTime=initkeyT/(cvGetTickFrequency()*1000);;
	//ftimeout<<initkeyTime<<"ms"<<endl;
}

int FrameDif::computeMatArea(Mat frame)
{
	int areaNum = 0;
	for (int y = 0; y < frame.rows; y++)
	{
		for (int x = 0; x< frame.cols; x++)
		{
			if (frame.ptr<uchar>(y)[x] == 255)
			{
				areaNum++;
			}
		}
	}
	return areaNum;
}

FrameDif::~FrameDif(void)
{
}
