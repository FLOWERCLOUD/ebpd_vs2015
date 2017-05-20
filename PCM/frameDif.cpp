#include "frameDif.h"
using namespace std;

FrameDif::FrameDif(void)
{

}
void FrameDif::initKeyFrame(set<int> &initKeyFrameNo, VideoCapture &capture, int totalFrameNumber)
{
	multimap<float, int> ratiomap;	//��ÿһ֡�����ռ�ȱ�����multimap��
	int keyframeNum = totalFrameNumber / perKF;	//�ؼ�֡����
	Mat preSrc, preFrame;
	capture.set(CV_CAP_PROP_POS_FRAMES, 0);	//��ͷ��ʼ��ȡԭʼ֡
	capture >> preSrc;

	////ͳ��ʱ��	
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
		initKeyFrameNo.insert(0);	//Ĭ�ϵ�һ֡Ϊ�ؼ�֡
		initKeyFrameNo.insert(totalFrameNumber - 1);	//Ĭ�����һ֡ҲΪ�ؼ�֡

		for (int frameNum = 1; frameNum < totalFrameNumber; frameNum++)		//�ӵڶ�֡��ʼ�ж�
		{
			Mat imgSrc;
			capture >> imgSrc;
			if (!imgSrc.data)
				break;
			Mat curframe;
			cvtColor(imgSrc, curframe, CV_BGR2GRAY);

			Mat difFrame;
			absdiff(curframe, preFrame, difFrame);	//֡�
													/*imshow("difFrame",difFrame);
													waitKey(10);*/

			threshold(difFrame, difFrame, 30, 255, THRESH_BINARY);	//��ֵ��
																	/*imshow("threshold",difFrame);
																	waitKey(50);*/
			int matArea = computeMatArea(difFrame);
			float ratio = (matArea*1.0) / (difFrame.rows*difFrame.cols);
			//if (ratio > 0.1) // �����������10% 
			//	initKeyFrameNo.push_back(frameNum);
			ratiomap.insert(pair<float, int>(ratio, frameNum));		//��ÿһ֡�����ռ�ȱ�����multimap��,Ĭ�ϴ�С��������
			preFrame = curframe;
		}

		multimap<float, int>::reverse_iterator rit = ratiomap.rbegin();
		for (int i = 0; i<keyframeNum; i++, rit++)
		{
			//Ϊ��ʹ�ؼ�֡����Ľ�Ϊ���ȣ����set�����������Ԫ��a����Ͻ���Ԫ�أ��˴�ȡ����[a-3,a+3]����rit++���������Ԫ��
			while (initKeyFrameNo.lower_bound(rit->second - 3) != initKeyFrameNo.upper_bound(rit->second + 3))		//��һ�����ڵ���a-3��Ԫ�����һ������a+3��Ԫ����ͬ����˵��set��û����������[a-3, a+3]��Ԫ��
			{
				rit++;
			}
			initKeyFrameNo.insert(rit->second);
		}

		//////////////////////////////////////////////////////////////////////////д���ļ�
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

	////ͳ��ʱ��
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
