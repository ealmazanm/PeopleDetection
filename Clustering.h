#pragma once
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <list>
#include <Utils.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <ctype.h>
#include "XnCppWrapper.h"

using namespace std;
using namespace cv;
using namespace xn;


const int WIDHT_BIN = 60;
const int HEIGHT_BIN = 40;

const int BIG_AXIS = 20;
const int SMALL_AXIS = 8;

const bool HSV_MEANSHIFT = true;


class Clustering
{
public:
	Clustering(void);
	~Clustering(void);
	//img must be 1280x480
//	void clusterImage(Mat& img, const Mat* colorMap, const Mat* heightMap);
	void clusterImage(const Mat* colorMap, const Mat* hsvImage);
	void drawPeople(Mat& img);

private:
	XnRGB24Pixel get_ColorHeight_Person(const Mat* colorMap, const Mat* heightMap, float& height, int row, int col);
	void enhanceColor(XnRGB24Pixel& c);
	void trackPreviousPeople(Mat& clusterImg, const Mat* colorImg);
	void createProbabilityImage(Mat& probImage, const Mat* colorMap, const Rect* roi, const Mat* mask, MatND& hist);
	int probabilityMatching(const Mat* probImage, const Rect* wind);

	list<MatND> peopleHist;
	list<Point> peopleMeans;
	list<Scalar> peopleColor;
	list<double> peopleAngle;
	list<int> peopleFrames;
	list<int> peopleNBGPixels;
	list<int> peopleMatching;


	//debug
	list<Point> peopleDirection;

	TermCriteria term;
	bool init;
};

