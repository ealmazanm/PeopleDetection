#pragma once
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <list>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <ctype.h>
#include "XnCppWrapper.h"

using namespace std;
using namespace cv;
using namespace xn;


const int WIDHT_BIN = 40;
const int HEIGHT_BIN = 20;

const int BIG_AXIS = 20;
const int SMALL_AXIS = 8;


class Clustering
{
public:
	Clustering(void);
	~Clustering(void);
	//img must be 1280x480
	list<Point> clusterImage(Mat& img, const Mat* colorMap, const Mat* heightMap);

private:
	XnRGB24Pixel get_ColorHeight_Person(const Mat* colorMap, const Mat* heightMap, float& height, int row, int col);
};

