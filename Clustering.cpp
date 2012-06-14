#include "Clustering.h"

ofstream outDebug("D:\\debug.txt", ios::out);
Clustering::Clustering(void)
{
}


Clustering::~Clustering(void)
{
}


void getEllipseParam(const Mat* covMat, Size& axesSize, double& angle)
{
	//x
	double xVals[2] = {-1, 0};
	Mat xBasis (2,1, CV_64F, xVals);

	//SVD of covMat
	SVD svd(*covMat);
//	double fA = svd.w.at<double>(0);
//	double sA = svd.w.at<double>(1);
//	axesSize = Size(40, 15);
	angle =  acosf(svd.u.col(1).dot(xBasis))*180/CV_PI;
//	if (svd.u.at<double>(1,0) < 0.0) // dot product always return the minimum angle between the two vectors. When the y coord.< 0 then 
// 		angle = -angle;
}


void getCovarianceMatx(const int row, const int col, const Mat* cluster, Mat& covMat, double& meanX, double& meanY)
{
	int maxRow = cluster->rows;
	int maxCols = cluster->cols;
	double sumR, sumC, sumRR, sumCC, sumRC, sumW;
	sumRR = sumCC = sumR = sumC = sumW = sumRC = 0.0;

	int cRow = row -1;
	while (cRow <= row+1)
	{
		int cCol = col -1;
		if (cRow >=0 && cRow < maxRow)
		{
			const float* ptr = cluster->ptr<float>(cRow);
			while (cCol <= col+1)
			{
				if (cCol >= 0 && cCol < maxCols)
				{
					int val = ptr[cCol];
					int r = cRow*val;
					int c = cCol*val;
					
					sumW += val;
					sumR += r;
					sumC += c;
					sumRR += r*cRow;
					sumCC += c*cCol;
					sumRC += val*cRow*cCol;
				}
				cCol++;
			}
		}
		cRow++;
	}

	meanX = sumC/sumW;
	meanY = sumR/sumW;
	double covRC = (sumRC/sumW)-(meanX*meanY);
	covMat.at<double>(0,0) = (sumRR/sumW)-powf(meanY,2);
	covMat.at<double>(0,1) = covRC;
	covMat.at<double>(1,0) = covRC;
	covMat.at<double>(1,1) = (sumCC/sumW)-powf(meanX,2);

//	double vals[2][2] = {{(sumRR/sumW)-powf(meanY,2), covRC},{covRC, (sumCC/sumW)-powf(meanX,2)}};
//	covMat =  Mat(2, 2, CV_64F, vals);

}

void getDistribution(const int row, const int col, const Mat* cluster, double& meanX, double& meanY, double& sigmaX, double& sigmaY)
{
	int maxRow = cluster->rows;
	int maxCols = cluster->cols;
	double sumR, sumC, sumRR, sumCC, sumW;
	sumRR = sumCC = sumR = sumC = sumW = 0.0;

	int cRow = row -1;
	while (cRow <= row+1)
	{
		int cCol = col -1;
		if (cRow >=0 && cRow < maxRow)
		{
			const float* ptr = cluster->ptr<float>(cRow);
			while (cCol <= col+1)
			{
				if (cCol >= 0 && cCol < maxCols)
				{
					int val = ptr[cCol];
					int r = cRow*val;
					int c = cCol*val;
					
					sumW += val;
					sumR += r;
					sumC += c;
					sumRR += r*r;
					sumCC += c*c;
				}
				cCol++;
			}
		}
		cRow++;
	}
	meanX = sumC/sumW;
	meanY = sumR/sumW;
	sigmaX = sqrtf((sumCC/sumW) - powf(meanX, 2));
	sigmaY = sqrtf((sumRR/sumW) - powf(meanY, 2));


}

bool isBiggestNeigh(int val, int row, int col, const Mat* img)
{
	int maxRow = img->rows;
	int maxCols = img->cols;
	bool bigger = true;
	int cRow = row -1;
	while (bigger &&  cRow <= row+1)
	{
		int cCol = col -1;
		if (cRow >=0 && cRow < maxRow)
		{
			const float* ptr = img->ptr<float>(cRow);
			while (bigger && cCol <= col+1)
			{
				if (cCol >= 0 && cCol < maxCols)
				{
					if (cRow != row || cCol != col)
						bigger = (val > ptr[cCol]);
				}
				cCol++;
			}
		}
		cRow++;
	}
	return bigger;
}


XnRGB24Pixel Clustering::get_ColorHeight_Person(const Mat* colorMap, const Mat* heightMap, float& height, int row, int col)
{
	XnRGB24Pixel color;
	int initY = max((row-1)*HEIGHT_BIN, 0);
	int initX = max((col-1)*WIDHT_BIN, 0);
	int endY = min((row+2)*HEIGHT_BIN, colorMap->rows);
	int endX = min((col+2)*WIDHT_BIN, colorMap->cols);

	int cBlue, cGreen, cRed;
	cBlue = cGreen = cRed = 0;

	int pointCont = 0;
	for (int y = initY; y < endY; y++)
	{
		const uchar* ptr = colorMap->ptr<uchar>(y);
		const float* ptrH = heightMap->ptr<float>(y);
		for (int x = initX; x < endX; x++)
		{
			if (ptr[3*x] != 255 && ptr[3*x+1]!=255 && ptr[3*x+2]!=255)
			{
				pointCont++;
				cBlue += ptr[3*x];
				cGreen += ptr[3*x+1];
				cRed += ptr[3*x+2];

				height += ptrH[x];
			}
		}
	}
			
	color.nBlue = cBlue/pointCont;
	color.nGreen = cGreen/pointCont;
	color.nRed = cRed/pointCont;
	height /= pointCont;
	return color;
}



list<Point> Clustering::clusterImage(Mat& img, const Mat* colorMap, const Mat* heightMap)
{
	list<Point> clusters;
	assert(img.channels() == 3);
	assert(img.type() == CV_8UC3);

	int nRows = img.rows/HEIGHT_BIN;
	int nCols = img.cols/WIDHT_BIN;
	Mat clusterImg = Mat::zeros(nRows ,nCols, CV_32F); //matrix to store the number of points in each bin
	Mat tmp = Mat::ones(img.rows, img.cols, CV_32F);
	
	//fill the bins with the number of red points in each bin
	for (int i = 0; i < img.rows; i++)
	{
		uchar* imgPtr = img.ptr<uchar>(i);
		if (i%HEIGHT_BIN == 0) // debug for drawing horizontal lines
			line(tmp, Point(0, i), Point(tmp.cols-1, i), Scalar::all(0));

		for (int j = 0; j < img.cols; j++)
		{
			if (j%WIDHT_BIN == 0)// debug for drawing vertical lines
				line(tmp, Point(j,0), Point(j, tmp.rows-1), Scalar::all(0));

			int col = j*3;

			//look for red spots in the image
			if (imgPtr[col]  == 0 && imgPtr[col+1] == 0  && imgPtr[col+2] == 255)
			{
////				outDebug << (int)imgPtr[j] << ", " << (int)imgPtr[j+1] << ", " << (int)imgPtr[j+2] << endl;
//
//				//find the correct bin
  				int rowBin = (int)i/HEIGHT_BIN;
				int colBin = (int)j/WIDHT_BIN;
				clusterImg.ptr<float>(rowBin)[colBin] += 1;
			//	imgPtr[col] = 0; imgPtr[col+1] = 0; imgPtr[col+2] = 0;
			}
		}
	}
	double meanX, meanY, angle;
	angle = meanX = meanY = 0.0;
	Size axesSize;
	Mat covMat(2,2, CV_64F);
	float height = -5000;
	char heightStr[50];
	//Non maximum supresion
	for (int i = 0; i < clusterImg.rows; i++)
	{
		float* ptr = clusterImg.ptr<float>(i);
		for (int j = 0; j < clusterImg.cols; j++)
		{
			//col and row in the original image
			int col = j*WIDHT_BIN;
			int row = i*HEIGHT_BIN;
			int num = ptr[j];
			if (num > 100 && isBiggestNeigh(num, i, j, &clusterImg))
			{
				getCovarianceMatx(i, j, &clusterImg, covMat, meanX, meanY);
				getEllipseParam(&covMat, axesSize, angle);
				//axesSize = Size(img.rows/15, img.cols/45);
				axesSize = Size(BIG_AXIS, SMALL_AXIS);
				meanX *= WIDHT_BIN;
				meanY *= HEIGHT_BIN;
				
				XnRGB24Pixel c = get_ColorHeight_Person(colorMap, heightMap, height, i, j);
				itoa(height, heightStr, 10);
				ellipse(tmp, Point(int(meanX)+(WIDHT_BIN/2), int(meanY) + (HEIGHT_BIN/2)), axesSize, angle, 0.0, 360, Scalar(c.nBlue, c.nGreen, c.nRed), -1);
				ellipse(img, Point(int(meanX)+(WIDHT_BIN/2), int(meanY) + (HEIGHT_BIN/2)), axesSize, angle, 0.0, 360, Scalar(c.nBlue, c.nGreen, c.nRed), -1);
				circle(tmp, Point(int(meanX)+(WIDHT_BIN/2), int(meanY) + (HEIGHT_BIN/2)), SMALL_AXIS, Scalar::all(0), -1);
 				circle(img, Point(int(meanX)+(WIDHT_BIN/2), int(meanY) + (HEIGHT_BIN/2)), SMALL_AXIS, Scalar::all(0), -1);
				putText(img, heightStr, Point(int(meanX)-(BIG_AXIS+10), int(meanY) - (SMALL_AXIS+10)),FONT_HERSHEY_PLAIN, 0.8, Scalar(0, 0, 0));
				clusters.push_back(Point(col+(WIDHT_BIN/2),row + (HEIGHT_BIN/2)));
			}
			//if (num > 0)
			//{
			//	char txt[15];
			//	itoa(num, txt, 10);
			//	putText(tmp, txt, Point(col+(WIDHT_BIN/2.5),row + (HEIGHT_BIN/2)),FONT_HERSHEY_PLAIN, 0.8, Scalar::all(0));
			//	//circle(img, Point(col, row), 3, Scalar::all(0), 3);
			//}
			//else
			//	putText(tmp, "0", Point(col+(WIDHT_BIN/2.5),row + (HEIGHT_BIN/2)),FONT_HERSHEY_PLAIN, 0.8, Scalar::all(0));
		}
	}
	
//	cv::imshow("Original", img);
//	cv::imshow("Clustering", tmp);
//	cv::waitKey(50);
	return clusters;

}