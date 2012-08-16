#include "Clustering.h"

ofstream outDebug("D:\\debug.txt", ios::out);
Mat tmp;
Clustering::Clustering(void)
{
	term = TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 50, 0.1);
	init = false;

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


void getCovarianceMatx(const int row, const int col, const Mat* cluster, Mat& covMat, Point& mean)
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

	mean.x = sumC/sumW;
	mean.y = sumR/sumW;
	double covRC = (sumRC/sumW)-(mean.x*mean.y);
	covMat.at<double>(0,0) = (sumRR/sumW)-powf(mean.y,2);
	covMat.at<double>(0,1) = covRC;
	covMat.at<double>(1,0) = covRC;
	covMat.at<double>(1,1) = (sumCC/sumW)-powf(mean.x,2);

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

int Clustering::probabilityMatching(const Mat* probImage, const Rect* wind)
{
	Mat prob_roi = (*probImage)(*wind);
	Scalar s = cv::sum(prob_roi);
	return cv::sum(prob_roi).val[0];
}

void Clustering::enhanceColor(XnRGB24Pixel& c)
{
	if (c.nRed > 200)
		c.nRed = 255;
	else if (c.nRed < 50)
		c.nRed = 0;
	else
		c.nRed = 127;

	if (c.nBlue > 200)
		c.nBlue = 255;
	else if (c.nBlue < 50)
		c.nBlue = 0;
	else
		c.nBlue = 127;

		if (c.nGreen > 200)
		c.nGreen = 255;
	else if (c.nGreen < 50)
		c.nGreen = 0;
	else
		c.nGreen = 127;
}

void Clustering::trackPreviousPeople(Mat& clusterImg, const Mat* img)
{
	Matx21d v1(1,0);
	list<Point>::iterator iterMeans = peopleMeans.begin();
	list<Scalar>::iterator iterColors = peopleColor.begin();
	list<MatND>::iterator iterHist = peopleHist.begin();
	list<double>::iterator iterAngle = peopleAngle.begin();
	list<Point>::iterator iterDirect = peopleDirection.begin();

	//Debug
	list<int>::iterator iterFrames = peopleFrames.begin();
	list<int>::iterator iterBGPixels = peopleNBGPixels.begin();
	list<int>::iterator iterMatching = peopleMatching.begin();

	//RGB values
 	float rranges[] = { 0, 256}; 
	float granges[] = { 0, 256 };
	float branges[] = { 0, 256 };
	const float* ranges_RGB[] = {rranges, granges, branges}; 
	int channels_RGB[] = {0, 1, 2};	

	//HSV values
	float hranges[] = { 0, 179}; 
	float sranges[] = { 0, 255};
	const float* ranges_HSV[] = {hranges, sranges}; 
	int channels_HSV[] = {0, 1};	

	int cont = 0;
	
	while (iterMeans != peopleMeans.end())
	{
		cont++;
		Mat probImage;
		//create prob image for every person
		if (HSV_MEANSHIFT)
			calcBackProject(img, 1, channels_HSV, *iterHist, probImage, ranges_HSV);
		else
			calcBackProject(img, 1, channels_RGB, *iterHist, probImage, ranges_RGB);
//BEGIN DEBUG
		int max = -1;
		for (int j = 0; j < probImage.rows; j++)
		{
			uchar* ptr = probImage.ptr<uchar>(j);
			for (int i = 0; i < probImage.cols; i++)
			{
				if (ptr[i] > max)
					max = ptr[i];
			}
		}
		if (cont == 1)
			imshow("ProbImage", probImage);
//END DEBUG

		Point meanInit = *iterMeans;

		int widthFinal = std::min(WIDHT_BIN/2, img->cols - meanInit.x)*2;
		int heightFinal = std::min(HEIGHT_BIN/2, img->rows - meanInit.y)*2;
			
		int xInit = meanInit.x-(widthFinal/2);
		int yInit = meanInit.y-(heightFinal/2);

		

		Rect wind(xInit, yInit, widthFinal, heightFinal);

		rectangle(tmp, wind, Scalar::all(0), 2);
		meanShift(probImage, wind, term);
		Point meanFinal(wind.x+WIDHT_BIN/2, wind.y+HEIGHT_BIN/2);
		rectangle(tmp, wind, Scalar::all(0), 3);



		int col = meanFinal.x/WIDHT_BIN;
		int row = meanFinal.y/HEIGHT_BIN;
		if (clusterImg.ptr<float>(row)[col] < 50)
		{
			//Create report (time, matching factor, BG subtraction
			//cout << "Number of frames: " << *iterFrames << endl;
			//cout << "BG previous: " << *iterBGPixels << endl;
			//cout << "BG current: " << clusterImg.ptr<float>(row)[col] << endl;
			//cout << "Matching previous: " << *iterMatching << endl;
			//cout << "Matching current: " << probabilityMatching(&probImage, &wind) << endl;
			//cout << endl;
			
			cout << "Probability Avg: " << *iterMatching/ *iterFrames << endl;
//			waitKey(0);
			iterMeans = peopleMeans.erase(iterMeans);
			iterColors = peopleColor.erase(iterColors);
			iterHist = peopleHist.erase(iterHist);
			iterAngle = peopleAngle.erase(iterAngle);
			iterDirect = peopleDirection.erase(iterDirect);

			iterFrames = peopleFrames.erase(iterFrames);
			iterBGPixels = peopleNBGPixels.erase(iterBGPixels);
			iterMatching = peopleMatching.erase(iterMatching);
		}
		else
		{
			int xInit, yInit, w, h;
			xInit = std::max(0,col-1);
			yInit = std::max(0, row-1);
			w = std::min(clusterImg.cols-xInit, 3);
			h = std::min(clusterImg.rows-yInit, 3);

			Mat clusterRoi = clusterImg(Rect(xInit, yInit, w, h));
			clusterRoi = clusterRoi.zeros(clusterRoi.rows, clusterRoi.cols, CV_32F);

			Matx21d v2 (-(meanInit.y-meanFinal.y), meanInit.x-meanFinal.x);
			*iterAngle = acosf(v1.dot(v2)/(norm(v1)*norm(v2)))*180/CV_PI;
			*iterMeans = meanFinal;
			*iterDirect = meanInit;

			*iterFrames += 1;

			int prob = probabilityMatching(&probImage, &wind);
			cout << prob << endl;
			*iterMatching += prob;
//			cout << *iterAngle << endl;

			iterMeans++;
			iterColors++;
			iterHist++;
			iterAngle++;
			iterDirect++;

			iterFrames++;
			iterBGPixels++;
			iterMatching++;
		}
	}
}

void Clustering::createProbabilityImage(Mat& probImage, const Mat* img, const Rect* roi, const Mat* mask, MatND& hist)
{
	//only rgb
	Mat img_roi = (*img)(*roi); //create a roi in hsv
	Mat mask_roi = (*mask)(*roi);	

	int rbins = 30, gbins = 32, bbins = 32; 
	int histSize_RGB[] = {rbins, gbins, bbins}; 
	float rranges[] = { 0, 256}; 
	float granges[] = { 0, 256 };
	float branges[] = { 0, 256 };
	const float* ranges_RGB[] = {rranges, granges, branges}; 
	int channels_RGB[] = {0, 1, 2};	

	int hbins = 30, sbins = 32;
	int histSize_HSV[] = {hbins, sbins}; 
	float hranges[] = { 0, 179}; 
	float sranges[] = { 0, 255};
	const float* ranges_HSV[] = {hranges, sranges}; 
	int channels_HSV[] = {0, 1};	

	if (HSV_MEANSHIFT)
	{
		calcHist(&img_roi, 1, channels_HSV, mask_roi, hist, 2, histSize_HSV, ranges_HSV, true, false);
		normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
		calcBackProject(img, 1, channels_HSV, hist, probImage, ranges_HSV);
	}
	else
	{
		calcHist(&img_roi, 1, channels_RGB, mask_roi, hist, 2, histSize_RGB, ranges_RGB, true, false);
		normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
		calcBackProject(img, 1, channels_RGB, hist, probImage, ranges_RGB);
	}
}

void Clustering::clusterImage(const Mat* colorMaps, const Mat* hsvImag)
{
	Mat gray;
	cvtColor(*colorMaps, gray, CV_RGB2GRAY);
	Mat mask;
	threshold( gray, mask, 128, 255,THRESH_BINARY_INV);
	
	Mat imgFinal;
	if (HSV_MEANSHIFT)
		imgFinal = *hsvImag;
	else
		imgFinal = *colorMaps;

	Mat clusterImg = Mat::zeros(hsvImag->rows/HEIGHT_BIN ,hsvImag->cols/WIDHT_BIN, CV_32F); //matrix to store the number of points in each bin
	tmp = Mat::ones(hsvImag->rows, hsvImag->cols, CV_32F);

	//fill the bins with the number of red points in each bin
	for (int i = 0; i < hsvImag->rows; i++)
	{
		const uchar* imgPtr = hsvImag->ptr<const uchar>(i);
		if (i%HEIGHT_BIN == 0) // debug for drawing horizontal lines
			line(tmp, Point(0, i), Point(tmp.cols-1, i), Scalar::all(0));

		for (int j = 0; j < hsvImag->cols; j++)
		{
			if (j%WIDHT_BIN == 0)// debug for drawing vertical lines
				line(tmp, Point(j,0), Point(j, tmp.rows-1), Scalar::all(0));

			//look for red spots in the image
			//if (imgPtr[j*3]  > 0 && imgPtr[j*3+1] > 0  && imgPtr[j*3+2] < 255)
			if (imgPtr[j*3+2] < 240)
			{
//				//find the correct bin
  				int rowBin = (int)i/HEIGHT_BIN;
				int colBin = (int)j/WIDHT_BIN;
				clusterImg.ptr<float>(rowBin)[colBin] += 1;
			}
		}
	}
	for (int i = 0; i < clusterImg.rows; i++)
	{
		float* ptr = clusterImg.ptr<float>(i);
		for (int j = 0; j < clusterImg.cols; j++)
		{
			int num = ptr[j];
			int col = j*WIDHT_BIN;
			int row = i*HEIGHT_BIN;
			if (num > 0)
			{
				char txt[15];
				itoa(num, txt, 10);
				putText(tmp, txt, Point(col+(WIDHT_BIN/2.5),row + (HEIGHT_BIN/2)),FONT_HERSHEY_PLAIN, 0.8, Scalar::all(0));
			}
			else
				putText(tmp, "0", Point(col+(WIDHT_BIN/2.5),row + (HEIGHT_BIN/2)),FONT_HERSHEY_PLAIN, 0.8, Scalar::all(0));

		}
	}

	//check the previos people
	if (peopleMeans.size() > 0)
		trackPreviousPeople(clusterImg, &imgFinal);

	init = true;
	Mat covMat(2,2, CV_64F);
	double angle = 0.0;
	Point meanInit;
	Size axesSize;
	srand(time(NULL));
	Matx21d xOrt(1,0);
	//Non maximum supresion
	for (int i = 0; i < clusterImg.rows; i++)
	{
		float* ptr = clusterImg.ptr<float>(i);
		for (int j = 0; j < clusterImg.cols; j++)
		{
			int col = j*WIDHT_BIN;
			int row = i*HEIGHT_BIN;

			int num = ptr[j];
			if (num > 100 && isBiggestNeigh(num, i, j, &clusterImg))
			{
				getCovarianceMatx(i, j, &clusterImg, covMat, meanInit);
				meanInit.x = meanInit.x*WIDHT_BIN+(WIDHT_BIN/2);
				meanInit.y = meanInit.y*HEIGHT_BIN+(HEIGHT_BIN/2);
				getEllipseParam(&covMat, axesSize, angle);
				Rect wind(meanInit.x-(WIDHT_BIN/2), meanInit.y-(HEIGHT_BIN/2), WIDHT_BIN, HEIGHT_BIN);
				//create probability image using the rgb roi
				MatND hist;
				Mat probImage;
				createProbabilityImage(probImage, &imgFinal, &wind, &mask, hist);

				peopleDirection.push_back(meanInit);
				peopleHist.push_back(hist);
				peopleMeans.push_back(meanInit);
				peopleAngle.push_back(angle);
				Scalar c (rand()&255, rand()&255, rand()&255);
				peopleColor.push_back(c);

				//Debug
				int prob = probabilityMatching(&probImage, &wind);
				cout << "Probability (new): " << prob << endl;
				peopleMatching.push_back(prob);
				peopleFrames.push_back(1);
				peopleNBGPixels.push_back(num);



				ellipse(tmp, Point(int(meanInit.x)+(WIDHT_BIN/2), int(meanInit.y) + (HEIGHT_BIN/2)), axesSize, angle, 0.0, 360, c, -1);
			}

			//if (num > 0)
			//{
			//	char txt[15];
			//	itoa(num, txt, 10);
			//	putText(tmp, txt, Point(col+(WIDHT_BIN/2.5),row + (HEIGHT_BIN/2)),FONT_HERSHEY_PLAIN, 0.8, Scalar::all(0));
			//}
			//else
			//	putText(tmp, "0", Point(col+(WIDHT_BIN/2.5),row + (HEIGHT_BIN/2)),FONT_HERSHEY_PLAIN, 0.8, Scalar::all(0));

		}
	}
	imshow("Clusters", tmp);
}

void Clustering::drawPeople(Mat& img)
{
	list<Point>::iterator iterMeans = peopleMeans.begin();
	list<Scalar>::iterator iterColors = peopleColor.begin();
	list<double>::iterator iterAngle = peopleAngle.begin();
	list<Point>::iterator iterDirect = peopleDirection.begin();

	while (iterMeans != peopleMeans.end())
	{
		Size axesSize = Size(BIG_AXIS, SMALL_AXIS);
		Point mean = *iterMeans;
		ellipse(img, *iterMeans, axesSize, *iterAngle, 0.0, 360, *iterColors, -1);
		circle(img, *iterMeans, SMALL_AXIS, Scalar::all(0), -1);
		Point direct = *iterDirect;

		//Point meanVectorPerp (-(mean.y-direct.y), mean.x-direct.x);
		//Point dest (mean.x+2*meanVectorPerp.x, mean.y+2*meanVectorPerp.y);
		//Point orig (mean.x + 50, mean.y);

		//Matx21d v1 (orig.x-mean.x, orig.y-mean.y);
		//Matx21d v2 (dest.x-mean.x, dest.y-mean.y);

		//double angle = acosf(v1.dot(v2)/(norm(v1)*norm(v2)))*180/CV_PI;
		//cout << "Angle: " << angle << endl;

		//line(img, mean, orig, Scalar(0,255,0));
		//line(img, mean, dest, Scalar(255,0,0)); 
		//line(img, mean, direct, Scalar(0,0,255));

		iterMeans++;
		iterColors++;
		iterAngle++;
		iterDirect++;
	}
}