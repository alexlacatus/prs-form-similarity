// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include<iostream>
#include<fstream>
#include <set>
#include <numeric>
using namespace std;


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

int minInMask(Mat_<uchar> img, int x, int y) {

	int min = 255;

	int w[3][3] = { {3,2,3 },
					{2,0,2},
					{3,2,3} };



	for (int i = x - 1;i <= x + 1;i++)
		for (int j = y - 1;j <= y + 1;j++)
			if (i >= 0 && i < img.rows && j >= 0 && j < img.cols) {
				if (min > img(i, j) + w[i - x + 1][j - y + 1])
					min = img(i, j) + w[i - x + 1][j - y + 1];
			}

	return min;


}

Mat_<uchar> dt(Mat_<uchar> img) {

	Mat_<uchar> res(img.rows, img.cols);

	int max = 255;
	//imshow("img", img);
	for (int i = 0;i < img.rows;i++)
		for (int j = 0;j < img.cols;j++) {
			if (img(i, j) == 0)
				res(i, j) = 0;
			else
				res(i, j) = max;
		}

	for (int i = 0;i < res.rows;i++)
		for (int j = 0;j < res.cols;j++) {

			res(i, j) = minInMask(res, i, j);


		}

	for (int i = res.rows - 1;i >= 0;i--)
		for (int j = res.cols - 1;j >= 0;j--) {

			res(i, j) = minInMask(res, i, j);


		}
	return res;





}




cv::Point2f computeCenterOfMass(const cv::Mat_<uchar> binaryImage) {
	cv::Point2f centerOfMass(0, 0);
	int count = 0;

	for (int i = 0; i < binaryImage.rows; i++) {
		for (int j = 0; j < binaryImage.cols; j++) {
			if (binaryImage(i, j) == 255) {
				centerOfMass.x += j;
				centerOfMass.y += i;
				count++;
			}
		}
	}

	if (count > 0) {
		centerOfMass.x /= static_cast<float>(count);
		centerOfMass.y /= static_cast<float>(count);
	}

	return centerOfMass;
}


float computeSimilarity2(Mat_<uchar> dtres, Mat_<uchar> image) {

	cv::Point2f center1 = computeCenterOfMass(image);
	cv::Point2f center2 = computeCenterOfMass(dtres);

	//float sizeDifference = static_cast<float>(image.rows * image.cols) / (dtres.rows * dtres.cols);
	//float aspectRatioDifference = static_cast<float>(image.cols) / image.rows - static_cast<float>(dtres.cols) / dtres.rows;

	//// Set penalty weights (adjust as needed)
	//float sizePenaltyWeight = 0.5;
	//float aspectRatioPenaltyWeight = 50;

	//// Calculate penalty factors
	//float sizePenalty = sizePenaltyWeight * std::abs(sizeDifference - 1.0);
	//float aspectRatioPenalty = aspectRatioPenaltyWeight * std::abs(aspectRatioDifference);


	// Recenter both images to the common center
	//recenterImage(image, commonCenter);
	//imshow("dtres", dtres);
	//recenterImage(dtres, commonCenter);



	int rows = min(image.rows, dtres.rows);
	int cols = min(image.cols, dtres.cols);

	float squaredDifference = 0;


	int similarity = 0;
	int count = 0;
	for (int i = 0;i < rows;i++) {
		for (int j = 0;j < cols;j++) {
			if (image(i, j) == 0) {
				int difference = dtres(i, j);
				squaredDifference += difference;
				count++;
			}

		}
	}

	//float mse = static_cast<float>(squaredDifference);
	float similarity2 = -squaredDifference / count;// -sizePenalty - aspectRatioPenalty;


	return similarity2;


}

Mat_<uchar> moveTopLeft(Mat_<uchar> img) {
	int minrow = img.rows;
	int mincols = img.cols;
	for (int i = 0;i < img.rows;i++) {
		for (int j = 0;j < img.cols;j++) {
			if (img(i, j) == 0) {
				minrow = min(minrow, i);
				mincols = min(mincols, j);
			}
		}
	}

	Mat_<uchar> res(img.rows, img.cols);
	res.setTo(255);
	for (int i = 0;i < img.rows;i++) {
		for (int j = 0;j < img.cols;j++) {
			if (img(i, j) == 0) {
				res(i - minrow, j - mincols) = 0;
			}
		}
	}

	return res;

}

struct similarStruct {
	float similarity;
	int index;
};

boolean compareSimilarities(similarStruct s1, similarStruct s2) {
	return s1.similarity > s2.similarity;
}


void reverseBinaryImage(cv::Mat_<uchar> binaryImage) {
	// Iterate through each pixel in the image
	for (int i = 0; i < binaryImage.rows; i++) {
		for (int j = 0; j < binaryImage.cols; j++) {
			// Invert the pixel value (swap 0 and 255)
			binaryImage(i, j) = 255 - binaryImage(i, j);
		}
	}
}

void proj() {

	Mat_<uchar>training[72];
	Mat_<uchar>test[8];
	char classes[100][100] = { "bat","beetle","camel","bird" };
	vector<int>ytraining;
	vector<int>ytest;
	for (int c = 0;c < 4;c++) {
		for (int i = 1; i <= 20; i++) {
			char fname[102];


			if (i < 19)
				sprintf(fname, "training/%s-%d.png", classes[c], i);
			else
				sprintf(fname, "test/%s-%d.png", classes[c], i);

			Mat_<uchar> img = imread(fname, 0);
			Mat_<uchar> res(img.rows, img.cols);

			GaussianBlur(img, img, Size(5, 5), 0.8, 0.8);

			Canny(img, res, 50, 150, 3);
			reverseBinaryImage(res);
			//res = moveTopLeft(res);
			char imgName[102];
			sprintf(imgName, "%s-%d", classes[c], i);
			//imshow(imgName, res);
			if (i < 19) {
				ytraining.push_back(c);
				training[i - 1 + 18 * c] = res;
			}
			else {
				ytest.push_back(c);
				test[i - 19 + 2 * c] = res;
			}
		}
	}
	int score = 0;
	for (int i = 0;i < 8;i++) {
		vector<similarStruct> similarities;
		float simClass[4];
		for (int cls = 0;cls < 4;cls++) {
			simClass[cls] = 0;
		}
		Mat_<uchar> dtres = dt(test[i]);
		//imshow("dtres", dtres);
		//waitKey();
		for (int j = 0;j < 72;j++) {
			similarities.push_back({ computeSimilarity2(dtres,training[j]),j });
			//printf("dd:similarity %f, i:%d, j:%d\n", similarities[j].similarity, i, j);
			//simClass[j / 18] += similarities[j].similarity; //<0.5 ? 0 : similarities[j].similarity;
		}
		sort(similarities.begin(), similarities.end(), compareSimilarities);
		//printf("sorted: ");

		for (int v = 0;v < 5;v++) {
			//printf("%f", similarities[v].similarity);
			int j = similarities[v].index;
			simClass[j / 18] ++; //<0.5 ? 0 : similarities[j].similarity;
			char imgName[102];
			sprintf(imgName, "%d-%d", i, v);
			imshow("original", test[i]);
			imshow(imgName, training[similarities[v].index]);
		}
		waitKey();
		destroyAllWindows();

		//printf("sorted: \n");

		//sort(similarities.begin(), similarities.end(), compareSimilarities);
		//printf("similarity %d, i:%d, j:%d, class:%s\n", similarities[0].similarity, i, similarities[1].index,classes[similarities[1].index/18]);
		int indexMax = 0;
		for (int ii = 0;ii < 4;ii++) {
			if (simClass[indexMax] < simClass[ii]) {
				indexMax = ii;
			}

		}
		//printf("similarity %f,i : %d ,class:%s expected:%s\n", simClass[indexMax], i, classes[indexMax],classes[ytest[i]]);
		printf("votes %f %f %f %f,i : %d ,class:%s expected:%s\n", simClass[0], simClass[1], simClass[2], simClass[3], i, classes[indexMax], classes[ytest[i]]);
		score += indexMax == ytest[i] ? 1 : 0;
	}
	printf("Score %d/8", score);
	int dd;
	scanf("%d", &dd);
	waitKey();

}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 100 - Project form similarities\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 100:
				proj();
				break;
		}
	}
	while (op!=0);
	return 0;
}