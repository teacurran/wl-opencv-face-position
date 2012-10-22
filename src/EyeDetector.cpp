#include "EyeDetector.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>
#include <iostream>
#include <vector>
#include <highgui.h>

using namespace std;

EyeDetector::EyeDetector() {
	init();
}

EyeDetector::EyeDetector(char *leftClassifier, char *rightClassifier) {
	init();

	setLeftEyeCascade(leftClassifier);
	setRightEyeCascade(rightClassifier);
}

EyeDetector::~EyeDetector() {
	if (eyeCascadeLeft != NULL) {
		cvReleaseHaarClassifierCascade(&eyeCascadeLeft);
	}
	if (eyeCascadeRight != NULL) {
		cvReleaseHaarClassifierCascade(&eyeCascadeRight);
	}

	if (storage != NULL) {
		cvReleaseMemStorage(&storage);
	}
	storage = 0;
}

void EyeDetector::find(IplImage *image, CvPoint *leftEye, CvPoint *rightEye) {
	IplImage *imgCopy = cvCreateImage(cvSize(image->width, image->height), 8,
			1);
	if (image->nChannels == 3) {
		cvCvtColor(image, imgCopy, CV_BGR2GRAY);
	} else {
		cvConvert(image, imgCopy);
	}

	double percentageFromTop = 6;
	double percentageFromBottom = 45;

	cvSetImageROI(imgCopy, /* the source image */
			cvRect(0, /* x = start from leftmost */
				(image->height * percentageFromTop / 100), /* y = a few pixels from the top */
				image->width, /* width = same width with the face */
				image->height - (image->height * percentageFromBottom / 100) /* height = 1/3 of face height */
			)
		);

	IplImage *eyeBlock = cvCreateImage(
			cvSize(imgCopy->width, image->height - (image->height * percentageFromBottom / 100)),
			8, 1);

	cvCopy(imgCopy, eyeBlock);

	cvResetImageROI(imgCopy);

	//cvConvert( image, eyeBlock);
	cvReleaseImage(&imgCopy);

	IplImage *eyeBlockSized = 0;
	double eyeScale = 1;
	if (eyeBlock->width < width_min) {
		eyeScale = width_min / (double) eyeBlock->width;

		eyeBlockSized = cvCreateImage(
				cvSize(eyeBlock->width * eyeScale, eyeBlock->height * eyeScale),
				eyeBlock->depth, 1);
		cvResize(eyeBlock, eyeBlockSized, CV_INTER_LINEAR);
	} else if (eyeBlock->width > width_max) {
		eyeScale = width_max / (double) eyeBlock->width;

		eyeBlockSized = cvCreateImage(
				cvSize(eyeBlock->width * eyeScale, eyeBlock->height * eyeScale),
				eyeBlock->depth, 1);
		cvResize(eyeBlock, eyeBlockSized, CV_INTER_LINEAR);
	}

	if (eyeBlockSized) {
		cvReleaseImage(&eyeBlock);
		eyeBlock = eyeBlockSized;
	}

	if (opt_debug) {
		printf("\tscaling eyes to %d, %g%%\n", eyeBlock->width, eyeScale);
	}

	//printf("eyeBlock size is %dx%d\n", eyeBlock->width, eyeBlock->height);

	vector<CvRect*> leftEyes;
	vector<CvRect*> rightEyes;

	int eye_min_neighbors = 3; // 2
	double eye_scale_factor = 1.05;
	int halfEyeBlock = cvRound(eyeBlock->width / 2);

	/* detect the left eyes */
	if (eyeCascadeLeft) {

		CvSeq *eyes = cvHaarDetectObjects(eyeBlock, /* the source image, with the estimated location defined */
		eyeCascadeLeft, /* the eye classifier */
		storage, /* memory buffer */
		eye_scale_factor, eye_min_neighbors, 0, cvSize(18, 12) /* minimum detection scale */
		);

		//printf("%d left eyes found\n", eyes->total);

		for (int i = 0; i < (eyes ? eyes->total : 0); i++) {
			/* get one eye */
			CvRect *eye = (CvRect*) cvGetSeqElem(eyes, i);

			if (CvUtils::getRectCenterPoint(eye).x < halfEyeBlock) {
				leftEyes.push_back(eye);
			} else {
				rightEyes.push_back(eye);
			}
		}
	} else {
		fprintf(stderr, "Error: left eye cascade not loaded\n");
	}

	/* detect the right eyes */
	if (eyeCascadeRight) {
		CvSeq *eyes = cvHaarDetectObjects(eyeBlock, /* the source image, with the estimated location defined */
		eyeCascadeRight, /* the eye classifier */
		storage, /* memory buffer */
		eye_scale_factor, eye_min_neighbors, 0, cvSize(18, 12) /* minimum detection scale */
		);

		for (int i = 0; i < (eyes ? eyes->total : 0); i++) {
			/* get one eye */
			CvRect *eye = (CvRect*) cvGetSeqElem(eyes, i);

			if (CvUtils::getRectCenterPoint(eye).x < halfEyeBlock) {
				leftEyes.push_back(eye);
			} else {
				rightEyes.push_back(eye);
			}
		}
	} else {
		fprintf(stderr, "Error: right eye cascade not loaded\n");
	}

	if (leftEyes.size() > 0 && rightEyes.size() > 0) {
		if (findMethod == FIND_AVERAGE) {
			CvPoint avgLeft = CvUtils::getAverageCenterPoint(leftEyes);
			leftEye->x = avgLeft.x;
			leftEye->y = avgLeft.y;

			CvPoint avgRight = CvUtils::getAverageCenterPoint(rightEyes);
			rightEye->x = avgRight.x;
			rightEye->y = avgRight.y;
		}
		if (findMethod == FIND_LARGEST) {
			CvRect *largestLeft = CvUtils::getLargestRect(leftEyes);
			CvRect *largestRight = CvUtils::getLargestRect(rightEyes);
			if (largestLeft && largestRight) {
				CvPoint largestLeftPoint = CvUtils::getRectCenterPoint(
						largestLeft);
				leftEye->x = largestLeftPoint.x;
				leftEye->y = largestLeftPoint.y;

				CvPoint largestRightPoint = CvUtils::getRectCenterPoint(
						largestRight);
				rightEye->x = largestRightPoint.x;
				rightEye->y = largestRightPoint.y;
			}
		}
	}

	if (opt_debug) {
		printf("\tLeft Eye: %d,%d\n", leftEye->x, leftEye->y);
		printf("\tRight Eye: %d,%d\n", rightEye->x, rightEye->y);
	}

	if (leftEye->x > 0) {
		leftEye->x = leftEye->x / eyeScale;
	}
	if (rightEye->x > 0) {
		rightEye->x = rightEye->x / eyeScale;
	}
	if (leftEye->y > 0) {
		leftEye->y = (image->height * percentageFromTop / 100) + (leftEye->y / eyeScale);
	}
	if (rightEye->y > 0) {
		rightEye->y = (image->height * percentageFromTop / 100) + (rightEye->y / eyeScale);
	}

	if (opt_debug) {
		printf("\tLeft Eye Corrected: %d,%d\n", leftEye->x, leftEye->y);
		printf("\tRight Eye Corrected: %d,%d\n", rightEye->x, rightEye->y);
	}

	if (opt_show_ui) {

		for (int i = 0; i < leftEyes.size(); i++) {
			CvRect *eye = leftEyes[i];
			cvRectangle(
				eyeBlock,
				cvPoint(
					eye->x,
					eye->y
				),
				cvPoint(
					eye->x + eye->width,
					eye->y + eye->height
				),
				CV_RGB(255, 0, 0),
				1, 8, 0
			);
		}
		CvUtils::drawCrosshair(leftEye, eyeBlock, 255, 0, 0);

		for (int i = 0; i < rightEyes.size(); i++) {
			CvRect *eye = rightEyes[i];
			cvRectangle(
				eyeBlock,
				cvPoint(
					eye->x,
					eye->y
				),
				cvPoint(
					eye->x + eye->width,
					eye->y + eye->height
				),
				CV_RGB(0, 0, 255),
				1, 8, 0
			);
		}

		CvUtils::drawCrosshair(rightEye, eyeBlock, 0, 0, 255);

		cvNamedWindow("eyes", 1);
		cvShowImage("eyes", eyeBlock);
		cvResizeWindow("eyes", eyeBlock->width, eyeBlock->height);
	}

	cvReleaseImage(&eyeBlock);

}

void EyeDetector::init() {
	eyeCascadeLeft = 0;
	eyeCascadeRight = 0;
	storage = cvCreateMemStorage(0);
	findMethod = FIND_LARGEST;
	opt_show_ui = false;
	opt_debug = false;
	opt_show_ui = false;
	width_min = 150;
	width_max = 300;
}

void EyeDetector::setDebug(bool debug) {
	opt_debug = debug;
}

void EyeDetector::setShowUi(bool value) {
	opt_show_ui = value;
}

void EyeDetector::setLeftEyeCascade(char * classifier) {
	eyeCascadeLeft = (CvHaarClassifierCascade*) cvLoad(classifier, 0, 0, 0);
	if (!eyeCascadeLeft) {
		fprintf(stderr,
				"ERROR: Could not load left eye classifier cascade: %s\n",
				classifier);
		return;
	}
	printf("Left Eye detection cascade loaded\n");
}

void EyeDetector::setRightEyeCascade(char * classifier) {
	eyeCascadeRight = (CvHaarClassifierCascade*) cvLoad(classifier, 0, 0, 0);
	if (!eyeCascadeRight) {
		fprintf(stderr,
				"ERROR: Could not load right eye classifier cascade: %s\n",
				classifier);
		return;
	}
	printf("Right Eye detection cascade loaded\n");
}
