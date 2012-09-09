#ifndef __EYEDETECTOR_H__
#define __EYEDETECTOR_H__

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <vector>
#include "CvUtils.hpp"

using namespace std;


class EyeDetector {
	public:
		EyeDetector();
		EyeDetector(char *leftClassifier, char *rightClassifier);
		void setDebug(bool opt_debug);
		void setShowUi(bool value);
		void setLeftEyeCascade(char *classifier);
		void setRightEyeCascade(char *classifier);
		void find(IplImage *image, CvPoint *leftEye, CvPoint *rightEye);

		static const int FIND_AVERAGE = 1;
		static const int FIND_LARGEST = 2;
		~EyeDetector();

	private:
		int findMethod;
		int scale;
		void init();
		char *cascade_name;
		CvMemStorage* storage;
		CvHaarClassifierCascade* eyeCascadeLeft;
		CvHaarClassifierCascade* eyeCascadeRight;
		bool opt_debug;
		bool opt_show_ui;
		int width_min;
		int width_max;

		static int sum(const vector<int>& x) {
			int total = 0;
			for (int i=0; i<x.size(); i++) {
				total = total + x[i];
			}
			return total;
		}

};


 #endif

