#ifndef CVUTILS_h
#define CVUTILS_h

using namespace std;

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <vector>
#include <cstdlib>
#include <stdio.h>

using std::abs;

class CvUtils {
private:
	void init(void);

public:


	static void createMask(IplImage * theMask, CvRect maskRegion, int zeroImage) {
		CvPoint topLeft, bottomRight;
		if(theMask!=NULL) {
			if(zeroImage) {
				cvZero(theMask);
			}
			topLeft.x = cvRound((maskRegion.x ));
			topLeft.y = cvRound((maskRegion.y ));
			bottomRight.x = cvRound((maskRegion.x + maskRegion.width));
			bottomRight.y =  cvRound((maskRegion.y + maskRegion.height));
			cvRectangle(theMask,topLeft,bottomRight,CV_RGB(255,255,255),-1,8,0);
		}
	}

	static void drawCrosshair(CvPoint *point, IplImage *image, int r, int g, int b) {
		cvLine(
			image,
			cvPoint(point->x-5, point->y),
			cvPoint(point->x+5, point->y),
			CV_RGB(r, g, b),
			1, 8, 0
		);
		cvLine(
			image,
			cvPoint(point->x, point->y-5),
			cvPoint(point->x, point->y+5),
			CV_RGB(r, g, b),
			1, 8, 0
		);
	}

	static void drawRect(CvRect rect, IplImage * theImage, CvScalar color) {
		double scale = 1.3;
		CvPoint topLeft, bottomRight;
		topLeft.x = cvRound((rect.x )/**scale*/);
		topLeft.y = cvRound((rect.y )/*scale*/);
		bottomRight.x = cvRound((rect.x + rect.width)/**scale*/);
		bottomRight.y =  cvRound((rect.y + rect.height)/**scale*/);
		cvRectangle(theImage,topLeft,bottomRight,color,1,8,0);
	}

	static int gamma_decompress(IplImage* src, IplImage* dest, const char* type) {
		if (src->depth != IPL_DEPTH_8U || (dest->depth != IPL_DEPTH_32F && dest->depth != IPL_DEPTH_64F)) {
			printf("gamma_decompress : input depth combination not supported \n");
			return -1;
		}

		if (src->nChannels != 1 || dest->nChannels != 1) {
			printf("gamma_decompress : single channel images only\n");
			return -1;
		}

		cvConvertScale(src, dest, 1.0 / 255.0);

		if (!strcmp(type, "Linear")) {
			// do nothing
		} else if (!strcmp(type, "NTSC")) {
			cvPow(dest, dest, 2.2);
		} else if (!strcmp(type, "sRGB")) {
			if (dest->depth == IPL_DEPTH_32F) {
				int i, j;
				float* p_fDest;

				for (i = 0; i < src->height; i++) {
					p_fDest = (float*) cvPtr2D(dest, i, 0);

					for (j = 0; j < src->width; j++) {
						if (p_fDest[j] > .04045f)
							p_fDest[j] = pow(double(p_fDest[j] + .055f) / 1.055f, 2.4);
						else
							p_fDest[j] /= 12.92f;
					}
				}
			} else {
				int i, j;
				double* p_dDest;

				for (i = 0; i < src->height; i++) {
					p_dDest = (double*) cvPtr2D(dest, i, 0);

					for (j = 0; j < src->width; j++) {
						if (p_dDest[j] > .04045)
							p_dDest[j] = pow((p_dDest[j] + .055) / 1.055, 2.4);
						else
							p_dDest[j] /= 12.92;
					}
				}
			}
		} else {
			printf("gamma_decompress: unrecognized gamma type %s, treating as linear\n", type);
			return -1;
		}

		return 0;
	}

	static IplImage* Sub_Image(IplImage *image, CvRect roi) {
		IplImage *result;

		// sub-image
		result = cvCreateImage( cvSize(roi.width, roi.height), image->depth, image->nChannels );
		cvSetImageROI(image,roi);
		cvCopy(image,result,0);
		cvResetImageROI(image); // release image ROI

		return result;
	}

	static CvPoint getAverageCenterPoint(vector <CvRect*>& rects) {
		vector <CvPoint*> points;
		for (int i=0; i<rects.size(); i++) {
			CvRect *thisRect = rects.at(i);
			CvPoint thisPoint = getRectCenterPoint(thisRect);
			points.push_back(&thisPoint);
		}
		return getAveragePoint(points);
	}

	static CvPoint getAveragePoint(vector <CvPoint*> points) {
		int totalX = 0;
		int totalY = 0;
		for (int i=0; i<points.size(); i++) {
			CvPoint *thisPoint = points.at(i);
			totalX = totalX + thisPoint->x;
			totalY = totalY + thisPoint->y;
		}
		CvPoint point = cvPoint (
			cvRound(totalX / (double)points.size()),
			cvRound(totalY / (double)points.size())
		);
		return point;
	}

	static CvRect getEncRect(CvRect *roi, int dst_width, int dst_height) {
			int centerX = 0, centerY=0;
			int encTlRectX = 0;
			int encTlRectY = 0;
			getRectCenter(roi,&centerX,&centerY);
			encTlRectX = cvRound(centerX-(dst_width*0.5));
			encTlRectY = cvRound(centerY-(dst_height*0.5));
			return cvRect(encTlRectX,encTlRectY,dst_width,dst_height);
	}

	static CvRect* getLargestRect(vector <CvRect*>& rects) {
		int largestArea = 0;
		int largestAreaIndex = -1;
		for (int i=0; i<rects.size(); i++) {
			CvRect *thisRect = rects.at(i);
			int thisArea = thisRect->x * thisRect->y;
			if (thisArea > largestArea) {
				largestArea = thisArea;
				largestAreaIndex = i;
			}
		}
		if (largestAreaIndex > -1) {
			return rects.at(largestAreaIndex);
		}
		return 0;
	}


	static void getRectCenter(CvRect *roi, int * centerX, int * centerY) {
		*centerX = cvRound((roi->x + roi->width*0.5));
		*centerY = cvRound((roi->y + roi->height*0.5));
	}

	static CvPoint getRectCenterPoint(CvRect *roi) {
		CvPoint centerPoint = cvPoint(
			cvRound(roi->x + roi->width/2),
			cvRound(roi->y + roi->height/2)
		);
		return centerPoint;
	}

	static CvScalar * getColors(void) {
		static CvScalar colors[] = {
			{0,0,255,0},
			{0,128,255,0},
			{0,255,255,0},
			{0,255,0,0},
			{255,128,0,0},
			{255,255,0,0},
			{255,0,0,0},
			{255,0,255,0}
		};
		return colors;
	}
	/*
	* returns 1 if object_rect contains point
	* returns o if not
	*/
	static int containsPoint(CvRect object_rect, CvPoint2D32f point) {
		if(point.x>object_rect.x) {
			if( point.x < object_rect.x+object_rect.width) {
				if(point.y > object_rect.y) {
					if(point.y < object_rect.y +object_rect.y+object_rect.height) {
						return 1;
					}
				}
			}
		}
		return 0;
	}

	static void defaultContours(void) {

	}

	static void defaultHist(IplImage * theImage, char * histDispName) {
		CvHistogram *hist;
		int hist_size = 200;
		float range_0[]={0,256};
		float* ranges[] = { range_0 };
		IplImage *hist_image = 0, *dst_image = 0, *hgrey = 0;
		uchar lut[256];
		CvMat* lut_mat;
		float max_value = 0;
		int i, bin_w;


		hgrey = cvCreateImage( cvGetSize(theImage), 8, 1 );
		cvCvtColor( theImage, hgrey, CV_BGR2GRAY );
		dst_image = cvCloneImage(hgrey);

		hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);
		hist_image = cvCreateImage(cvSize(400,400), 8, 1);
		lut_mat = cvCreateMatHeader( 1, 256, CV_8UC1 );
		cvSetData( lut_mat, lut, 0 );
		/*{
		int _brightness = 100;
		int _contrast = 100;
		int brightness = _brightness - 100;
		int contrast = _contrast - 100;
		double delta = -128.*contrast/100;
		double a = (256.-delta*2)/255.;
		double b = a*brightness + delta;
		for( i = 0; i < 256; i++ )
		{
		int v = cvRound(a*i + b);
		if( v < 0 )
		v = 0;
		if( v > 255 )
		v = 255;
		lut[i] = (uchar)v;
		}
		}
		cvLUT( hgrey, dst_image, lut_mat );    */

		cvCalcHist( &dst_image, hist, 0, NULL );
		cvZero( dst_image );
		cvGetMinMaxHistValue( hist, 0, &max_value, 0, 0 );
		cvScale( hist->bins, hist->bins, ((double)hist_image->height)/max_value, 0 );
		//cvNormalizeHist( hist, 1000 );

		cvSet( hist_image, cvScalarAll(255), 0 );
		bin_w = cvRound((double)hist_image->width/hist_size);

		for( i = 0; i < hist_size; i++ ){
			cvRectangle( hist_image, cvPoint(i*bin_w, hist_image->height),
				cvPoint((i+1)*bin_w, hist_image->height - cvRound(cvGetReal1D(hist->bins,i))),
				cvScalarAll(0), -1, 8, 0 );

		}

		cvShowImage( histDispName, hist_image );
		cvReleaseImage(&hgrey);
		cvReleaseImage(&hist_image);
		cvReleaseHist(&hist);
	}


	void cvu_PrintMatrix(CvMat *M) {
		int i, j;

		printf("Matrix is %d x %d \n", M->rows, M->cols);

		for (i = 0; i < M->rows; ++i) {
			for (j = 0; j < M->cols; ++j)
				printf("%f  ", cvmGet(M, i, j));

			printf("\n");
		}
		printf("\n");;
	}

	static void rotateWithQuadrangle(IplImage *src, IplImage *dst, float angle, CvPoint *center) {
		// Compute rotation matrix
		/*
		CvMat *rot_mat = cvCreateMat(2,3,CV_32FC1);
		CvPoint2D32f center32 = cvPointTo32f(*center);
		cv2DRotationMatrix( center32, angle, 1, rot_mat );
		// Do the transformation
		//cvWarpAffine( src, dst, rot_mat );
		*/

		/****************
		** using cvGetQuadrangleSubPix rotation for speed and so we can get edge distortion
		*****************/

		float m[6];
		CvMat M = cvMat(2, 3, CV_32F, m);
		int w = src->width;
		int h = src->height;

		m[0] = (float)(cos(-angle*2*CV_PI/180.));
		m[1] = (float)(sin(-angle*2*CV_PI/180.));
		m[3] = -m[1];
		m[4] = m[0];

		//m[2] = w*0.5f;
		//m[5] = h*0.5f;

		m[2] = center->x;
		m[5] = center->y;
		cvGetQuadrangleSubPix( src, dst, &M);

		// cvGetQuadrangleSubPix( src, dst, rot_mat);

		/*
		cvNamedWindow("rotation", 1);
		cvShowImage("rotation", dst);
		*/
		//angle =(int)(angle + delta) % 360;
	}


};


#endif
