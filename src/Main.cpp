#include <iostream>

#include <dirent.h>

#include "cv.h"
#include "highgui.h"

#include "AnyOption.hpp"
#include "CvUtils.hpp"
#include "EyeDetector.hpp"

using namespace std;
using namespace std;

// this is where in the final face image we would like to have the right and left eyes be. x, y
const double canonicalLeftEye[]  = { 17, 21 };
const double canonicalRightEye[] = { 65, 21 };

const int targetImageWidth = 100;
const int targetImageHeight = 100;

char* opt_file = 0;
char* opt_cascade_face = 0;
char* opt_cascade_eye_right = 0;
char* opt_cascade_eye_left = 0;
char* opt_output_path = 0;
char* opt_input_path = 0;

EyeDetector *eyeDetector = 0;

bool opt_show_ui = false;
bool opt_draw_features = false;
bool opt_debug = false;
bool opt_rotate = false;
bool opt_zoom = false;

CvScalar *colors = CvUtils::getColors();
AnyOption *opt = 0;

static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;

void processImage(char * imageFile);
void processFace(IplImage* detectedFaceImg);
int loadOptions(int argc, char *argv[]);
void printVersion();
void printUsage();

int main(int argc, char** argv) {

	// load the command line options
	if (!loadOptions(argc, argv)) {
		return 0;
	}

	// initialize the face detect cascade
	CvCapture* capture = 0;
	if (opt_cascade_face) {
		cascade = (CvHaarClassifierCascade*) cvLoad(opt_cascade_face, 0, 0, 0);
	}

	// make sure a face detect cascade was successfully loaded
	if (cascade) {
		if (opt_debug) {
			printf("Face detection cascade loaded\n");
		}
	} else {
		fprintf(stderr, "ERROR: unable to load face cascade: %s\n", opt_cascade_face);
		return -1;
	}

	// load the eye detector cascades
	eyeDetector = new EyeDetector();
	eyeDetector->setDebug(opt_debug);
	eyeDetector->setShowUi(opt_show_ui);

	if (opt_cascade_eye_left) {
		eyeDetector->setLeftEyeCascade(opt_cascade_eye_left);
	}

	if (opt_cascade_eye_right) {
		eyeDetector->setRightEyeCascade(opt_cascade_eye_right);
	}

	storage = cvCreateMemStorage(0);

	if (opt_file) {
		processImage(opt_file);
	}

	if (opt_input_path) {

	    DIR *dp;
	    struct dirent *dirp;
	    if ((dp  = opendir(opt_input_path)) == NULL) {
	    	fprintf(stderr, "Unable to open directory: %s\n", opt_input_path);
	    } else {
	    	// loop over all files in the input directory and look for images
			while ((dirp = readdir(dp)) != NULL) {
				char *fileName = dirp->d_name;

				// I know. probably not the best way to look for images, but whatevz.
				if (strcmp(".bmp", fileName+strlen(fileName)-4)==0 ||
					strcmp(".BMP", fileName+strlen(fileName)-4)==0 ||
					strcmp(".png", fileName+strlen(fileName)-4)==0 ||
					strcmp(".PNG", fileName+strlen(fileName)-4)==0 ||
					strcmp(".jpg", fileName+strlen(fileName)-4)==0 ||
					strcmp(".JPG", fileName+strlen(fileName)-4)==0 ) {

					char *fileUri = new char[strlen(opt_input_path) + strlen(dirp->d_name) + 1];
					sprintf(fileUri, "%s%s", opt_input_path, dirp->d_name);

					processImage(fileUri);
				}
			}
			closedir(dp);
	    }
	}

	cvDestroyWindow("result");
	cvDestroyWindow("histogram");
	return 0;
}

void processImage(char * imageFile) {

	if (opt_debug) {
		printf("\n\nLoading Image:%s\n", imageFile);
	}

	IplImage* img = cvLoadImage(imageFile, 1);
	if (!img) {
		fprintf(stderr, "ERROR: Loading image Failed: %s\n", imageFile);
		return;
	}

	// separate just the file name so we can use it to output files.
	char * imageFileName;
	std::string imageFileString = imageFile;
	if (imageFileString.find_last_of("/") != std::string::npos) {
		std::string imageFileNameString = imageFileString.substr(imageFileString.find_last_of("/") + 1);

		imageFileName = new char [imageFileNameString.size()+1];
		strcpy (imageFileName, imageFileNameString.c_str());
	} else {
		imageFileName = imageFile;
	}

	if (opt_debug) {
		printf("Image has dimensions: w%d h%d\n", img->width, img->height);
	}

	double scale = 1;
	if (img->width > 450 || img->width < 200) {
		scale = 450 / (double) img->width;
	}
	if (img->width > 450) {
		scale = 450 / (double) img->width;
	}
	if (opt_debug) {
		printf("Using a scale of: %g\n", scale);
	}

	// how much should we scale the markup image
	double scaleMarkup = scale;

	//cvCopy(img, imgMarkup);

	//IplImage* gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );
	IplImage* small_img = cvCreateImage(cvSize(cvRound(img->width * scale), cvRound(img->height * scale)), img->depth,
			img->nChannels);
	IplImage* small_img_gray = cvCreateImage(cvSize(small_img->width, small_img->height), 8, 1);

	// imgMarkup is the image that we will draw boxes and crosshairs on for visualization
	IplImage* imgMarkup = cvCreateImage(cvSize(cvRound(img->width * scaleMarkup), cvRound(img->height * scaleMarkup)),
			img->depth, img->nChannels);
	cvResize(img, imgMarkup, CV_INTER_LINEAR);

	IplImage* imgOperation = cvCreateImage(cvSize(cvRound(img->width * scaleMarkup), cvRound(img->height * scaleMarkup)),
			img->depth, img->nChannels);
	cvResize(img, imgOperation, CV_INTER_LINEAR);

	int i;

	cvCvtColor(small_img, small_img_gray, CV_BGR2GRAY);
	cvResize(img, small_img, CV_INTER_LINEAR);
	//cvEqualizeHist( small_img, small_img );
	cvClearMemStorage(storage);

	// set up timers
	double ticks = (double) cvGetTickCount();

	// Look for faces in the photo
	CvSeq* faces = cvHaarDetectObjects(small_img, cascade, storage, 1.1, 5, 0/*CV_HAAR_DO_CANNY_PRUNING*/,
			cvSize(30, 30));
	ticks = (double) cvGetTickCount() - ticks;

	// loop over all of the faces found in the image.
	for (i = 0; i < (faces ? faces->total : 0); i++) {

		CvRect* faceRect = (CvRect*) cvGetSeqElem(faces, i);
		CvRect* operationRect = (CvRect*) cvGetSeqElem(faces, i);
		CvPoint center;

		//int radius;
		center.x = cvRound((faceRect->x + faceRect->width * 0.5));
		center.y = cvRound((faceRect->y + faceRect->height * 0.5));
		//radius = cvRound((faceRect->width + faceRect->height)*0.25);

		// we want to expand the detected face a little bit so we have padding on the rotation
		double expandPercentage = 10;
		double expandX = (faceRect->width * (100 + expandPercentage) / 100) - faceRect->width;
		double expandY = (faceRect->height * (100 + expandPercentage) / 100) - faceRect->height;

		faceRect->x = faceRect->x - expandX/2;
		faceRect->y = faceRect->y - expandY/2;
		faceRect->width = faceRect->width + expandX;
		faceRect->height = faceRect->height + expandY;

		if (opt_debug) {
			printf("face found\n");
			printf("\tx: %d\n", faceRect->x);
			printf("\ty: %d\n", faceRect->y);
			printf("\twidth: %d\n", faceRect->width);
			printf("\theight: %d\n", faceRect->height);
		}

		IplImage *faceBlock = CvUtils::Sub_Image(
				small_img,
				cvRect(faceRect->x, /* x = start from leftmost */
						faceRect->y, /* y = a few pixels from the top */
						faceRect->width, /* width = same width with the face */
						faceRect->height /* height = 1/3 of face height */
				)
			);

		// draw a box around the face
		CvUtils::drawRect(*faceRect, imgMarkup, colors[i % 8]);

		CvPoint leftEye = cvPoint(0, 0);
		CvPoint rightEye = cvPoint(0, 0);
		CvPoint leftEyeAbsolute = cvPoint(0, 0);
		CvPoint rightEyeAbsolute = cvPoint(0, 0);

		// look for eyes in the photo
		//cvClearMemStorage( storage );
		eyeDetector->find(faceBlock, &leftEye, &rightEye);
		leftEyeAbsolute.x = (faceRect->x + leftEye.x);
		leftEyeAbsolute.y = (faceRect->y + leftEye.y);
		rightEyeAbsolute.x = (faceRect->x + rightEye.x);
		rightEyeAbsolute.y = (faceRect->y + rightEye.y);

		if (opt_debug) {
			printf("\tleft eye x: %d, y: %d\n", leftEye.x, leftEye.y);
			printf("\tleft eye x: %d, y: %d\n", rightEye.x, rightEye.y);
		}

		// draw crosshairs where we think the eyes are
		CvUtils::drawCrosshair(&leftEyeAbsolute, imgMarkup, 255, 0, 0);
		CvUtils::drawCrosshair(&rightEyeAbsolute, imgMarkup, 0, 0, 255);


		// write the face as it was found in the image
		if (opt_output_path) {
			char *destImageUri = new char[strlen(opt_output_path) + strlen("face_") + strlen(imageFileName) + 1];
			sprintf(destImageUri, "%sface_%s", opt_output_path, imageFileName);
			if (opt_debug) {
				printf("Saving image to %s\n", destImageUri);
			}
			cvSaveImage(destImageUri, faceBlock);
			delete[] destImageUri;
		}

		if (opt_rotate) {
			float distanceX = (float) (leftEye.y) - (float) rightEye.y;
			float distanceY = (float) (leftEye.x) - (float) rightEye.x;
			float angle = atan2(distanceX, distanceY) * 180 / CV_PI;

			cout << "ANGLE:" << angle << "\n";

			//angle = angle + ((180-angle)/2)-180;
			//angle = angle -180;
			angle = angle + ((180 - angle) / 2);

			//angle = angle;
			cout << "ANGLE2:" << angle << "\n";
			printf("Calculated Rotation: %f degrees\n", angle);

			if (angle < 15 || (angle > 165 && angle < 180)) {
				imgOperation = cvCreateImage(cvSize(small_img->width, small_img->height), small_img->depth,
						small_img->nChannels);
				CvPoint centerRotatePoint = cvPoint(faceRect->x + (faceRect->width / 2),
						faceRect->y + (faceRect->height / 2));
				CvUtils::drawCrosshair(&centerRotatePoint, imgMarkup, 255, 0, 0);
				//CvPoint centerRotatePoint = cvPoint((leftEyeAvg.x + rightEyeAvg.x) / 2, (leftEyeAvg.y + rightEyeAvg.y) /2);
				//CvPoint centerRotatePoint = cvPoint(leftEyeAvg.x, leftEyeAvg.y);
				CvUtils::rotateWithQuadrangle(small_img, imgOperation, angle, &centerRotatePoint);
				//cvCopy( rotatedImage, small_img);
				//cvCopy( rotatedImage, imgMarkup);
				//cvCvtColor( small_img, small_img_gray, CV_BGR2GRAY );
				//cvReleaseImage( &rotatedImage );

				if (opt_show_ui) {
					cvNamedWindow("rotated", 1);
					cvShowImage("rotated", imgOperation);
					cvResizeWindow("rotated", imgOperation->width, imgOperation->height);
				}

				operationRect->x = imgOperation->width / 2 - faceRect->width / 2;	/* x = start from leftmost */
				operationRect->y = imgOperation->height / 2 - faceRect->height / 2; /* y = a few pixels from the top */
				operationRect->width = faceRect->width;	/* width = same width with the face */
				operationRect->height = faceRect->height; /* height = 1/3 of face height */

				cvReleaseImage(&faceBlock);
				faceBlock = CvUtils::Sub_Image(imgOperation, *operationRect);

				// write the rotated image
				if (opt_output_path) {
					char *destImageUri = new char[strlen(opt_output_path) + strlen("rotated_") + strlen(imageFileName) + 1];
					sprintf(destImageUri, "%srotated_%s", opt_output_path, imageFileName);
					if (opt_debug) {
						printf("Saving image to %s\n", destImageUri);
					}
					cvSaveImage(destImageUri, faceBlock);
					delete[] destImageUri;
				}

			}
		}

		// TODO: this block was pasted in from another project, it isn't currently working.

		double targetScale = double(faceRect->height)/double(targetImageHeight);
		if (opt_zoom) {
			double zoomScale = 1;
			// scale the train rectangle to the size it will need to be to align the eyes.
			if (leftEye.x > 0 && rightEye.x > 0) {

				// see if we need to resize the height/width of the face block
				int targetLeftEyeX = int(canonicalLeftEye[0] * targetScale);
				int targetRightEyeX = int(canonicalRightEye[0] * targetScale);
				int targetXWidth = targetRightEyeX - targetLeftEyeX;
				int actualXWidth = rightEye.x - leftEye.x;

				if (actualXWidth > 0) {
					zoomScale = (double)actualXWidth / (double)targetXWidth;


					printf("face zoom scale:%f\n", zoomScale);

					int newHeight = int(faceRect->height * zoomScale);
					int newWidth = int(faceRect->width * zoomScale);
					int moveX = int((newWidth - faceRect->width)/2);
					int moveY = int((newHeight - faceRect->height)/2);

					//printf("\tnewHeight:%d\n\tnewWidth:%d\n\tnewX:%d\n", newHeight, newWidth, moveX);

					leftEye.x = leftEye.x + moveX;
					rightEye.x = rightEye.x + moveX;
					leftEye.y = leftEye.y + moveY;
					rightEye.y = rightEye.y + moveY;

					faceRect->x = faceRect->x - moveX;
					faceRect->y = faceRect->y - moveY;

					faceRect->width = newWidth;
					faceRect->height = newHeight;
	//					printf ("FACEBLOCK\n\tx:%d\ty:%d\n\twidth:%d\n\theight:%d\n", trainRect->x, trainRect->y, trainRect->width, trainRect->height);
	//					printf("EYES\n\tleft:%d,%d\n\tright:%d,%d\n\n", leftEye.x, leftEye.y, rightEye.x, rightEye.y);
				}

				// balance the eye position
				int space = faceRect->width - (rightEye.x - leftEye.x);
				int moveX = space/2 - leftEye.x;
				faceRect->x = faceRect->x - moveX;
				leftEye.x = leftEye.x + moveX;
				rightEye.x = rightEye.x + moveX;

				// move the box so the eyes are at the right height
				int targetEyeY = int(canonicalLeftEye[1] * targetScale * zoomScale);
				int moveY = targetEyeY - leftEye.y;
				//printf("MOVING targetEyeY:%d, leftEyeY:%d, moveY:%d\n", targetEyeY, leftEye.y, moveY);
				//cout << "\nMOVING Y TO:" << trainRect->y << " : " << moveY << "\n";
				faceRect->y = faceRect->y - moveY;
				leftEye.y = leftEye.y + moveY;
				rightEye.y = rightEye.y + moveY;

	//				printf ("FACEBLOCK\n\tx:%d\ty:%d\n\twidth:%d\n\theight:%d\n", trainRect->x, trainRect->y, trainRect->width, trainRect->height);
	//				printf("EYES\n\tleft:%d,%d\n\tright:%d,%d\n\n", leftEye.x, leftEye.y, rightEye.x, rightEye.y);

			}

			IplImage *faceTrain = cvCreateImage(cvSize(targetImageWidth, targetImageHeight), imgOperation->depth, imgOperation->nChannels);

			cvResetImageROI(imgOperation);
			cvSetImageROI(imgOperation, *operationRect);

			//sprintf(file_num,îG:\\faces_from_movie\\%d.jpgî,int(framenum));
			//cvReleaseImage(faceimg);
			//faceimg = cvCloneImage(img);
			cvResize(imgOperation, faceTrain, CV_INTER_LINEAR);

			if (opt_show_ui) {
				cvNamedWindow("scaled", 1);
				cvShowImage("scaled", faceTrain);
				cvResizeWindow("scaled", faceTrain->width, faceTrain->height);
			}

			// write the rotated image
			if (opt_output_path) {
				char *destImageUri = new char[strlen(opt_output_path) + strlen("scaled_") + strlen(imageFileName) + 1];
				sprintf(destImageUri, "%scaled_%s", opt_output_path, imageFileName);
				if (opt_debug) {
					printf("Saving scaled image to %s\n", destImageUri);
				}
				cvSaveImage(destImageUri, faceTrain);
				delete[] destImageUri;
			}

			cvReleaseImage(&faceTrain);

			CvPoint leftEyeTrained = cvPoint(
					int(leftEye.x / targetScale / zoomScale),
					int(leftEye.y  / targetScale / zoomScale)
			);
			CvPoint rightEyeTrained = cvPoint(
					int(rightEye.x  / targetScale / zoomScale),
					int(rightEye.y  / targetScale / zoomScale)
			);


//			CvUtils::drawCrosshair(&leftEyeActual, faceTrain, 255, 0, 0);
//			CvUtils::drawCrosshair(&rightEyeActual, faceTrain, 0, 0, 255);
//			cvNamedWindow("TRAIN");
//			cvShowImage("TRAIN", faceTrain);
//			cvResizeWindow("TRAIN", faceTrain->width, faceTrain->height);


		}


		if (opt_show_ui) {
			cvNamedWindow("face", 1);
			cvShowImage("face", faceBlock);
			//cvResizeWindow("face", faceRect->width, faceRect->height);
		}

		// write the marked up image
		if (opt_output_path) {
			char *destImageUri = new char[strlen(opt_output_path) + strlen("markup_") + strlen(imageFileName) + 1];
			sprintf(destImageUri, "%smarkup_%s", opt_output_path, imageFileName);
			if (opt_debug) {
				printf("Saving markup image to %s\n", destImageUri);
			}
			cvSaveImage(destImageUri, imgMarkup);
			delete[] destImageUri;
		}

		// display the marked up image
		if (opt_show_ui) {
			cvNamedWindow("markup", 1);
			cvShowImage("markup", imgMarkup);
			cvResizeWindow("markup", imgMarkup->width, imgMarkup->height);

			// wait for the user to press a key.
			cvWaitKey(0);
		}

		//try to make sure that we keep tracking the same face
		/*if(faces->total>2) {
		 if(!isTrackFace(width, height,&topLeft)){
		 continue;
		 }
		 }*/

		//cvSetImageROI(img,faceRect);
		/*printf("face x=%d y=%d w=%d h=%d\n",
		 topLeft.x,
		 topLeft.y,
		 width,
		 height);*/
		//crop image to face
		//printf("Entering Crop...");
		//cropped = Sub_Image(img,faceRect);
		//printf("Done\n");
		//cvShowImage("cropped",img);
		//doFacialFeatureTrack(img,"result", topLeft.x,topLeft.y,width, height);
		//featureTracker->doFacialFeatureTracking(img,topLeft.x,topLeft.y,width, height);
		//cvReleaseImage(&cropped);
		//break;
	}
	/*
	 if(faces->total==0) {
	 featureTracker->doTrackingInit = 1;
	 }
	 tt = (double)cvGetTickCount() - tt;
	 */
	//doHist(img);
	//printf( "round trip time = %gms\n", tt/((double)cvGetTickFrequency()*1000.) );
	//cvShowImage( "result", img );
	//cvReleaseImage( &gray );
	cvReleaseImage(&imgMarkup);
	cvReleaseImage(&imgOperation);
	cvReleaseImage(&small_img);
	cvReleaseImage(&small_img_gray);

	cout << "RESULT:";
	cout << "\tfile:" << imageFile;
	cout << "\tfaces:" << (faces ? faces->total : 0);
	cout << "\n";

	cvReleaseImage(&img);
}

void processFace(IplImage* detectedFaceImg) {


}


void printVersion() {
	// Display if the IPP library got loaded into opencv.
	const char *opencvLibraries = 0;
	const char *addonModules = 0;
	cvGetModuleInfo(0, &opencvLibraries, &addonModules);
	cout << "*******************************\n";
	cout << "* Wirelust opencv face position\n";
	cout << "* OpenCV: " << opencvLibraries << "\n";
	cout << "* Add-on Modules: " << addonModules << "\n";
	cout << "* http://www.wirelust.com\n";
	cout << "*******************************\n";
}

void printUsage() {
	if (opt) {
		opt->printUsage();
	}
}

int loadOptions(int argc, char *argv[]) {
	/* 1. CREATE AnyOption OBJECT */
	opt = new AnyOption(20);
	//opt->setVerbose();

	vector<string*> errors;

	opt->addUsage("");

	opt->addUsage("Flags: ");
	opt->addUsage(" -h  --help           Prints this help ");
	opt->addUsage(" -v  --version        Print version ");
	opt->addUsage(" -D  --draw-features  Draw the facial feature regions");
	opt->addUsage(" -u  --show-ui        Show UI");
	opt->addUsage(" -d  --debug          Debug");
	opt->addUsage(" -R  --rotate         Rotate face based on eye detection");
	opt->addUsage(" -z  --zoom           Zoom face based on eye detection");

	opt->addUsage("");
	opt->addUsage("Options: ");
	opt->addUsage(" -c  --cascade            The face detector cascade to use");
	opt->addUsage(" -l  --left-eye-cascade   The Cascade for the left eye");
	opt->addUsage(" -r  --right-eye-cascade  The Cascade for the right eye");
	opt->addUsage(" -f  --file               Input image file to use");
	opt->addUsage(" -i  --input-path         Path of directory to read for images. must include trailing slash");
	opt->addUsage(" -o  --output-path        Path to output altered images. must include trailing slash");

	opt->setFlag("help", 'h');
	opt->setFlag("version", 'v');
	opt->setFlag("draw-features", 'D');
	opt->setFlag("show-ui", 'u');
	opt->setFlag("debug", 'd');
	opt->setFlag("rotate", 'R');
	opt->setFlag("zoom", 'z');

	opt->setOption("cascade", 'c');
	opt->setOption("left-eye-cascade", 'l');
	opt->setOption("right-eye-cascade", 'r');
	opt->setOption("file", 'f');
	opt->setOption("input-path", 'i');
	opt->setOption("output-path", 'o');

	// process the options in order of importance with each overwriting the other
	opt->processFile("configuration.conf");
	opt->processCommandArgs(argc, argv);

	if (opt->getFlag("help") || opt->getFlag('h')) {
		printVersion();
		opt->printUsage();
		delete opt;
		return 0;
	}

	if (opt->getFlag("version") || opt->getFlag('v')) {
		printVersion();
		delete opt;
		return 0;
	}

	if (opt->getFlag("draw-features") || opt->getFlag('D')) {
		opt_draw_features = true;
	}

	if (opt->getFlag("show-ui") || opt->getFlag('u')) {
		opt_show_ui = true;
	}

	if (opt->getFlag("debug") || opt->getFlag('d')) {
		opt_debug = true;
	}

	if (opt->getFlag('R') || opt->getFlag("rotate")) {
		opt_rotate = true;
	}

	if (opt->getFlag('z') || opt->getFlag("zoom")) {
		opt_zoom = true;
	}

	if (opt->getValue('c') != NULL || opt->getValue("cascade") != NULL) {
		opt_cascade_face = opt->getValue('c');
	}

	if (opt->getValue('l') != NULL || opt->getValue("left-eye-cascade") != NULL) {
		opt_cascade_eye_left = opt->getValue('l');
	}

	if (opt->getValue('r') != NULL || opt->getValue("right-eye-cascade") != NULL) {
		opt_cascade_eye_right = opt->getValue('r');
	}

	if (opt->getValue('f') != NULL || opt->getValue("file") != NULL) {
		opt_file = opt->getValue('f');
	}

	if (opt->getValue('i') != NULL || opt->getValue("input-path") != NULL) {
		opt_input_path = opt->getValue('i');
	}

	if (opt->getValue('o') != NULL || opt->getValue("output-path") != NULL) {
		opt_output_path = opt->getValue('o');
	}

	if (opt_debug) {
		printf("---------------- Settings ----------------\n");
		printf("cascade:%s\n", opt_cascade_face);
		printf("right-eye-cascade: %s\n", opt_cascade_eye_right);
		printf("left-eye-cascade: %s\n", opt_cascade_eye_left);
		printf("\n");
		printf("file: %s\n", opt_file);
		printf("show-ui: %s\n", (opt_show_ui) ? "true" : "false");
		printf("rotate: %s\n", (opt_rotate) ? "true" : "false");
		printf("------------------------------------------\n\n");
	}

	int validated = 1;
	if (errors.size() > 0) {
		printVersion();
		cout << "ERROR:\n";
		for (unsigned int i = 0; i < errors.size(); i++) {
			//string thisError = &errors.at(i);
			cout << "\t" << *errors.at(i) << "\n";

			delete errors.at(i);
		}
		validated = 0;
	}

	delete opt;

	return validated;
}

