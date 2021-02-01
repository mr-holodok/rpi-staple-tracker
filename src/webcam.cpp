#include <chrono>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
// for openCV verbose logging
//#include <opencv2/core/utils/logger.hpp>

#include "tracker.hpp"

// var for storing left mouse btn state
static bool isLBtnDown = false;
// points that hold coords of selection rect
static cv::Point selectionRectPoint1(0, 0), selectionRectPoint2(0, 0);

static bool isTracked = false;
static cv::Mat image;
static StapleTracker staple;

cv::Rect calcRectByPoints(cv::Point const &p1, cv::Point const &p2) {
    bool x_inc = p1.x < p2.x;
    bool y_inc = p1.y < p2.y;
    // ?_inc means increasing order of coords

    if (x_inc && y_inc)
        return cv::Rect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
    else if (!x_inc && !y_inc)
        return cv::Rect(p2.x, p2.y, p1.x - p2.x, p1.y - p2.y);
    else if (!x_inc && y_inc)
        return cv::Rect(p2.x, p1.y, p1.x - p2.x, p2.y - p1.y);
    else
        return cv::Rect(p1.x, p2.y, p2.x - p1.x, p1.y - p2.y);
}

void mouseCallBack(int event, int x, int y, int flags, void *userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        isTracked = false;
        isLBtnDown = true;
        selectionRectPoint1.x = x;
        selectionRectPoint1.y = y;
        selectionRectPoint2.x = x;
        selectionRectPoint2.y = y;
    } else if (event == cv::EVENT_LBUTTONUP) {
        isLBtnDown = false;

        staple.trackerInit(image, calcRectByPoints(selectionRectPoint1, selectionRectPoint2));
        isTracked = true;
    } else if (event == cv::EVENT_MOUSEMOVE && isLBtnDown) {
        selectionRectPoint2.x = x;
        selectionRectPoint2.y = y;
    }
}


int webcam_main() {
    //cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_VERBOSE);

    bool stop_flag = false;

    std::cout << "Opening camera..." << std::endl;
    cv::VideoCapture capture(0);// open the first camera

    if (!capture.isOpened()) {
        std::cerr << "ERROR: Can't initialize camera capture" << std::endl;
        return 1;
    }

    if (capture.set(cv::CAP_PROP_FPS, 20)) {
        std::cout << "[log]: set target FPS" << std::endl;
    } else {
        std::cerr << "[log]: failed to set target FPS!" << std::endl;
    }

    if (capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280)) {
        std::cout << "[log]: set target width" << std::endl;
    } else {
        std::cerr << "[log]: failed to set target width!" << std::endl;
    }

    if (capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720)) {
        std::cout << "[log]: set target height" << std::endl;
    } else {
        std::cerr << "[log]: failed to set target height!" << std::endl;
    }

    std::cout << "width: " << capture.get(cv::CAP_PROP_FRAME_WIDTH) << " "
              << "height: " << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << " "
              << "fps: " << capture.get(cv::CAP_PROP_FPS) << std::endl;


    std::cout << std::endl
              << "Press 'ESC' to quit, 'space' to toggle image processing" << std::endl;
    std::cout << std::endl
              << "Start grabbing..." << std::endl;

    size_t nFrames = 0;

    cv::namedWindow("STAPLE", cv::WINDOW_NORMAL);
    cv::setMouseCallback("STAPLE", mouseCallBack, nullptr);

    cv::Rect_<float> location;

    long update_duration = 0l;

    while (!stop_flag) {

        capture >> image;
        if (image.empty()) {
            std::cerr << "ERROR: Can't grab camera frame." << std::endl;
            break;
        }

        if (isTracked) {

            auto start = std::chrono::high_resolution_clock::now();
            location = staple.getNextPos(image);
            auto stop = std::chrono::high_resolution_clock::now();
            update_duration += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

            cv::rectangle(image, location, cv::Scalar(0, 0, 255), 2);

            if (nFrames % 10 == 0) {
                std::cout << "Update: " << update_duration / 10 <<  std::endl;
                update_duration = 0l;
            }
        } else if (isLBtnDown) {
            // else we are drawing selection rect
            rectangle(image, selectionRectPoint1, selectionRectPoint2, cv::Scalar(255, 0, 0), 2);
        }

        nFrames++;
        //		if (nFrames % 10 == 0)
        //		{
        //			const int N = 10;
        //			int64 t1 = cv::getTickCount();
        //			std::cout << "Frames captured: " << cv::format("%5lld", (long long int)nFrames)
        //				<< "    Average FPS: " << cv::format("%9.1f", (double)cv::getTickFrequency() * N / (t1 - t0))
        //				<< "    Average time per frame: " << cv::format("%9.2f ms", (double)(t1 - t0) * 1000.0f / (N * cv::getTickFrequency()))
        //				<< std::endl;
        //			t0 = t1;
        //		}

        imshow("STAPLE", image);

        char key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            stop_flag = true;
        } else if (key == 99) /*c = cancel*/
        {
            isTracked = false;
        }
    }

    cv::destroyAllWindows();

    return 0;
}