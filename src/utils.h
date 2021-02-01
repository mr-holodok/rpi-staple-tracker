#ifndef STAPLE_UTILS_H
#define STAPLE_UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <fstream>

namespace Utils {

    template <typename SingleChannelElementT>
    bool matIsEqual(const cv::Mat& mat1, const cv::Mat& mat2, bool verbose = false) {
        if (mat1.empty() && mat2.empty()) {
            if (verbose)
                std::cout << "Warning: (matIsEqual) matrixes are empty!" << std::endl;
            return true;
        }

        // if dimensionality of two mat is not identical, these two mat is not identical
        if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims) {
            if (verbose) {
                std::cout << "(matIsEqual) matrixes are different in sizes! Rows: " <<
                        mat1.rows << " , " << mat2.rows <<
                        " Cols: " << mat1.cols << " , " << mat2.cols <<
                        " Dims: " << mat1.dims << " , " << mat2.dims << std::endl;
            }
            return false;
        }

        if (mat1.channels() > 1) {
            std::vector<cv::Mat> mat1_channels, mat2_channels;
            cv::split(mat1, mat1_channels);
            cv::split(mat2, mat2_channels);

            int sum = 0;
            for (int i = 0; i < mat1_channels.size(); ++i) {
                cv::Mat diff;
                // comparing 2 mat and result is written to diff,
                // if elements comparison is true (as cmpop parameter states) then in diff in according place will be 255, else - 0
                cv::compare(mat1_channels[i], mat2_channels[i], diff, cv::CMP_NE);
                sum += countNonZero(diff);

                if (verbose && countNonZero(diff) > 0) {
                    std::cout << "channel " << i << std::endl;
                    for (int x = 0; x < diff.rows; ++x)
                        for (int y = 0; y < diff.cols; ++y)
                            if (diff.at<char>(x, y) != (char) 0)
                                std::cout << "[" <<  x << "][" << y << "] " << mat1_channels[i].at<SingleChannelElementT>(x, y) <<
                                        " != " << mat2_channels[i].at<SingleChannelElementT>(x, y) << std::endl;
                }
            }

            return sum == 0;
        }
        else {
            cv::Mat diff;
            // comparing 2 mat and result is written to diff,
            // if elements comparison is true (as cmpop parameter states) then in diff in according place will be 255, else - 0
            cv::compare(mat1, mat2, diff, cv::CMP_NE);
            int nz = cv::countNonZero(diff);
            return nz == 0;
        }
    }

    cv::Rect_<float> getAxisAlignedBB(std::vector<cv::Point2f> polygon) {
        double cx = double(polygon[0].x + polygon[1].x + polygon[2].x + polygon[3].x) / 4.;
        double cy = double(polygon[0].y + polygon[1].y + polygon[2].y + polygon[3].y) / 4.;
        double x1 = std::min(std::min(std::min(polygon[0].x, polygon[1].x), polygon[2].x), polygon[3].x);
        double x2 = std::max(std::max(std::max(polygon[0].x, polygon[1].x), polygon[2].x), polygon[3].x);
        double y1 = std::min(std::min(std::min(polygon[0].y, polygon[1].y), polygon[2].y), polygon[3].y);
        double y2 = std::max(std::max(std::max(polygon[0].y, polygon[1].y), polygon[2].y), polygon[3].y);
        double A1 = norm(polygon[1] - polygon[2]) * norm(polygon[2] - polygon[3]);
        double A2 = (x2 - x1) * (y2 - y1);
        double s = sqrt(A1 / A2);
        double w = s * (x2 - x1) + 1;
        double h = s * (y2 - y1) + 1;
        cv::Rect_<float> rect(cx - 1 - w / 2.0, cy - 1 - h / 2.0, w, h);
        return rect;
    }

    std::vector<cv::Rect_<float>> getgroundtruth(std::string txt_file) {
        std::vector<cv::Rect_<float>> rects;
        std::ifstream gt;
        gt.open(txt_file.c_str());
        if (!gt.is_open())
            std::cout << "Ground truth file " << txt_file
                      << " can not be read" << std::endl;
        std::string line;
        float x1, y1, x2, y2, x3, y3, x4, y4;
        while (getline(gt, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::stringstream ss;
            ss.str(line);
            ss >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4;
            std::vector<cv::Point2f> polygon;
            polygon.push_back(cv::Point2f(x1, y1));
            polygon.push_back(cv::Point2f(x2, y2));
            polygon.push_back(cv::Point2f(x3, y3));
            polygon.push_back(cv::Point2f(x4, y4));
            rects.push_back(getAxisAlignedBB(polygon));//0-index
        }
        gt.close();
        return rects;
    }

}

#endif//STAPLE_UTILS_H
