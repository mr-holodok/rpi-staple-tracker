#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "staple_dummy.h"
#include <iostream>
#include "../src/utils.h"
#include <array>

static cv::MatND xt_windowed;
static std::vector<cv::Mat> xtsplit1{28}, xtsplit2{28}, xtsplit3{28};


void splitMatND(const cv::MatND &xt, std::vector<cv::Mat> &xtsplit) {
    int cn = xt.channels();

    assert(cn == 28);
    assert(xtsplit.size() == 28);

    for (int k = 0; k < cn; k++)
    {
        typedef cv::Vec<float, 28> Vec28f;

        xtsplit[k].forEach<cv::Vec2f>
                (
                        [&xt, k](cv::Vec2f &pair, const int * pos) {
                          pair[0] = xt.at<Vec28f>(pos)[k];
                          pair[1] = 0.0f;
                        }
                );
    }
}


void splitMatND_impr(const cv::MatND &xt, std::vector<cv::Mat> &xtsplit) {
    int w = xt.cols;
    int h = xt.rows;
    int cn = xt.channels();

    assert(cn == 28);
    assert(xtsplit.size() == 28);

    for (int k = 0; k < cn; k++)
    {
        for (int j = 0; j < h; ++j) {
            float *pDst = xtsplit[k].ptr<float>(j);
            const float *pSrc = xt.ptr<float>(j);

            for (int i = 0; i < w; ++i) {
                pDst[0] = pSrc[k];
                pDst[1] = 0.0f;

                pSrc += cn;
                pDst += 2;
            }
        }
    }
}

void splitMatND_cache(const cv::MatND &xt, std::vector<cv::Mat> &xtsplit) {
    int w = xt.cols;
    int h = xt.rows;
    int cn = xt.channels();

//    assert(cn == 28);
//    assert(xtsplit.size() == 28);

    const float *pSrc = xt.ptr<float>(0);

    for (int j = 0; j < h; ++j) {

        for (int i = 0; i < w; ++i) {

            for (int k = 0; k < cn; ++k) {

                cv::Vec2f *pDst = xtsplit[k].ptr<cv::Vec2f>(j, i);

                pDst[0] = *pSrc;
                pDst[1] = 0.0f;

                ++pSrc;
            }
        }
    }
}

void splitMatND_threaded(const cv::MatND &xt, std::vector<cv::Mat> &xtsplit) {
    int w = xt.cols;
    int h = xt.rows;
    int cn = xt.channels();

    assert(cn == 28);
    assert(xtsplit.size() == 28);

#pragma omp parallel for num_threads(2)
    for (int k = 0; k < cn; k++)
    {
        for (int j = 0; j < h; ++j) {
            float *pDst = xtsplit[k].ptr<float>(j);
            const float *pSrc = xt.ptr<float>(j);

            for (int i = 0; i < w; ++i) {
                pDst[2*i] = pSrc[cn*i+k];
                pDst[2*i+1] = 0.0f;
            }
        }
    }
}

void preparation_step() {
    std::string sequence = "../sequence";

    std::string pattern_jpg = sequence + "/*.jpg";
    std::string txt_base_path = sequence + "/groundtruth.txt";

    std::vector<cv::String> image_files;
    cv::glob(pattern_jpg, image_files);
    if (image_files.size() == 0)
        return;

    std::vector<cv::Rect_<float>> groundtruth_rect;
    groundtruth_rect = Utils::getgroundtruth(txt_base_path);

    cv::Rect_<float> location = groundtruth_rect[0];
    cv::Mat image = cv::imread(image_files[0]);
    std::vector<cv::Rect_<float>> result_rects;


    StapleDummy tracker{};

    tracker.get_feature_map_windowed(image, location, xt_windowed);

    // initialising vector before call
    for (int ch = 0; ch < xt_windowed.channels(); ++ch) {
        xtsplit1[ch] = cv::Mat(xt_windowed.rows, xt_windowed.cols, CV_32FC2);
        xtsplit2[ch] = cv::Mat(xt_windowed.rows, xt_windowed.cols, CV_32FC2);
        xtsplit3[ch] = cv::Mat(xt_windowed.rows, xt_windowed.cols, CV_32FC2);
    }
}

void result_correctness_check_step() {
    preparation_step();

    splitMatND_impr(xt_windowed, xtsplit1);

    splitMatND_threaded(xt_windowed, xtsplit2);

    bool result = true;
    for (int i = 0; i < xtsplit1.size(); ++i) {
        if (!Utils::matIsEqual<float>(xtsplit1[i], xtsplit2[i], true)) {
            result = false;
            break;
        }
    }

    if (result)
        std::cout << "[Success] Matrixes are equal. Test passed!" << std::endl;
    else
        std::cout << "[Failure] Matrixes are non-equal. Test NOT PASSED!" << std::endl;
}

static void BM_SplitMatND(benchmark::State& state) {

    preparation_step();

    for (auto _ : state) {
        splitMatND(xt_windowed, xtsplit1);
    }
}
BENCHMARK(BM_SplitMatND);

static void BM_SplitMatND_improved(benchmark::State& state) {

    preparation_step();

    for (auto _ : state) {
        splitMatND_impr(xt_windowed, xtsplit2);
    }
}
BENCHMARK(BM_SplitMatND_improved);

static void BM_SplitMatND_threaded(benchmark::State& state) {

    preparation_step();

    for (auto _ : state) {
        splitMatND_threaded(xt_windowed, xtsplit2);
    }
}
BENCHMARK(BM_SplitMatND_threaded);

static void BM_SplitMatND_cache(benchmark::State& state) {

    preparation_step();

    for (auto _ : state) {
        splitMatND_cache(xt_windowed, xtsplit3);
    }
}
BENCHMARK(BM_SplitMatND_cache);

BENCHMARK_MAIN();

//int main() {
//    result_correctness_check_step();
//    return 0;
//}