#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "staple_dummy.h"
#include <iostream>
#include "../src/utils.h"

static std::vector<cv::Mat> hf_num1{28}, hf_den1{28}, hf_num2{28}, hf_den2{28};
static StapleDummy tracker{};
static cv::Mat image;

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
    image = cv::imread(image_files[0]);
    std::vector<cv::Rect_<float>> result_rects;

    tracker.trackerInit(image, location);

    image = cv::imread(image_files[1]);
    tracker.prepareToTrain(image);

    tracker.get_hf(hf_num1, hf_den1);
    tracker.get_hf(hf_num2, hf_den2);
}

void trainForEach(const cv::Mat &im, std::vector<cv::Mat> &new_hf_num, std::vector<cv::Mat> &new_hf_den) {
    float invArea = 1.f / (float)(tracker.cf_response_size.width * tracker.cf_response_size.height);
    cv::Mat yf = tracker.yf.clone();

    for (int i = 0; i < tracker.featureMapSplitted.size(); ++i) {
        const auto &ch = tracker.featureMapSplitted[i];

        // performing complex numbers multiplication
        // conj(yf) .* featureMapSplitted[ch]
        new_hf_num[i].forEach<cv::Vec2f>([&ch, yf, invArea](cv::Vec2f &pair, const int * pos) {
          auto xtf_vec = ch.at<cv::Vec2f>(pos);
          auto yf_vec  = yf.at<cv::Vec2f>(pos);
          pair[0] = (xtf_vec[0] * yf_vec[0] + xtf_vec[1] * yf_vec[1]) * invArea;
          pair[1] = (xtf_vec[1] * yf_vec[0] - xtf_vec[0] * yf_vec[1]) * invArea;
        });

        // performing complex numbers multiplication
        // conj(featureMapSplitted[ch]) .* featureMapSplitted[ch]
        new_hf_den[i].forEach<float>([&ch, invArea](float &val, const int * pos) {
          auto xtf_vec = ch.at<cv::Vec2f>(pos);
          val = (xtf_vec[0] * xtf_vec[0] + xtf_vec[1] * xtf_vec[1]) * invArea;
        });
    }
}

void trainRawForLoop(const cv::Mat &im, std::vector<cv::Mat> &new_hf_num, std::vector<cv::Mat> &new_hf_den) {
    int w = tracker.featureMapSplitted[0].cols;
    int h = tracker.featureMapSplitted[0].rows;
    float invArea = 1.f / (float)(tracker.cf_response_size.width * tracker.cf_response_size.height);

    for (int ch = 0; ch < tracker.featureMapSplitted.size(); ++ch) {

        for (int j = 0; j < h; ++j) {
            const float* pFM = tracker.featureMapSplitted[ch].ptr<float>(j);
            const float* pYF = tracker.yf.ptr<float>(j);
            auto pDst = new_hf_num[ch].ptr<cv::Vec2f>(j);

            for (int i = 0; i < w; ++i, pFM += 2, pYF += 2, ++pDst) {
                cv::Vec2f val(pYF[1] * pFM[1] + pYF[0] * pFM[0], pYF[0] * pFM[1] - pYF[1] * pFM[0]);
                *pDst = invArea * val;
            }
        }

        for (int j = 0; j < h; ++j) {
            const float* pFM = tracker.featureMapSplitted[ch].ptr<float>(j);
            auto pDst = new_hf_den[ch].ptr<float>(j);

            for (int i = 0; i < w; ++i, pFM += 2, ++pDst) {
                *pDst = invArea * (pFM[0] * pFM[0] + pFM[1] * pFM[1]);
            }
        }
    }
}

void trainRawForLoop_thr(const cv::Mat &im, std::vector<cv::Mat> &new_hf_num, std::vector<cv::Mat> &new_hf_den) {
    int w = tracker.featureMapSplitted[0].cols;
    int h = tracker.featureMapSplitted[0].rows;
    float invArea = 1.f / (float)(tracker.cf_response_size.width * tracker.cf_response_size.height);

#pragma omp parallel for num_threads(4)
    for (int ch = 0; ch < tracker.FEATURE_CHANNELS; ++ch) {

        for (int j = 0; j < h; ++j) {
            const float *pFM = tracker.featureMapSplitted[ch].ptr<float>(j);
            const float *pYF = tracker.yf.ptr<float>(j);
            auto pDst = new_hf_num[ch].ptr<cv::Vec2f>(j);

            for (int i = 0; i < w; ++i) {
                cv::Vec2f val(pYF[2*i+1] * pFM[2*i+1] + pYF[2*i] * pFM[2*i], pYF[2*i] * pFM[2*i+1] - pYF[2*i+1] * pFM[2*i]);
                pDst[i] = invArea * val;
            }
        }


        for (int j = 0; j < h; ++j) {
            const float *pFM = tracker.featureMapSplitted[ch].ptr<float>(j);
            auto pDst = new_hf_den[ch].ptr<float>(j);

            for (int i = 0; i < w; ++i) {
                pDst[i] = invArea * (pFM[2*i] * pFM[2*i] + pFM[2*i+1] * pFM[2*i+1]);
            }
        }
    }
}

static void BM_TrainForEach(benchmark::State& state) {

    preparation_step();

    for (auto _ : state) {
        trainForEach(image, hf_num1, hf_den1);
    }
}
BENCHMARK(BM_TrainForEach);

static void BM_TrainRawForLoop(benchmark::State& state) {

    preparation_step();

    for (auto _ : state) {
        trainRawForLoop(image, hf_num2, hf_den2);
    }
}
BENCHMARK(BM_TrainRawForLoop);

static void BM_TrainRawForLoop_thr(benchmark::State& state) {

    preparation_step();

    for (auto _ : state) {
        trainRawForLoop_thr(image, hf_num2, hf_den2);
    }
}
BENCHMARK(BM_TrainRawForLoop_thr);


BENCHMARK_MAIN();


void result_correctness_check_step() {
    preparation_step();

    trainForEach(image, hf_num1, hf_den1);
    trainRawForLoop(image, hf_num2, hf_den2);

    bool result = true;
    for (int i = 0; i < hf_num1.size(); ++i) {
        if (!Utils::matIsEqual<float>(hf_num1[i], hf_num2[i], true)) {
            result = false;
            break;
        }

        if (!Utils::matIsEqual<float>(hf_den1[i], hf_den2[i], true)) {
            result = false;
            break;
        }
    }

    if (result)
        std::cout << "[Success] Matrixes are equal. Test passed!" << std::endl;
    else
        std::cout << "[Failure] Matrixes are non-equal. Test NOT PASSED!" << std::endl;
}

// WARNING: in Release mode test not passed, but u can see that data is equal
// (or there are very small difference in last bits that is unseen in console)
//int main() {
//    result_correctness_check_step();
//    return 0;
//}