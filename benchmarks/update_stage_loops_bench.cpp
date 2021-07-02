#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "staple_dummy.h"
#include <iostream>
#include "../src/utils.h"

static std::vector<cv::Mat> hf_num{28}, hf_den{28};
static std::vector<cv::Mat> hf1, hf2{28};
static cv::Mat response_cff, response_cfi, response_cf_inv;
static StapleDummy tracker{};

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

    tracker.trackerInit(image, location);

    image = cv::imread(image_files[1]);
    tracker.splitFeatureMap(image);

    tracker.get_hf(hf_num, hf_den);

    for (int i = 0; i < hf_num.size(); ++i) {
        hf2[i] = cv::Mat(tracker.featureMap.rows, tracker.featureMap.cols, CV_32FC2, cv::Scalar_<float>(0.f, 0.f));
    }

    response_cff = cv::Mat(tracker.featureMap.rows, tracker.featureMap.cols, CV_32FC2, cv::Scalar_<float>(0.0f, 0.0f));
    response_cfi = cv::Mat(tracker.featureMap.rows, tracker.featureMap.cols, CV_32FC2);
    response_cf_inv = cv::Mat(tracker.featureMap.rows, tracker.featureMap.cols, CV_32FC2);
}


void sum1() {
    cv::Mat sum_hf_den(tracker.featureMap.rows, tracker.featureMap.cols, CV_32FC1, (float)tracker._params.lambda);
    for (auto & ch : hf_den) {
        sum_hf_den += ch;
    }
}

void sum2() {
    const int w = tracker.featureMap.cols;
    const int h = tracker.featureMap.rows;
    std::vector<float> hf_den_sum(w * h, (float)tracker._params.lambda);

    for (int ch = 0; ch < tracker.featureMap.channels(); ++ch) {
        float* pDst = &hf_den_sum[0];
        for (int j = 0; j < h; ++j) {
            const float* pDen = hf_den[ch].ptr<float>(j);
            for (int i = 0; i < w; ++i, ++pDst) {
                *pDst += pDen[i];
            }
        }
    }
}

void hfUpdateForEachLoop(std::vector<cv::Mat> &new_hf) {

    cv::Mat sum_hf_den(tracker.featureMap.rows, tracker.featureMap.cols, CV_32FC1, (float)tracker._params.lambda);
    for (auto & ch : hf_den) {
        sum_hf_den += ch;
    }

    for (auto & ch : hf_num) {
        cv::Mat dim(tracker.featureMap.rows, tracker.featureMap.cols, CV_32FC2, cv::Scalar_<float>(0.f, 0.f));

        dim.forEach<cv::Vec2f>([&sum_hf_den, &ch](cv::Vec2f &pair, const int *pos) {
            pair[0] = ch.at<cv::Vec2f>(pos)[0] / sum_hf_den.at<float>(pos);
            pair[1] = ch.at<cv::Vec2f>(pos)[1] / sum_hf_den.at<float>(pos);
        });

        new_hf.push_back(std::move(dim));
    }
}


void hfUpdateForLoop(std::vector<cv::Mat> &new_hf) {
    const int w = tracker.featureMap.cols;
    const int h = tracker.featureMap.rows;
    std::vector<float> hf_den_sum(w * h, (float)tracker._params.lambda);

    for (int ch = 0; ch < tracker.featureMap.channels(); ++ch) {
        float* pDst = &hf_den_sum[0];
        for (int j = 0; j < h; ++j) {
            const float* pDen = hf_den[ch].ptr<float>(j);
            for (int i = 0; i < w; ++i, ++pDst) {
                *pDst += pDen[i];
            }
        }
    }

    for (int ch = 0; ch < tracker.featureMap.channels(); ++ch) {

        const float* pDenSum = &hf_den_sum[0];
        for (int j = 0; j < h; ++j) {
            const cv::Vec2f* pSrc = hf_num[ch].ptr<cv::Vec2f>(j);
            auto pDst = new_hf[ch].ptr<cv::Vec2f>(j);

            for (int i = 0; i < w; ++i, ++pDenSum, ++pDst, ++pSrc) {
                *pDst = *pSrc / *pDenSum;
            }
        }
    }
}


void hfUpdateForLoop_thr(std::vector<cv::Mat> &new_hf) {
    const int w = tracker.featureMap.cols;
    const int h = tracker.featureMap.rows;
    std::vector<float> hf_den_sum(w * h, (float)tracker._params.lambda);

    for (int ch = 0; ch < tracker.featureMap.channels(); ++ch) {
        float* pDst = &hf_den_sum[0];
        for (int j = 0; j < h; ++j) {
            const float* pDen = hf_den[ch].ptr<float>(j);
            for (int i = 0; i < w; ++i, ++pDst) {
                *pDst += pDen[i];
            }
        }
    }

#pragma omp parallel for num_threads(2)
    for (int ch = 0; ch < tracker.featureMap.channels(); ++ch) {
        const float* pDenSum = &hf_den_sum[0];
        for (int j = 0; j < h; ++j) {
            const cv::Vec2f* pSrc = hf_num[ch].ptr<cv::Vec2f>(j);
            auto pDst = new_hf[ch].ptr<cv::Vec2f>(j);

            for (int i = 0; i < w; ++i) {
                pDst[i] = pSrc[i] / pDenSum[i];
            }
        }
    }
}


void invUpdateForEachLoop(const std::vector<cv::Mat> &hf) {
    cv::Mat response_cf_sum(tracker.featureMap.rows, tracker.featureMap.cols, CV_32FC2, cv::Scalar_<float>(0.0f, 0.0f));

    for (int ch = 0; ch < tracker.featureMapSplitted.size(); ch++)
    {
        // performing complex numbers multiplication
        // conj(hf[ch]) .* featureMapSplitted[ch]
        response_cf_sum.forEach<cv::Vec2f>([&hf, &ch](cv::Vec2f &pair, const int *pos) {
                          auto xtf_vec = tracker.featureMapSplitted[ch].at<cv::Vec2f>(pos);
                          auto hf_vec  = hf[ch].at<cv::Vec2f>(pos);
                          pair[0] += xtf_vec[0] * hf_vec[0] + xtf_vec[1] * hf_vec[1];
                          pair[1] += xtf_vec[1] * hf_vec[0] - xtf_vec[0] * hf_vec[1];
                        }
                );
    }

    cv::dft(response_cf_sum, response_cf_inv, cv::DFT_SCALE | cv::DFT_INVERSE);
}


void invUpdateForLoop(const std::vector<cv::Mat> &hf) {
    const int w = tracker.featureMap.cols;
    const int h = tracker.featureMap.rows;

    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            response_cff.at<cv::Vec2f>(j, i) = cv::Vec2f(0.0f, 0.0f);

    for (size_t ch = 0; ch < hf.size(); ++ch) {
        for (int j = 0; j < h; ++j) {
            cv::Vec2f* pDst = response_cff.ptr<cv::Vec2f>(j);

            for (int i = 0; i < w; ++i, ++pDst) {
                cv::Vec2f pHF = hf[ch].at<cv::Vec2f>(j,i);
                cv::Vec2f pXTF = tracker.featureMapSplitted[ch].at<cv::Vec2f>(j,i);

                float sumr = (pHF[0] * pXTF[0] + pHF[1] * pXTF[1]);
                float sumi = (pHF[0] * pXTF[1] - pHF[1] * pXTF[0]);

                *pDst += cv::Vec2f(sumr, sumi);
            }
        }
    }

    cv::dft(response_cff, response_cfi, cv::DFT_SCALE | cv::DFT_INVERSE);
}


static void BM_sum1(benchmark::State& state) {
    preparation_step();
    for (auto _ : state) {
        sum1();
    }
}
BENCHMARK(BM_sum1);

static void BM_sum2(benchmark::State& state) {
    preparation_step();
    for (auto _ : state) {
        sum2();
    }
}
BENCHMARK(BM_sum2);

static void BM_hfUpdateForEachLoop(benchmark::State& state) {
    preparation_step();
    for (auto _ : state) {
        hfUpdateForEachLoop(hf1);
    }
}
BENCHMARK(BM_hfUpdateForEachLoop);

static void BM_hfUpdateForLoop(benchmark::State& state) {
    preparation_step();
    for (auto _ : state) {
        hfUpdateForLoop(hf2);
    }
}
BENCHMARK(BM_hfUpdateForLoop);

static void BM_hfUpdateForLoop_thr(benchmark::State& state) {
    preparation_step();
    for (auto _ : state) {
        hfUpdateForLoop_thr(hf2);
    }
}
BENCHMARK(BM_hfUpdateForLoop_thr);

static void BM_invUpdateForEachLoop(benchmark::State& state) {
    preparation_step();
    hfUpdateForEachLoop(hf1);
    for (auto _ : state) {
        invUpdateForEachLoop(hf1);
    }
}
BENCHMARK(BM_invUpdateForEachLoop);

static void BM_invUpdateForLoop(benchmark::State& state) {
    preparation_step();
    hfUpdateForLoop(hf2);
    for (auto _ : state) {
        invUpdateForLoop(hf2);
    }
}
BENCHMARK(BM_invUpdateForLoop);


BENCHMARK_MAIN();


// WARNING: test not passed, but u can see that data is equal
// (or there are very small difference in last bits that is unseen in console)
void result_correctness_check_step() {
    preparation_step();

    hfUpdateForEachLoop(hf1);
    hfUpdateForLoop(hf2);

    // WARNING: in result elements differ only in last bits
    bool result = true;
    for (int i = 0; i < hf2.size(); ++i) {
        if (!Utils::matIsEqual<float>(hf1[i], hf2[i], true)) {
            result = false;
            break;
        }
    }

    // using hf2 to avoid differences in input data, and so reduce diffs in output
    invUpdateForEachLoop(hf2);
    invUpdateForLoop(hf2);

    if (!Utils::matIsEqual<float>(response_cf_inv, response_cfi, true)) {
        result = false;
    }

    if (result)
        std::cout << "[Success] Matrixes are equal. Test passed!" << std::endl;
    else
        std::cout << "[Failure] Matrixes are non-equal. Test NOT PASSED!" << std::endl;
}

//int main() {
//    result_correctness_check_step();
//    return 0;
//}