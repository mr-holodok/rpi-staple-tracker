#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "staple_dummy.h"
#include <iostream>
#include "../src/utils.h"


static StapleDummy tracker{};
static cv::Mat image;


void preparation_step() {
    std::string sequence = "../sequence";

    std::string pattern_jpg = sequence + "/*.jpg";
    std::string txt_base_path = sequence + "/groundtruth.txt";

    std::vector<cv::String> image_files;
    cv::glob(pattern_jpg, image_files);
    if (image_files.empty())
        return;

    std::vector<cv::Rect_<float>> groundtruth_rect;
    groundtruth_rect = Utils::getgroundtruth(txt_base_path);

    cv::Rect_<float> location = groundtruth_rect[0];
    image = cv::imread(image_files[0]);
    std::vector<cv::Rect_<float>> result_rects;

    tracker.trackerInit(image, location);

    image = cv::imread(image_files[1]);

    cv::Mat patch_paded;
    tracker.getSubwindow(image, tracker.center_pos, tracker.bg_size, patch_paded);
    cv::resize(patch_paded, patch_paded, tracker.norm_bg_size, 0, 0, cv::INTER_LINEAR);
    tracker.updateHistModel_new(true, patch_paded);

    tracker.trackerTrain_new(image);
    tracker.firstFrame = false;
}

static void BM_histBasic(benchmark::State& state) {
    preparation_step();
    cv::Mat patch_paded;
    tracker.getSubwindow(image, tracker.center_pos, tracker.bg_size, patch_paded);
    cv::resize(patch_paded, patch_paded, tracker.norm_bg_size, 0, 0, cv::INTER_LINEAR);

    for (auto _ : state) {
        tracker.updateHistModel(false, patch_paded, tracker._params.learning_rate_pwp);
    }
}
BENCHMARK(BM_histBasic);


static void BM_histStatic(benchmark::State& state) {
    preparation_step();
    cv::Mat patch_paded;
    tracker.getSubwindow(image, tracker.center_pos, tracker.bg_size, patch_paded);
    cv::resize(patch_paded, patch_paded, tracker.norm_bg_size, 0, 0, cv::INTER_LINEAR);

    for (auto _ : state) {
        tracker.updateHistModel_new(false, patch_paded, tracker._params.learning_rate_pwp);
    }
}
BENCHMARK(BM_histStatic);


static void BM_trainBasic(benchmark::State& state) {
    preparation_step();

    for (auto _ : state) {
        tracker.trackerTrain(image);
    }
}
BENCHMARK(BM_trainBasic);


static void BM_train_thr(benchmark::State& state) {
    preparation_step();

    for (auto _ : state) {
        tracker.trackerTrain_new(image);
    }
}
BENCHMARK(BM_train_thr);


BENCHMARK_MAIN();
