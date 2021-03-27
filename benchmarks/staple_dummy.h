#ifndef STAPLE_STAPLE_DUMMY_H
#define STAPLE_STAPLE_DUMMY_H

#include "../src/tracker.hpp"

class StapleDummy : public StapleTracker {
public:
    // for benchmark
    void get_feature_map_windowed(const cv::Mat &im, cv::Rect bbox, cv::MatND &output) {
        trackerInit(im, bbox);

        cv::Mat im_patch_cf;
        getSubwindow(im, center_pos, bg_size, im_patch_cf);
        cv::resize(im_patch_cf, im_patch_cf, norm_bg_size, 0, 0, cv::INTER_LINEAR);

        output = cv::MatND(im_patch_cf.rows / _params.hog_cell_size, im_patch_cf.cols  / _params.hog_cell_size, CV_32FC(28));
        // compute feature map
        getFeatureMap(im_patch_cf, output);
    }

    void get_patch_for_feature_map(const cv::Mat &im, cv::Rect bbox, cv::Mat &patch) {
        trackerInit(im, bbox);
        getSubwindow(im, center_pos, bg_size, patch);
        cv::resize(patch, patch, norm_bg_size, 0, 0, cv::INTER_LINEAR);
    }

    void get_channels_for_dft(const cv::Mat &im, cv::Rect bbox, std::vector<cv::Mat>& channels) {
        trackerInit(im, bbox);

        cv::Mat im_patch_cf;
        getSubwindow(im, center_pos, bg_size, im_patch_cf);
        cv::resize(im_patch_cf, im_patch_cf, norm_bg_size, 0, 0, cv::INTER_LINEAR);

        cv::MatND xt_windowed = cv::MatND(im_patch_cf.rows / _params.hog_cell_size, im_patch_cf.cols  / _params.hog_cell_size, CV_32FC(28));
        getFeatureMap(im_patch_cf, xt_windowed);

        for (int ch = 0; ch < xt_windowed.channels(); ++ch) {
            // 2 channel because after dft we get complex num (real + imaginary parts)
            channels[ch] = cv::Mat(xt_windowed.rows, xt_windowed.cols, CV_32FC2);
        }
        splitMatND(xt_windowed, channels);
    }

    void prepareToTrain(const cv::Mat &im) {
        splitFeatureMap(im);
        cv::Rect newPos = trackerUpdate(im);
    }

    void get_hf(std::vector<cv::Mat> &hf_num_out, std::vector<cv::Mat> &hf_den_out) {
        assert(hf_num_out.size() == FEATURE_CHANNELS);
        assert(hf_den_out.size() == FEATURE_CHANNELS);

        for (int i = 0; i < FEATURE_CHANNELS; ++i) {
            // with 2 channels
            hf_num_out[i] = cv::Mat(featureMap.rows, featureMap.cols, CV_32FC2);
            // with only 1 channel because after multiplication imaginary part are zeroed, so unnecessary
            hf_den_out[i] = cv::Mat(featureMap.rows, featureMap.cols, CV_32FC1);
        }

        for (int ch = 0; ch < FEATURE_CHANNELS; ++ch) {
            hf_den_out[ch] = hf_den[ch].clone();
            hf_num_out[ch] = hf_num[ch].clone();
        }
    }

    using StapleTracker::yf;
    using StapleTracker::featureMapSplitted;
    using StapleTracker::cf_response_size;
};

#endif//STAPLE_STAPLE_DUMMY_H
