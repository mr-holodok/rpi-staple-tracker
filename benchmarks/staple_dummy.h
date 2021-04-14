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
        firstFrame = false;
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

    void updateHistModel_new(bool new_model, const cv::Mat &patch, double learning_rate_pwp=0.0) {
        static cv::Mat fg_mask_new, bg_mask_new;
        if (new_model) {
            // Get BG mask (frame around target_sz)
            cv::Size pad_offset1;
            // we constrained the difference to be mod2, so we do not have to round here
            pad_offset1.width = (bg_size.width - target_sz.width) / 2;
            pad_offset1.height = (bg_size.height - target_sz.height) / 2;

            pad_offset1.width = std::fmax(pad_offset1.width, 1);
            pad_offset1.height = std::fmax(pad_offset1.height, 1);

            cv::Mat bg_mask(bg_size, CV_8UC1, cv::Scalar(1));// init bg_mask

            cv::Rect pad1_rect(
                    pad_offset1.width,
                    pad_offset1.height,
                    bg_size.width - 2 * pad_offset1.width,
                    bg_size.height - 2 * pad_offset1.height);

            bg_mask(pad1_rect) = false;

            // Get FG mask (inner portion of target_sz)
            cv::Size pad_offset2;

            // we constrained the difference to be mod2, so we do not have to round here
            pad_offset2.width = (bg_size.width - fg_size.width) / 2;
            pad_offset2.height = (bg_size.height - fg_size.height) / 2;

            pad_offset2.width = std::fmax(pad_offset2.width, 1);
            pad_offset2.height = std::fmax(pad_offset2.height, 1);

            cv::Mat fg_mask(bg_size, CV_8UC1, cv::Scalar(0));// init fg_mask

            cv::Rect pad2_rect(
                    pad_offset2.width,
                    pad_offset2.height,
                    bg_size.width - 2 * pad_offset2.width,
                    bg_size.height - 2 * pad_offset2.height);

            fg_mask(pad2_rect) = true;

            cv::resize(fg_mask, fg_mask_new, norm_bg_size, 0, 0, cv::INTER_LINEAR);
            cv::resize(bg_mask, bg_mask_new, norm_bg_size, 0, 0, cv::INTER_LINEAR);
        }

        int imgCount = 1;
        int dims = 3;
        const int sizes[] = { _params.n_bins, _params.n_bins, _params.n_bins };
        const int channels[] = { 0, 1, 2 };
        float colorRange[] = { 0, 256 };
        const float *ranges[] = { colorRange, colorRange, colorRange };

        // (TRAIN) BUILD THE MODEL
        if (new_model) {
            cv::calcHist(&patch, imgCount, channels, bg_mask_new, bg_hist, dims, sizes, ranges);
            int bgtotal = cv::countNonZero(bg_mask_new);
            if (bgtotal == 0) {
                bgtotal = 1;
            }
            bg_hist = bg_hist / bgtotal;

            cv::calcHist(&patch, imgCount, channels, fg_mask_new, fg_hist, dims, sizes, ranges);
            int fgtotal = cv::countNonZero(fg_mask_new);
            if (fgtotal == 0) {
                fgtotal = 1;
            }
            fg_hist = fg_hist / fgtotal;
        }
        else { // update the model
            cv::MatND bg_hist_tmp;
            cv::calcHist(&patch, imgCount, channels, bg_mask_new, bg_hist_tmp, dims, sizes, ranges);

            int bgtotal = cv::countNonZero(bg_mask_new);
            if (bgtotal == 0) {
                bgtotal = 1;
            }
            bg_hist_tmp = bg_hist_tmp / bgtotal;
            bg_hist = (1 - learning_rate_pwp) * bg_hist + learning_rate_pwp * bg_hist_tmp;

            cv::MatND fg_hist_tmp;
            cv::calcHist(&patch, imgCount, channels, fg_mask_new, fg_hist_tmp, dims, sizes, ranges);

            int fgtotal = cv::countNonZero(fg_mask_new);
            if (fgtotal == 0) {
                fgtotal = 1;
            }
            fg_hist_tmp = fg_hist_tmp / fgtotal;
            fg_hist = (1 - learning_rate_pwp) * fg_hist + learning_rate_pwp * fg_hist_tmp;
        }
    }

    void trackerTrain_new(const cv::Mat &im) {

        // before TRAIN stage feature map should be generated and splitted to featureMapSplitted

        // FILTER UPDATE
        // Compute expectations over circular shifts,
        // therefore divide by number of pixels.

        static std::vector<cv::Mat> new_hf_num{FEATURE_CHANNELS};
        static std::vector<cv::Mat> new_hf_den{FEATURE_CHANNELS};
        assert(featureMapSplitted.size() == FEATURE_CHANNELS);

        // making init only in first frame, aka lazy-init instead of init in every call of function
        if (firstFrame) {
            for (int i = 0; i < featureMapSplitted.size(); ++i) {
                // with 2 channels
                new_hf_num[i] = cv::Mat(featureMap.rows, featureMap.cols, CV_32FC2);
                // with only 1 channel because after multiplication imaginary part are zeroed, so unnecessary
                new_hf_den[i] = cv::Mat(featureMap.rows, featureMap.cols, CV_32FC1);
            }
        }

        int w = featureMap.cols;
        int h = featureMap.rows;
        float invArea = 1.f / (float)(cf_response_size.width * cf_response_size.height);

#pragma omp parallel for num_threads(2)
        for (int ch = 0; ch < featureMapSplitted.size(); ++ch) {

            // performing complex numbers multiplication
            // conj(yf) .* featureMapSplitted[ch]
            for (int j = 0; j < h; ++j) {
                const float* pFM = featureMapSplitted[ch].ptr<float>(j);
                const float* pYF = yf.ptr<float>(j);
                auto pDst = new_hf_num[ch].ptr<cv::Vec2f>(j);

                for (int i = 0; i < w; ++i) {
                    cv::Vec2f val(pYF[2*i+1] * pFM[2*i+1] + pYF[2*i] * pFM[2*i], pYF[2*i] * pFM[2*i+1] - pYF[2*i+1] * pFM[2*i]);
                    pDst[i] = invArea * val;
                }
            }

            // performing complex numbers multiplication
            // conj(featureMapSplitted[ch]) .* featureMapSplitted[ch]
            for (int j = 0; j < h; ++j) {
                const float* pFM = featureMapSplitted[ch].ptr<float>(j);
                auto pDst = new_hf_den[ch].ptr<float>(j);

                for (int i = 0; i < w; ++i) {
                    pDst[i] = invArea * (pFM[2*i] * pFM[2*i] + pFM[2*i+1] * pFM[2*i+1]);
                }
            }
        }

        if (firstFrame) {
            // first frame, train with a single image
            // as Mat type operator = performs only header copy, we need to do deep copy instead
            for (int ch = 0; ch < FEATURE_CHANNELS; ++ch) {
                hf_den[ch] = new_hf_den[ch].clone();
                hf_num[ch] = new_hf_num[ch].clone();
            }
        } else {
            // subsequent frames, update the model by linear interpolation
#pragma omp parallel for num_threads(2)
            for (int ch =  0; ch < featureMap.channels(); ch++) {
                hf_den[ch] = (1 - _params.learning_rate_cf) * hf_den[ch] + _params.learning_rate_cf * new_hf_den[ch];
                hf_num[ch] = (1 - _params.learning_rate_cf) * hf_num[ch] + _params.learning_rate_cf * new_hf_num[ch];
            }

            cv::Mat im_patch_bg;
            getSubwindow(im, center_pos, bg_size, im_patch_bg);
            cv::resize(im_patch_bg, im_patch_bg, norm_bg_size, 0, 0, cv::INTER_LINEAR);
            updateHistModel_new(false, im_patch_bg, _params.learning_rate_pwp);
        }

        // update bbox position
        if (firstFrame) {
            rect_position.x = center_pos.x - target_sz.width/2;
            rect_position.y = center_pos.y - target_sz.height/2;
            rect_position.width = target_sz.width;
            rect_position.height = target_sz.height;
        }
    }

    using StapleTracker::getFeatureMap;
    using StapleTracker::splitFeatureMap;
    using StapleTracker::getSubwindow;
    using StapleTracker::splitMatND;
    using StapleTracker::updateHistModel;
    using StapleTracker::trackerTrain;

    using StapleTracker::yf;
    using StapleTracker::featureMapSplitted;
    using StapleTracker::cf_response_size;
    using StapleTracker::featureMap;
    using StapleTracker::_params;
    using StapleTracker::center_pos;
    using StapleTracker::bg_size;
    using StapleTracker::norm_bg_size;
    using StapleTracker::FEATURE_CHANNELS;
    using StapleTracker::firstFrame;
};

#endif//STAPLE_STAPLE_DUMMY_H
