#include "tracker.hpp"
#include "fhog.h"
#include <opencv2/imgproc.hpp>


void StapleTracker::trackerInit(const cv::Mat &im, const cv::Rect& bbox) {

    // needed for update and train routines
    firstFrame = true;

    target_sz.width = bbox.width;
    target_sz.height = bbox.height;
    
    // center_pos is the centre of the initial bounding box
    center_pos.x = bbox.x + bbox.width / 2;
    center_pos.y = bbox.y + bbox.height / 2;

    initAllAreas(im.size());

    // patch of the target + padding
    cv::Mat patch_paded;
    getSubwindow(im, center_pos, bg_size, patch_paded);
    cv::resize(patch_paded, patch_paded, norm_bg_size, 0, 0, cv::INTER_LINEAR);

    // init hist model
    updateHistModel(true, patch_paded);

    // Hann (cosine window)
    cv::createHanningWindow(hann_window, cf_response_size, CV_32FC1);

    // gaussian-shaped desired response, centred in (1,1)
    // bandwidth proportional to target size
    double output_sigma = std::sqrt(norm_target_size.width * norm_target_size.height) * 
            _params.output_sigma_factor / _params.hog_cell_size;
    
    cv::Mat y;
    createGaussianResponse(cf_response_size, output_sigma, y);
    cv::dft(y, yf);

    // FURTHER STEPS needed for first training
    // extract patch of size bg_size and resize to norm_bg_size
    cv::Mat im_patch_bg = patch_paded;

    // init adn compute feature map, of cf_response_size
    featureMap = cv::MatND(cf_response_size.height, cf_response_size.width, CV_32FC(FEATURE_CHANNELS));
    getFeatureMap(im_patch_bg, featureMap);

    // initializing feature map splits
    assert(featureMapSplitted.size() == featureMap.channels());
    for (int ch = 0; ch < FEATURE_CHANNELS; ++ch) {
        // 2 channel because after dft we get complex num (real + imaginary parts)
        featureMapSplitted[ch] = cv::Mat(featureMap.rows, featureMap.cols, CV_32FC2);
    }

    // compute FFT
    splitMatND(featureMap, featureMapSplitted);
    for (auto& channel : featureMapSplitted) {
        cv::dft(channel, channel);
    }

    // initial first train
    trackerTrain(im);
}


// returns optimal bg_size value in terms of optimal size for DFT
// cause cf_response_size depends on bg_size
// also optimal DFT size is faster to compute, so method used for optimisation purposes
cv::Size StapleTracker::getOptimalBgSize(const cv::Size &scene_sz, const cv::Size &target_sz, int fixed_area, int hog_cell_size) {
    // we want a regular frame surrounding the object
    auto avg_dim = (target_sz.width + target_sz.height) / 2.0;

    // size from which we extract features
    cv::Size bg_size;
    bg_size.width = std::round(target_sz.width + avg_dim);
    bg_size.height = std::round(target_sz.height + avg_dim);

    // saturate to image size
    bg_size.width = std::min(bg_size.width, scene_sz.width - 1);
    bg_size.height = std::min(bg_size.height, scene_sz.height - 1);

    // make copy for situations when there is no optimal size
    auto initial_bg_size = bg_size;

    int width_limit_max = std::min(bg_size.width + (int)(0.4 * bg_size.width), scene_sz.width - 1);
    int width_limit_min = bg_size.width - (int)(0.1 * bg_size.width);

    int height_limit_max = std::min(bg_size.height + (int)(0.4 * bg_size.height), scene_sz.height - 1);
    int height_limit_min = bg_size.height - (int)(0.1 * bg_size.height);

    int bg_size_width, bg_size_height;
    bool found = false;

    // check that grid of values and try to find [width x height] which will give optimal sizes
    // for cf_response_sizes when used in DFT
    for (bg_size_width = width_limit_min; bg_size_width <= width_limit_max; ++bg_size_width) {
        for (bg_size_height = height_limit_min; bg_size_height <= height_limit_max; ++bg_size_height) {

            bg_size.width = bg_size_width;
            bg_size.height = bg_size_height;

            // make sure the differences are a multiple of 2 (makes things easier later in color histograms)
            bg_size.width = bg_size.width - (bg_size.width - target_sz.width) % 2;
            bg_size.height = bg_size.height - (bg_size.height - target_sz.height) % 2;

            // Compute the rectangle with (or close to) params.fixedArea
            // and same aspect ratio as the target bbox
            double area_resize_factor = std::sqrt(fixed_area / double(bg_size.width * bg_size.height));

            cv::Size norm_bg_size, cf_response_size;
            norm_bg_size.width = std::round(bg_size.width * area_resize_factor);
            norm_bg_size.height = std::round(bg_size.height * area_resize_factor);

            // Correlation Filter (HOG) feature space
            cf_response_size.width = std::floor(norm_bg_size.width / hog_cell_size);
            cf_response_size.height = std::floor(norm_bg_size.height / hog_cell_size);

            if (cv::getOptimalDFTSize(cf_response_size.width) == cf_response_size.width &&
                    cv::getOptimalDFTSize(cf_response_size.height) == cf_response_size.height) {
                found = true;
                break;
            }
        }

        if (found) break;
    }

    if (found)
        return bg_size;
    else
        return initial_bg_size;
}


void StapleTracker::initAllAreas(const cv::Size &scene_sz) {

    // we want a regular frame surrounding the object
    auto avg_dim = (target_sz.width + target_sz.height) / 2.0;

    // size from which we extract features
    bg_size = getOptimalBgSize(scene_sz, target_sz, _params.fixed_area, _params.hog_cell_size);

    // pick a "safe" region smaller than bbox to avoid mislabeling
    fg_size.width = std::round(target_sz.width - avg_dim * _params.inner_padding);
    fg_size.height = std::round(target_sz.height - avg_dim * _params.inner_padding);

    // saturate to image size
    bg_size.width = std::min(bg_size.width, scene_sz.width - 1);
    bg_size.height = std::min(bg_size.height, scene_sz.height - 1);

    // make sure the differences are a multiple of 2 (makes things easier later in color histograms)
    bg_size.width = bg_size.width - (bg_size.width - target_sz.width) % 2;
    bg_size.height = bg_size.height - (bg_size.height - target_sz.height) % 2;

    fg_size.width = fg_size.width + (bg_size.width - fg_size.width) % 2;
    fg_size.height = fg_size.height + (bg_size.height - fg_size.width) % 2;

    // Compute the rectangle with (or close to) params.fixedArea
    // and same aspect ratio as the target bbox
    area_resize_factor = std::sqrt(_params.fixed_area / double(bg_size.width * bg_size.height));
    norm_bg_size.width = std::round(bg_size.width * area_resize_factor);
    norm_bg_size.height = std::round(bg_size.height * area_resize_factor);

    // Correlation Filter (HOG) feature space
    // It smaller that the norm bg area if HOG cell size is > 1
    cf_response_size.width = std::floor(norm_bg_size.width / _params.hog_cell_size);
    cf_response_size.height = std::floor(norm_bg_size.height / _params.hog_cell_size);

    // given the norm BG area, which is the corresponding target w and h?
    double norm_target_sz_w = 0.75 * norm_bg_size.width  - 0.25 * norm_bg_size.height;
    double norm_target_sz_h = 0.75 * norm_bg_size.height - 0.25 * norm_bg_size.width;

    norm_target_size.width = std::round(norm_target_sz_w);
    norm_target_size.height = std::round(norm_target_sz_h);

    // distance (on one side) between target and bg area
    cv::Size norm_pad;

    norm_pad.width = std::floor((norm_bg_size.width - norm_target_size.width) / 2.0);
    norm_pad.height = std::floor((norm_bg_size.height - norm_target_size.height) / 2.0);

    int radius = std::floor(std::fmin(norm_pad.width, norm_pad.height));

    // norm_delta_area is the number of rectangles that are considered.
    // it is the "sampling space" and the dimension of the final merged resposne
    // it is squared to not privilege any particular direction
    norm_delta_area.width  = 2 * radius + 1;
    norm_delta_area.height = 2 * radius + 1;

    // Rectangle in which the integral images are computed.
    // Grid of rectangles ( each of size norm_target_size) has size norm_delta_area.
    norm_pwp_search_size.width  = norm_target_size.width  + norm_delta_area.width - 1;
    norm_pwp_search_size.height = norm_target_size.height + norm_delta_area.height - 1;
}


void StapleTracker::getSubwindow(const cv::Mat &im, const cv::Point_<float> &center_pnt, const cv::Size &orig_sz, cv::Mat &out) {
    cv::Size sz = orig_sz; // scale adaptation

    // make sure the size is not to small
    sz.width = std::fmax(sz.width, 2);
    sz.height = std::fmax(sz.height, 2);

    // minimum one point of the region must be in the 'im', so values are choosen correspondigly
    cv::Point lefttop(
        std::min(im.cols - 1, std::max(-sz.width + 1, int(center_pnt.x + 1 - sz.width / 2.0 + 0.5))),
        std::min(im.rows - 1, std::max(-sz.height + 1, int(center_pnt.y + 1 - sz.height / 2.0 + 0.5)))
        );
    // minimum one point of the region must be in the 'im', so values are choosen correspondigly
    cv::Point rightbottom(
        std::max(0, int(lefttop.x + sz.width - 1)),
        std::max(0, int(lefttop.y + sz.height - 1))
        );
    // now select left-top point in the 'im'
    cv::Point lefttopLimit(
        std::max(lefttop.x, 0),
        std::max(lefttop.y, 0)
        );
    // now select right-bottom point in the 'im'
    cv::Point rightbottomLimit(
        std::min(rightbottom.x, im.cols - 1),
        std::min(rightbottom.y, im.rows - 1)
        );

    rightbottomLimit.x += 1;
    rightbottomLimit.y += 1;
    // rect on the image to be extracted
    cv::Rect roiRect(lefttopLimit, rightbottomLimit);
	
    // widths of borders
    int top = lefttopLimit.y - lefttop.y;
    int bottom = rightbottom.y - rightbottomLimit.y + 1;
    int left = lefttopLimit.x - lefttop.x;
    int right = rightbottom.x - rightbottomLimit.x + 1;
	
    cv::copyMakeBorder(im(roiRect), out, top, bottom, left, right, cv::BORDER_REPLICATE);
}


void StapleTracker::updateHistModel(bool new_model, const cv::Mat &patch, double learning_rate_pwp) {
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
#ifdef RPi
#pragma omp parallel sections
        {
#pragma omp section
            {
#endif
        cv::MatND bg_hist_tmp;
        cv::calcHist(&patch, imgCount, channels, bg_mask_new, bg_hist_tmp, dims, sizes, ranges);

        int bgtotal = cv::countNonZero(bg_mask_new);
        if (bgtotal == 0) {
            bgtotal = 1;
        }
        bg_hist_tmp = bg_hist_tmp / bgtotal;
        bg_hist = (1 - learning_rate_pwp) * bg_hist + learning_rate_pwp * bg_hist_tmp;
#ifdef RPi
            }
#pragma omp section
            {
#endif
        cv::MatND fg_hist_tmp;
        cv::calcHist(&patch, imgCount, channels, fg_mask_new, fg_hist_tmp, dims, sizes, ranges);

        int fgtotal = cv::countNonZero(fg_mask_new);
        if (fgtotal == 0) {
            fgtotal = 1;
        }
        fg_hist_tmp = fg_hist_tmp / fgtotal;

        fg_hist = (1 - learning_rate_pwp) * fg_hist + learning_rate_pwp * fg_hist_tmp;
#ifdef RPi
            }
        }
#endif
    }
}


void StapleTracker::createGaussianResponse(const cv::Size& rect_size, double sigma, cv::Mat &output) {

    double s = 2.0 * sigma * sigma; 

    output = cv::Mat(rect_size.height, rect_size.width, CV_32FC2);

    cv::Size half;
    half.width = std::floor((rect_size.width - 1) / 2);
    half.height = std::floor((rect_size.height - 1) / 2);

    // our response not like in original : /\ -> \/

    // generating kernel 
    for (int x = -half.width; x <= rect_size.width - (half.width + 1); x++) { 
        for (int y = -half.height; y <= rect_size.height - (half.height + 1); y++) { 
            // we need to put values for x and y in special position
            // to create down bell-shaped response (\/) so we 
            // shuffle halves like:
            // in original 1st half - /, 2nd half - \, in our case we place 2nd half \ on 1st position, 
            // and 1st half on 2nd position and form \/
            int x_pos = x < 1 ? rect_size.width  + x - 1 : x - 1; 
            int y_pos = y < 1 ? rect_size.height + y - 1 : y - 1;

            // formula is simplified
            cv::Vec2f val((std::exp(-(x*x + y*y) / s)), 0);
            output.at<cv::Vec2f>(y_pos, x_pos) = val; 
        }
    } 
}


void StapleTracker::trackerTrain(const cv::Mat &im) {
    
    // before TRAIN stage feature map should be generated and splitted to featureMapSplitted

    // FILTER UPDATE
    // Compute expectations over circular shifts,
    // therefore divide by number of pixels.

    static std::vector<cv::Mat> new_hf_num{FEATURE_CHANNELS};
    static std::vector<cv::Mat> new_hf_den{FEATURE_CHANNELS};
    assert(featureMapSplitted.size() == FEATURE_CHANNELS);

    // making init only in first frame, aka lazy-init instead of init in every call of function
    if (firstFrame) {
        for (int i = 0; i < FEATURE_CHANNELS; ++i) {
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
    for (int ch = 0; ch < FEATURE_CHANNELS; ++ch) {

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
        for (int ch =  0; ch < FEATURE_CHANNELS; ch++) {
            hf_den[ch] = (1 - _params.learning_rate_cf) * hf_den[ch] + _params.learning_rate_cf * new_hf_den[ch];
            hf_num[ch] = (1 - _params.learning_rate_cf) * hf_num[ch] + _params.learning_rate_cf * new_hf_num[ch];
        }

        cv::Mat im_patch_bg;
        getSubwindow(im, center_pos, bg_size, im_patch_bg);
        cv::resize(im_patch_bg, im_patch_bg, norm_bg_size, 0, 0, cv::INTER_LINEAR);
        updateHistModel(false, im_patch_bg, _params.learning_rate_pwp);
    }

    // update bbox position
    if (firstFrame) {
        rect_position.x = center_pos.x - target_sz.width/2;
        rect_position.y = center_pos.y - target_sz.height/2;
        rect_position.width = target_sz.width;
        rect_position.height = target_sz.height;
    }
}


void StapleTracker::getFeatureMap(cv::Mat &im_patch, cv::MatND &output) {

    fhog28(output, im_patch, _params.hog_cell_size, 9);

    int w = cf_response_size.width;
    int h = cf_response_size.height;

    cv::Mat new_im_patch;

    if (_params.hog_cell_size > 1) {
        cv::Size newsz(w, h);
        cv::resize(im_patch, new_im_patch, newsz, 0, 0, cv::INTER_LINEAR);
    } else {
        new_im_patch = im_patch;
    }

    cv::Mat grayimg;
    cv::cvtColor(new_im_patch, grayimg, cv::COLOR_BGR2GRAY);

    float alpha = 1. / 255.0;
    float betta = 0.5;

    typedef cv::Vec<float, 28> Vec28f;

    for (int j = 0; j < h; ++j)
    {
        Vec28f* pDst = output.ptr<Vec28f>(j);
        const float* pHann = hann_window.ptr<float>(j);
        const uchar* pGray = grayimg.ptr<uchar>(j);

        for (int i = 0; i < w; ++i, ++pDst, ++pHann, ++pGray)
        {
            // apply Hann window
            Vec28f& val = pDst[0];

            val = val * pHann[0];
            val[0] = (alpha * (float)(pGray[0]) - betta) * pHann[0];
        }
    }
}


void StapleTracker::splitFeatureMap(const cv::Mat &im) {
    // extract patch of size bg_size and resize to norm_bg_size
    cv::Mat im_patch_bg;
    getSubwindow(im, center_pos, bg_size, im_patch_bg);
    cv::resize(im_patch_bg, im_patch_bg, norm_bg_size, 0, 0, cv::INTER_LINEAR);

    // compute feature map, of cf_response_size
    getFeatureMap(im_patch_bg, featureMap);

    // compute FFT
    splitMatND(featureMap, featureMapSplitted);
    for (auto& channel : featureMapSplitted) {
        cv::dft(channel, channel);
    }
}


void StapleTracker::splitMatND(const cv::MatND &featureMap, std::vector<cv::Mat> &xtsplit) {
    int w = featureMap.cols;
    int h = featureMap.rows;
    int cn = featureMap.channels();

    assert(cn == FEATURE_CHANNELS);
    assert(xtsplit.size() == FEATURE_CHANNELS);

#pragma omp parallel for num_threads(2)
    for (int k = 0; k < cn; k++)
    {
        for (int j = 0; j < h; ++j) {
            auto *pDst = xtsplit[k].ptr<float>(j);
            const auto *pSrc = featureMap.ptr<float>(j);

            for (int i = 0; i < w; ++i) {
                pDst[2*i] = pSrc[cn*i + k];
                pDst[2*i + 1] = 0.0f;
            }
        }
    }
}


namespace {
    // Returns 1-channel real part of given complex Mat
    // and in debug mode checks that imaginary part is small
    //
    // cv::Mat &real_part_out - should be inited before call, as CV_32FC1 with
    //                          rows and cols same as in complex Mat.
    void ensure_real(const cv::Mat &complex, cv::Mat &real_part_out) {
        int w = complex.cols;
        int h = complex.rows;

        // evaluate next section only in debug mode
#ifndef NDEBUG
        double sum_r{0}, sum_i{0};

        for (int i = 0; i < w * h; i++) {
            sum_r += complex.at<cv::Vec2f>(i)[0] * complex.at<cv::Vec2f>(i)[0];
            sum_i += complex.at<cv::Vec2f>(i)[1] * complex.at<cv::Vec2f>(i)[1];
        }

        assert(std::sqrt(sum_r) * 1e-5 >= std::sqrt(sum_i));
#endif


        for (int j = 0; j < h; ++j) {
            auto pDst = real_part_out.ptr<float>(j);
            const float* pSrc = complex.ptr<float>(j);

            for (int i = 0; i < w; ++i, ++pDst, pSrc += 2) {
                *pDst = *pSrc;
            }
        }
    }
}


// TESTING step
cv::Rect StapleTracker::trackerUpdate(const cv::Mat &im) {

    // before UPDATE stage feature map should be generated and splited to featureMapSplitted

    // Correlation between filter and test patch gives the response
    // Solve diagonal system per pixel.

    static std::vector<cv::Mat> hf {FEATURE_CHANNELS};

    // first time initialization
    if (firstFrame) {
        for (int i = 0; i < hf_num.size(); ++i)
            hf[i] = cv::Mat(featureMap.rows, featureMap.cols, CV_32FC2, cv::Scalar_<float>(0.f, 0.f));
    }

    const int w = featureMap.cols;
    const int h = featureMap.rows;

    if (_params.den_per_channel) {
        for (uint ch = 0; ch < hf_num.size(); ++ch) {
            for (int j = 0; j < h; ++j) {
                const cv::Vec2f* pSrc = hf_num[ch].ptr<cv::Vec2f>(j);
                const float* pDen = hf_den[ch].ptr<float>(j);
                auto pDst = hf[ch].ptr<cv::Vec2f>(j);

                for (int i = 0; i < w; ++i) {
                    pDst[i] = pSrc[i] / (pDen[i] + _params.lambda);
                }
            }
        }
    }
    else {
        std::vector<float> hf_den_sum(w * h, (float)_params.lambda);

        for (int ch = 0; ch < FEATURE_CHANNELS; ++ch) {
            float* pDst = &hf_den_sum[0];
            for (int j = 0; j < h; ++j) {
                const float* pDen = hf_den[ch].ptr<float>(j);
                for (int i = 0; i < w; ++i, ++pDst) {
                    *pDst += pDen[i];
                }
            }
        }

#pragma omp parallel for num_threads(2)
        for (int ch = 0; ch < FEATURE_CHANNELS; ++ch) {
            const float* pDenSum = &hf_den_sum[0];
            for (int j = 0; j < h; ++j) {
                const cv::Vec2f* pSrc = hf_num[ch].ptr<cv::Vec2f>(j);
                auto pDst = hf[ch].ptr<cv::Vec2f>(j);

                for (int i = 0; i < w; ++i, ++pDenSum, ++pDst, ++pSrc) {
                    *pDst = *pSrc / *pDenSum;
                }
            }
        }
    }

    static cv::Mat response_cf_sum, response_cf_inv;

    if (firstFrame) {
        response_cf_sum = cv::Mat(h, w, CV_32FC2, cv::Scalar_<float>(0.f, 0.f));
        response_cf_inv = cv::Mat(h, w, CV_32FC2);
    }

    // as response_cf_sum accumulates sum, it need to be zeroed before accumulation
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            response_cf_sum.at<cv::Vec2f>(j, i) = cv::Vec2f(0.0f, 0.0f);

    for (int ch = 0; ch < FEATURE_CHANNELS; ch++)
    {
        // performing complex numbers multiplication
        // conj(hf[ch]) .* featureMapSplitted[ch]
        for (int j = 0; j < h; ++j) {
            auto pDst = response_cf_sum.ptr<cv::Vec2f>(j);

            for (int i = 0; i < w; ++i, ++pDst) {
                cv::Vec2f pHF = hf[ch].at<cv::Vec2f>(j,i);
                cv::Vec2f pXTF = featureMapSplitted[ch].at<cv::Vec2f>(j,i);

                float sumr = (pHF[0] * pXTF[0] + pHF[1] * pXTF[1]);
                float sumi = (pHF[0] * pXTF[1] - pHF[1] * pXTF[0]);

                *pDst += cv::Vec2f(sumr, sumi);
            }
        }
    }

    cv::dft(response_cf_sum, response_cf_inv, cv::DFT_SCALE | cv::DFT_INVERSE);

    static cv::Mat response_cf;
    if (firstFrame) {
        response_cf = cv::Mat(h, w, CV_32FC1);
    }
    ensure_real(response_cf_inv, response_cf);

    // Crop square search region (in feature pixels).

    static cv::Size newsz;
    if (firstFrame) {
        newsz = norm_delta_area;
        newsz.width = std::floor(newsz.width / _params.hog_cell_size);
        newsz.height = std::floor(newsz.height / _params.hog_cell_size);

        // newsz must be odd for function cropFilterResponse
        if (newsz.width % 2 == 0) {
            newsz.width -= 1;
        }
        if (newsz.height % 2 == 0) {
            newsz.height -= 1;
        }
    }

    cv::Mat response_cf_cropped;
    cropFilterResponse(response_cf, newsz, response_cf_cropped);

    if (_params.hog_cell_size > 1) {
        cv::resize(response_cf_cropped, response_cf_cropped, norm_delta_area, 0, 0, cv::INTER_LINEAR);
    }

    static cv::Size pwp_search_size;
    if (firstFrame) {
        pwp_search_size.width = std::round(norm_pwp_search_size.width / area_resize_factor);
        pwp_search_size.height = std::round(norm_pwp_search_size.height / area_resize_factor);
    }

    // extract patch of size pwp_search_size and resize to norm_pwp_search_size
    cv::Mat im_patch_pwp;
    getSubwindow(im, center_pos, pwp_search_size, im_patch_pwp);
    cv::resize(im_patch_pwp, im_patch_pwp, norm_pwp_search_size, 0, 0, cv::INTER_LINEAR);

    static cv::Mat likelihood_map;
    if (firstFrame) {
        likelihood_map = cv::Mat(im_patch_pwp.rows, im_patch_pwp.cols, CV_32FC1);
    }
    getColourMap(im_patch_pwp, likelihood_map);

    // each pixel of response_pwp loosely represents the likelihood that
    // the target (of size norm_target_size) is centred on it
    static cv::Mat response_pwp;
    if (firstFrame) {
        response_pwp = cv::Mat(likelihood_map.rows - norm_target_size.height + 1,
                               likelihood_map.cols - norm_target_size.width + 1, CV_32FC1);
    }
    getCenterLikelihood(likelihood_map, norm_target_size, response_pwp);

    // ESTIMATION
    static cv::Mat response;
    if (firstFrame) {
        response = cv::Mat(norm_delta_area.height, norm_delta_area.width, CV_32FC1);
    }
    mergeResponses(response_cf_cropped, response_pwp, response);

    double max_val = 0;
    cv::Point max_loc;

    // find max value and it's indices
    cv::minMaxLoc(response, nullptr, &max_val, nullptr, &max_loc);

    cv::Point2f center((norm_delta_area.width + 1) / 2.0 - 1, (norm_delta_area.height + 1) / 2.0 - 1);
    
    center_pos.x += (max_loc.x - center.x) / area_resize_factor;
    center_pos.y += (max_loc.y - center.y) / area_resize_factor;

    cv::Rect bounding_rect;
    bounding_rect.x = center_pos.x - target_sz.width / 2;
    bounding_rect.y = center_pos.y - target_sz.height / 2;
    bounding_rect.width = target_sz.width;
    bounding_rect.height = target_sz.height;

    return bounding_rect;
}


// CROPFILTERRESPONSE makes RESPONSE_CF of size RESPONSE_SIZE (i.e. same size of colour response)
// prerequisite: RESPONSE_SIZE width and height must be odd
void StapleTracker::cropFilterResponse(const cv::Mat &response_cf, const cv::Size &response_size, cv::Mat &output) {
    
    int w = response_cf.cols;
    int h = response_cf.rows;

    output = cv::Mat(response_size.height, response_size.width, CV_32FC1);

    cv::Size half;
    half.width = response_size.width / 2;
    half.height = response_size.height / 2;

    std::vector<int> i_mod_range;
    i_mod_range.reserve(response_size.height);
    std::vector<int> j_mod_range;
    j_mod_range.reserve(response_size.width);

    for (int k = -half.width; k <= half.width; k++) {
        j_mod_range.emplace_back((k - 1 + w) % w);
    }

    for (int k = -half.height; k <= half.height; k++) {
        i_mod_range.emplace_back((k - 1 + h) % h);
    }

    for (int i = 0; i < i_mod_range.size(); ++i) { 
        for (int j = 0; j < j_mod_range.size(); ++j) { 
            output.at<float>(i, j) = response_cf.at<float>(i_mod_range[i], j_mod_range[j]); 
        }
    } 
}


void StapleTracker::getColourMap(const cv::Mat &patch, cv::Mat& output) const {
    
    int h = patch.rows;
    int w = patch.cols;
    int d = patch.channels();

    int bin_width = 256 / _params.n_bins;

    assert(output.rows == h);
    assert(output.cols == w);
    assert(output.channels() == 1);

    // convert image to d channels array
    //patch_array = reshape(double(patch), w*h, d);

    for (int j = 0; j < h; ++j)
    {
        const uchar* pSrc = patch.ptr<uchar>(j);
        float* pDst = output.ptr<float>(j);

        for (int i = 0; i < w; ++i)
        {
            int b1 = pSrc[0] / bin_width;
            int b2 = pSrc[1] / bin_width;
            int b3 = pSrc[2] / bin_width;

            float* histd = (float*)bg_hist.data;
            float probg = histd[b1 * _params.n_bins * _params.n_bins + b2 * _params.n_bins + b3];

            histd = (float*)fg_hist.data;
            float profg = histd[b1 * _params.n_bins * _params.n_bins + b2 * _params.n_bins + b3];

            if (profg + probg == 0.0) {
                *pDst = 0.0;
            } 
            else {
                *pDst = profg / (profg + probg);
            }

            if (std::isnan(*pDst)) { 
                *pDst = 0.0;
            }

            pSrc += d;
            ++pDst;
        }
    }    
}


// GETCENTERLIKELIHOOD computes the sum over rectangles of size M.
// CENTER_LIKELIHOOD is the 'colour response'
void StapleTracker::getCenterLikelihood(const cv::Mat &object_likelihood, const cv::Size& m, cv::Mat& center_likelihood) {
    
    int h = object_likelihood.rows;
    int w = object_likelihood.cols;
    
    int n1 = w - m.width + 1;
    int n2 = h - m.height + 1;
    
    float invArea = 1.f / (float)(m.width * m.height);

    cv::Mat integral_img;
    // compute integral image
    cv::integral(object_likelihood, integral_img);

    assert(center_likelihood.rows == n2);
    assert(center_likelihood.cols == n1);
    assert(center_likelihood.channels() == 1);

    for (int j = 0; j < n2; ++j)
    {
        for (int i = 0; i < n1; ++i)
        {
            center_likelihood.at<float>(j, i) =  invArea * (integral_img.at<double>(j, i) + integral_img.at<double>(j+m.height, i+m.width) 
                - integral_img.at<double>(j, i+m.width) - integral_img.at<double>(j+m.height, i));
        }
    }
}


// MERGERESPONSES interpolates the two responses with the hyperparameter MERGE_FACTOR
void StapleTracker::mergeResponses(const cv::Mat &response_cf, const cv::Mat &response_pwp, cv::Mat &response) const {
    response = (1 - _params.merge_factor) * response_cf + _params.merge_factor * response_pwp;
}


cv::Rect StapleTracker::getNextPos(const cv::Mat &im) {
    splitFeatureMap(im);
    cv::Rect newPos = trackerUpdate(im);
    firstFrame = false;
    trackerTrain(im);
    return newPos;
}
