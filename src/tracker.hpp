#ifndef TRACKER_TRACKER_H
#define TRACKER_TRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


class StapleTracker {
public:
    
    void trackerInit(const cv::Mat &im, const cv::Rect& bbox);
    cv::Rect getNextPos(const cv::Mat &im);

protected:
    void trackerTrain(const cv::Mat &im);
    cv::Rect trackerUpdate(const cv::Mat &im);

    void initAllAreas(const cv::Size &scene_sz);
    void updateHistModel(bool new_model, const cv::Mat &patch, double learning_rate_pwp=0.0);
    void getFeatureMap(cv::Mat &im_patch, cv::MatND &output);
    void splitFeatureMap(const cv::Mat &im);
    void getColourMap(const cv::Mat &patch, cv::Mat& output) const;
    void mergeResponses(const cv::Mat &response_cf, const cv::Mat &response_pwp, cv::Mat &response) const;

    static void getSubwindow(const cv::Mat &im, const cv::Point_<float>& center_pnt, const cv::Size &orig_sz, cv::Mat &out);
    static void createGaussianResponse(const cv::Size& rect_size, double sigma, cv::Mat &output);
    static void splitMatND(const cv::MatND &xt, std::vector<cv::Mat> &xtsplit);
    static void cropFilterResponse(const cv::Mat &response_cf, const cv::Size& response_size, cv::Mat& output);
    static void getCenterLikelihood(const cv::Mat &object_likelihood, const cv::Size& m, cv::Mat& center_likelihood);
    static cv::Size getOptimalBgSize(const cv::Size &scene_sz, const cv::Size &target_sz, int fixed_area, int hog_cell_size);

    struct params {
        // default values are present in case of failed config-file read
        int hog_cell_size = 4;
        int fixed_area = 22500;
        int n_bins = 32;
        double learning_rate_pwp = 0.04;
        double inner_padding = 0.2;
        double output_sigma_factor = 0.0625;
        double lambda = 0.001;
        double learning_rate_cf = 0.01;
        double merge_factor = 0.3;
        bool den_per_channel = false;
    } _params;

    cv::MatND featureMap;
    const static uint FEATURE_CHANNELS = 28;
    std::vector<cv::Mat> featureMapSplitted{FEATURE_CHANNELS};

    bool firstFrame = false;

    cv::Point center_pos;
    cv::Size target_sz;

    cv::Size bg_size;
    cv::Size fg_size;
    double area_resize_factor;
    cv::Size cf_response_size;

    cv::Size norm_bg_size;
    cv::Size norm_target_size;
    cv::Size norm_delta_area;
    cv::Size norm_pwp_search_size;

    cv::MatND bg_hist;
    cv::MatND fg_hist;

    cv::Mat hann_window;
    cv::Mat yf;

    std::vector<cv::Mat> hf_den{FEATURE_CHANNELS};
    std::vector<cv::Mat> hf_num{FEATURE_CHANNELS};

    cv::Rect rect_position;
};

#endif 
