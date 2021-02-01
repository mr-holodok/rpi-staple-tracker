#ifndef TRACKER_TRACKER_H
#define TRACKER_TRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


class StapleTracker {
public:
    
    void trackerInit(const cv::Mat &im, cv::Rect bbox);
    cv::Rect getNextPos(const cv::Mat &im);

protected:
    void trackerTrain(const cv::Mat &im);
    cv::Rect trackerUpdate(const cv::Mat &im);

    void initAllAreas(const cv::Mat &im);
    void getSubwindow(const cv::Mat &im, cv::Point_<float> center_pnt, cv::Size model_sz, cv::Size orig_sz, cv::Mat &out);
    void updateHistModel(bool new_model, cv::Mat &patch, double learning_rate_pwp=0.0);
    void createGaussianResponse(cv::Size rect_size, double sigma, cv::Mat &output);
    void getFeatureMap(cv::Mat &im_patch, cv::MatND &output);
    void splitMatND(const cv::MatND &xt, std::vector<cv::Mat> &xtsplit);
    void cropFilterResponse(const cv::Mat &response_cf, cv::Size response_size, cv::Mat& output);
    void getColourMap(const cv::Mat &patch, cv::Mat& output);
    void getCenterLikelihood(const cv::Mat &object_likelihood, cv::Size m, cv::Mat& center_likelihood);
    void mergeResponses(const cv::Mat &response_cf, const cv::Mat &response_pwp, cv::Mat &response);

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

    std::vector<cv::Mat> hf_den;
    std::vector<cv::Mat> hf_num;

    cv::Rect rect_position;
};

#endif 
