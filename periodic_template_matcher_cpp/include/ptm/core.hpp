#pragma once
#include <array>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

namespace ptm {
using Json = nlohmann::json;
using Config = Json; // normalized/validated runtime configuration, independent of Python
using Matrix = cv::Matx33d;
struct Features {
    cv::Mat norm, grad, edges, mask;
};
struct ObjectRef {
    std::string identity;
    cv::Mat mask;
    bool background_known = true;
    int shape_index = -1;
};
struct Inputs {
    std::vector<cv::Mat> templates, labels, template_valids, annotation_valids;
    cv::Mat target, target_valid;
    std::vector<std::vector<ObjectRef>> objects;
    bool has_objects = false;
    Json regions = Json::object();
};
struct MatchResult {
    Json registration = Json::object();
    cv::Mat input_image, template_image, warped_template, geometric_valid;
    cv::Mat mapped_labels, label_valid, completed_template, completed_valid;
    cv::Mat periodic_valid, label_periodic_valid, object_identities, object_valid, base_labels,
        base_valid;
};
struct ObjectPrediction {
    cv::Mat labels, identities, valid;
    Json diagnostics;
};

// Masks throughout the core are CV_8U with values 0/1, never OpenCV's 0/255 comparisons.
cv::Mat binary(const cv::Mat &a);
cv::Mat gray(const cv::Mat &image);
cv::Mat warp(const cv::Mat &a, const Matrix &H, cv::Size size, int interpolation = 1);
cv::Mat warp_mask(const cv::Mat &a, const Matrix &H, cv::Size size);
std::pair<cv::Mat, cv::Mat> resize_matching(const cv::Mat &gray, const cv::Mat &mask,
                                            cv::Size size);
double percentile(std::vector<double> values, double p);
int pyround(double v);
double matching_scale(const Config &cfg, const std::vector<cv::Size> &sizes);
Matrix scale_matrix(cv::Size original, cv::Size resized);
Json matrix_json(const Matrix &H);
Matrix matrix_from_json(const Json &value);
Matrix result_matrix(const MatchResult &r);
Config default_config();
void validate_config(const Config &cfg);
std::vector<std::string> periodic_axes(const Config &cfg, int class_id);
bool allows_shift(const Config &cfg, int class_id, const std::vector<cv::Vec2d> &vectors);
std::string repeat_axis(const Config &cfg);
std::vector<double> selected(const cv::Mat &image, const cv::Mat &mask);
cv::Mat roi_mask(cv::Size size, const cv::Mat &mask, std::optional<int> height);

Features features(const cv::Mat &gray, const cv::Mat &mask);
double ncc(const std::vector<double> &x, const std::vector<double> &y);
Json score(const Features &src, const Features &dst, const Matrix &H, const Config &cfg);
std::pair<Matrix, bool> refine(const Features &src, const Features &dst, const Matrix &H,
                               const Config &cfg);
cv::Mat correlate(const cv::Mat &a, const cv::Mat &b, cv::Size fft_size);
std::vector<cv::Vec3d> translation_peaks(const cv::Mat &source, const cv::Mat &target,
                                         const cv::Mat &sm, const cv::Mat &tm, const Config &cfg);
Json periods(const cv::Mat &feature, const cv::Mat &mask);
Json select_templates(const std::vector<cv::Mat> &templates, const cv::Mat &target,
                      const Config &cfg, const std::vector<cv::Mat> &template_valids,
                      const cv::Mat &target_valid, const std::string &label_mode);

ObjectPrediction match_ordered_objects(const std::vector<cv::Mat> &images,
                                       const std::vector<std::vector<ObjectRef>> &references,
                                       const std::vector<cv::Mat> &source_valids,
                                       const cv::Mat &target, const cv::Mat &target_valid,
                                       const std::vector<std::optional<Matrix>> &matrices,
                                       const Config &cfg);
void complete_object_support(ObjectPrediction &prediction, const std::vector<cv::Mat> &images,
                             const std::vector<std::vector<ObjectRef>> &references,
                             const std::vector<cv::Mat> &source_valids, const cv::Mat &target,
                             const cv::Mat &target_valid,
                             const std::vector<cv::Vec2d> &target_vectors, const Config &cfg);
std::vector<cv::Vec2d> anchor_vectors(const cv::Mat &labels, const cv::Mat &valid, int class_id);
void complete_periodic(MatchResult &result, const cv::Mat &labels, const cv::Mat &annotation_valid,
                       const cv::Mat &source_valid, const cv::Mat &target_valid, const Config &cfg);
Json evaluate_segmentation(const cv::Mat &prediction, const cv::Mat &truth,
                           const cv::Mat &prediction_valid, const cv::Mat &truth_valid,
                           bool include_background = false);
MatchResult match(const Inputs &inputs, const Config &cfg);
} // namespace ptm
