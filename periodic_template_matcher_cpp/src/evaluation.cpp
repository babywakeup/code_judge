#include "ptm/core.hpp"
#include <set>
#include <stdexcept>
namespace ptm {
namespace {
Json metrics(const cv::Mat &p, const cv::Mat &t, const cv::Mat &known, const cv::Mat &roi,
             bool bg) {
    std::set<int> ids;
    int count = 0, correct = 0;
    for (int y = 0; y < t.rows; y++)
        for (int x = 0; x < t.cols; x++)
            if (roi.at<uchar>(y, x)) {
                ++count;
                int a = t.at<uint16_t>(y, x), b = p.at<uint16_t>(y, x);
                ids.insert(a);
                if (known.at<uchar>(y, x)) {
                    ids.insert(b);
                    correct += a == b;
                }
            }
    Json per = Json::array();
    double si = 0, sd = 0;
    int n = 0;
    for (int id : ids) {
        if (!bg && id == 0)
            continue;
        int tp = 0, fp = 0, fn = 0;
        for (int y = 0; y < t.rows; y++)
            for (int x = 0; x < t.cols; x++)
                if (roi.at<uchar>(y, x)) {
                    bool a = t.at<uint16_t>(y, x) == id,
                         b = known.at<uchar>(y, x) && p.at<uint16_t>(y, x) == id;
                    tp += a && b;
                    fp += b && !a;
                    fn += a && !b;
                }
        int u = tp + fp + fn;
        Json i = nullptr, d = nullptr;
        if (u) {
            i = double(tp) / u;
            d = 2. * tp / (2 * tp + fp + fn);
            si += i.get<double>();
            sd += d.get<double>();
            ++n;
        }
        per.push_back(
            {{"class_id", id}, {"tp", tp}, {"fp", fp}, {"fn", fn}, {"iou", i}, {"dice", d}});
    }
    return {{"mean_iou", n ? Json(si / n) : Json(nullptr)},
            {"mean_dice", n ? Json(sd / n) : Json(nullptr)},
            {"pixel_accuracy", count ? Json(double(correct) / count) : Json(nullptr)},
            {"evaluated_pixels", count},
            {"per_class", per}};
}
} // namespace
Json evaluate_segmentation(const cv::Mat &prediction, const cv::Mat &truth,
                           const cv::Mat &prediction_valid, const cv::Mat &truth_valid,
                           bool include_background) {
    if (prediction.size() != truth.size() || prediction_valid.size() != truth.size() ||
        truth_valid.size() != truth.size())
        throw std::invalid_argument(
            "Prediction, truth and validity masks must have identical shapes");
    cv::Mat p, t;
    prediction.convertTo(p, CV_16U);
    truth.convertTo(t, CV_16U);
    auto known = binary(prediction_valid), roi = binary(truth_valid);
    Json out = metrics(p, t, known, roi, include_background);
    int count = out["evaluated_pixels"], covered = cv::countNonZero(known & roi);
    out["status"] = count ? "evaluated" : "no_valid_ground_truth";
    out["score"] =
        out["mean_iou"].is_null() ? Json(nullptr) : Json(100 * out["mean_iou"].get<double>());
    out["score_definition"] =
        "100 * mean_iou over ground-truth ROI; unknown predictions are abstentions";
    out["include_background"] = include_background;
    out["prediction_coverage"] = count ? Json(double(covered) / count) : Json(nullptr);
    out["covered_pixels"] = covered;
    out["uncovered_pixels"] = count - covered;
    out["overlap_only"] = metrics(p, t, known, roi & known, include_background);
    return out;
}
} // namespace ptm
