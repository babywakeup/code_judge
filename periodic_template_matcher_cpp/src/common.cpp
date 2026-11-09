#include "ptm/core.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <set>
#include <stdexcept>

namespace ptm {
cv::Mat binary(const cv::Mat &a) {
    if (a.empty())
        return {};
    cv::Mat b;
    cv::compare(a, 0, b, cv::CMP_GT);
    b /= 255;
    return b;
}
cv::Mat gray(const cv::Mat &image) {
    if (image.empty() || image.depth() != CV_8U || image.rows < 32 || image.cols < 32 ||
        (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
        throw std::invalid_argument("Expected uint8 gray/BGR/BGRA image at least 32x32");
    if (image.channels() == 1)
        return image;
    cv::Mat g;
    cv::cvtColor(image, g, image.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
    return g;
}
cv::Mat warp(const cv::Mat &a, const Matrix &H, cv::Size size, int interpolation) {
    cv::Mat b;
    cv::warpPerspective(a, b, cv::Mat(H), size, interpolation, cv::BORDER_CONSTANT, 0);
    return b;
}
cv::Mat warp_mask(const cv::Mat &a, const Matrix &H, cv::Size size) {
    for (double v : H.val)
        if (!std::isfinite(v))
            throw std::invalid_argument("Nonfinite transform");
    if (std::abs(cv::determinant(cv::Mat(H))) < 1e-10)
        throw std::invalid_argument("Singular transform");
    return binary(warp(binary(a), H, size, cv::INTER_NEAREST));
}
int pyround(double v) {
    if (!std::isfinite(v))
        throw std::invalid_argument("Cannot round a nonfinite value");
    const double rounded = std::nearbyint(v);
    if (rounded < std::numeric_limits<int>::min() || rounded > std::numeric_limits<int>::max())
        throw std::out_of_range("Rounded dimension exceeds the native integer range");
    return static_cast<int>(rounded);
}
double percentile(std::vector<double> values, double p) {
    if (values.empty())
        throw std::invalid_argument("Empty percentile");
    if (!std::isfinite(p) || p < 0 || p > 100)
        throw std::invalid_argument("Percentile must be in 0..100");
    std::sort(values.begin(), values.end());
    double i = (values.size() - 1) * p / 100.;
    auto lo = static_cast<std::size_t>(std::floor(i)), hi = static_cast<std::size_t>(std::ceil(i));
    return values[lo] + (values[hi] - values[lo]) * (i - lo);
}
std::vector<double> selected(const cv::Mat &image, const cv::Mat &mask) {
    if (image.size() != mask.size() || image.channels() != 1)
        throw std::invalid_argument("Selection size mismatch");
    cv::Mat d;
    image.convertTo(d, CV_64F);
    std::vector<double> values;
    values.reserve(cv::countNonZero(mask));
    for (int y = 0; y < d.rows; y++)
        for (int x = 0; x < d.cols; x++)
            if (mask.at<uchar>(y, x))
                values.push_back(d.at<double>(y, x));
    return values;
}
std::pair<cv::Mat, cv::Mat> resize_matching(const cv::Mat &image, const cv::Mat &mask,
                                            cv::Size size) {
    cv::Mat weights;
    binary(mask).convertTo(weights, CV_32F);
    cv::Mat resized_mask;
    cv::resize(binary(mask), resized_mask, size, 0, 0, cv::INTER_NEAREST);
    if (cv::countNonZero(mask) == mask.total()) {
        cv::Mat out;
        cv::resize(image, out, size, 0, 0, cv::INTER_AREA);
        return {out, resized_mask};
    }
    cv::Mat imagef, numerator, denominator;
    image.convertTo(imagef, CV_32F);
    cv::resize(imagef.mul(weights), numerator, size, 0, 0, cv::INTER_AREA);
    cv::resize(weights, denominator, size, 0, 0, cv::INTER_AREA);
    cv::Mat out(size, CV_8U, cv::Scalar(0));
    for (int y = 0; y < size.height; y++)
        for (int x = 0; x < size.width; x++) {
            float den = denominator.at<float>(y, x);
            if (den > 1e-6)
                out.at<uchar>(y, x) =
                    cv::saturate_cast<uchar>(pyround(numerator.at<float>(y, x) / den));
            else
                resized_mask.at<uchar>(y, x) = 0;
        }
    return {out, resized_mask};
}
double matching_scale(const Config &cfg, const std::vector<cv::Size> &sizes) {
    if (cfg.contains("scale") && !cfg["scale"].is_null())
        return cfg["scale"].get<double>();
    int longest = 1;
    for (auto s : sizes)
        longest = std::max({longest, s.width, s.height});
    return std::min(1., cfg.value("max_side", 800) / double(longest));
}
Matrix scale_matrix(cv::Size original, cv::Size resized) {
    return Matrix(double(resized.width) / original.width, 0, 0, 0,
                  double(resized.height) / original.height, 0, 0, 0, 1);
}
Json matrix_json(const Matrix &H) {
    Json j = Json::array();
    for (int y = 0; y < 3; y++)
        j.push_back({H(y, 0), H(y, 1), H(y, 2)});
    return j;
}
Matrix matrix_from_json(const Json &v) {
    if (!v.is_array() || v.size() != 3)
        throw std::invalid_argument("Expected 3x3 matrix");
    Matrix m;
    for (int y = 0; y < 3; y++) {
        if (v[y].size() != 3)
            throw std::invalid_argument("Expected 3x3 matrix");
        for (int x = 0; x < 3; x++) {
            m(y, x) = v[y][x].get<double>();
            if (!std::isfinite(m(y, x)))
                throw std::invalid_argument("Nonfinite transform");
        }
    }
    return m;
}
Matrix result_matrix(const MatchResult &r) {
    int i = r.registration.at("ranking").at(0);
    return matrix_from_json(r.registration.at("matches").at(i).at("matrix_template_to_target"));
}
cv::Mat roi_mask(cv::Size size, const cv::Mat &mask, std::optional<int> height) {
    int end = height.value_or(size.height);
    if (end < 1 || end > size.height)
        throw std::invalid_argument("valid_height outside image");
    if (!mask.empty() && mask.size() != size)
        throw std::invalid_argument("Valid-mask size must match image");
    cv::Mat out = mask.empty() ? cv::Mat::ones(size, CV_8U) : binary(mask);
    if (end < size.height)
        out.rowRange(end, size.height).setTo(0);
    if (cv::countNonZero(out) < 1024)
        throw std::invalid_argument("Too few valid image pixels");
    return out;
}
Config default_config() {
    return {{"angles", {-2., 0., 2.}},
            {"scales", {.97, 1., 1.03}},
            {"max_side", 800},
            {"peaks_per_hypothesis", 4},
            {"refine_candidates", 5},
            {"min_overlap", .45},
            {"min_score", 60.},
            {"min_template_gap", 3.},
            {"min_candidate_gap", 2.},
            {"valid_height", nullptr},
            {"edge_tolerance_px", 2.},
            {"ecc_iterations", 70},
            {"seed", 7},
            {"label_mode", "semantic"},
            {"include_background", false},
            {"backend", "fft"},
            {"template_workers", 0},
            {"periodic_completion", false},
            {"period_vectors", Json::array()},
            {"periodic_anchor_class", 1},
            {"periodic_max_steps", 2},
            {"periodic_min_ncc", .75},
            {"periodic_min_edge", .6},
            {"class_periodicity", Json::object()},
            {"scale", nullptr},
            {"ordered_object_class", nullptr},
            {"object_order_axis", "x"},
            {"object_cycle", {"A", "B", "C"}},
            {"object_visible_count", nullptr},
            {"object_min_score", .7},
            {"object_min_gap", .03},
            {"object_min_visible", .15},
            {"object_candidates", 8},
            {"object_spacing_weight", .3}};
}
std::string repeat_axis(const Config &cfg) {
    return cfg.value("object_order_axis", std::string("x")) == "x" ? "y" : "x";
}
std::vector<std::string> periodic_axes(const Config &cfg, int class_id) {
    auto policies = cfg.value("class_periodicity", Json::object());
    if (policies.empty()) {
        if (!cfg.at("ordered_object_class").is_null() &&
            cfg["ordered_object_class"].get<int>() == class_id)
            return {repeat_axis(cfg)};
        return {"x", "y"};
    }
    auto key = std::to_string(class_id);
    if (!policies.contains(key))
        return {};
    std::vector<std::string> out;
    for (auto a : {"x", "y"})
        for (auto &v : policies[key])
            if (v == a)
                out.push_back(a);
    return out;
}
bool allows_shift(const Config &cfg, int class_id, const std::vector<cv::Vec2d> &vectors) {
    auto axes = periodic_axes(cfg, class_id);
    if (axes.empty())
        return false;
    if (axes.size() == 2)
        return true;
    int a = axes[0] == "x" ? 0 : 1;
    for (auto v : vectors)
        if (std::abs(v[a]) <= 0 || std::abs(v[1 - a]) > .15 * std::abs(v[a]))
            return false;
    return true;
}
void validate_config(const Config &c) {
    auto require = [](bool condition, const char *message) {
        if (!condition)
            throw std::invalid_argument(message);
    };
    auto integer = [&](const char *k, int lo, int hi) {
        require(c.at(k).is_number_integer(), "Expected integer config value");
        int v = c[k];
        require(v >= lo && v <= hi, "Config integer out of range");
    };
    auto range = [&](const char *k, double lo, double hi, bool lower_open = false) {
        require(c.at(k).is_number(), "Expected number");
        double v = c[k];
        require(std::isfinite(v) && v <= hi && (lower_open ? v > lo : v >= lo),
                "Config number out of range");
    };
    integer("template_workers", 0, 2147483647);
    integer("max_side", 96, 2147483647);
    integer("object_candidates", 1, 16);
    integer("periodic_max_steps", 1, 5);
    integer("periodic_anchor_class", 1, 65535);
    for (auto k : {"peaks_per_hypothesis", "refine_candidates", "ecc_iterations"})
        integer(k, 1, 2147483647);
    if (!c["valid_height"].is_null())
        integer("valid_height", 1, 2147483647);
    if (!c["scale"].is_null())
        range("scale", 0, 1, true);
    range("min_overlap", 0, 1, true);
    range("edge_tolerance_px", 0, 1e100, true);
    for (auto k : {"min_score", "min_template_gap", "min_candidate_gap"})
        range(k, 0, 100);
    for (auto k : {"periodic_min_ncc", "periodic_min_edge", "object_min_score", "object_min_gap",
                   "object_spacing_weight"})
        range(k, 0, 1);
    range("object_min_visible", 0, 1, true);
    for (auto k : {"periodic_completion", "include_background"})
        require(c[k].is_boolean(), "Expected boolean config");
    require(c["backend"] == "fft" || c["backend"] == "opencv", "backend must be fft/opencv");
    require(c["label_mode"] == "semantic" || c["label_mode"] == "instance", "Invalid label mode");
    require(c["object_order_axis"] == "x" || c["object_order_axis"] == "y",
            "Invalid object order axis");
    for (auto k : {"angles", "scales"}) {
        require(c[k].is_array() && !c[k].empty(), "Nonempty angles/scales required");
        for (auto v : c[k]) {
            require(v.is_number() && std::isfinite(v.get<double>()), "Invalid angle/scale");
            if (std::string(k) == "scales")
                require(v.get<double>() > 0, "scales must be positive");
        }
    }
    require(c["object_cycle"].is_array() && !c["object_cycle"].empty() &&
                c["object_cycle"].size() <= 65535,
            "Invalid object cycle");
    std::set<std::string> names;
    for (auto v : c["object_cycle"]) {
        require(v.is_string() && !v.get<std::string>().empty(), "Invalid identity");
        require(names.insert(v.get<std::string>()).second, "Duplicate identity");
    }
    if (!c["object_visible_count"].is_null())
        integer("object_visible_count", 1, int(names.size()));
    require(c["class_periodicity"].is_object(), "class_periodicity must be object");
    for (auto it = c["class_periodicity"].begin(); it != c["class_periodicity"].end(); ++it) {
        int id = 0;
        try {
            id = std::stoi(it.key());
        } catch (...) {
            throw std::invalid_argument("Invalid class ID");
        }
        require(id >= 1 && id <= 65535 && std::to_string(id) == it.key(), "Invalid class ID");
        require(it.value().is_array(), "Invalid repeat axes");
        std::set<std::string> seen;
        for (auto v : it.value()) {
            require(v == "x" || v == "y", "Invalid repeat axis");
            require(seen.insert(v.get<std::string>()).second, "Duplicate repeat axis");
        }
    }
    if (!c["ordered_object_class"].is_null()) {
        integer("ordered_object_class", 1, 65535);
        require(c["label_mode"] == "semantic", "Ordered objects require semantic labels");
        auto axes = periodic_axes(c, c["ordered_object_class"].get<int>());
        require(std::find(axes.begin(), axes.end(), c["object_order_axis"].get<std::string>()) ==
                    axes.end(),
                "Cannot repeat along identity order axis");
    }
    require(c["period_vectors"].is_array() && c["period_vectors"].size() <= 2,
            "Invalid period vectors");
    std::vector<cv::Vec2d> vectors;
    for (auto v : c["period_vectors"]) {
        require(v.is_array() && v.size() == 2, "Invalid period vector");
        double x = v[0], y = v[1];
        require(std::isfinite(x) && std::isfinite(y) && x * x + y * y >= 1,
                "Invalid period vector");
        vectors.emplace_back(x, y);
    }
    if (vectors.size() == 2)
        require(std::abs(vectors[0][0] * vectors[1][1] - vectors[0][1] * vectors[1][0]) >=
                    .05 * cv::norm(vectors[0]) * cv::norm(vectors[1]),
                "Dependent period vectors");
}
} // namespace ptm
