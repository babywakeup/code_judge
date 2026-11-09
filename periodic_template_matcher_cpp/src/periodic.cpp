#include "ptm/core.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <opencv2/imgproc.hpp>
#include <set>
namespace ptm {
namespace {
cv::Mat eq(const cv::Mat &a, int value) {
    return binary(a == value);
}
cv::Mat neg(const cv::Mat &a) {
    return 1 - a;
}
Json vector_json(const std::vector<cv::Vec2d> &v) {
    Json a = Json::array();
    for (auto x : v)
        a.push_back({x[0], x[1]});
    return a;
}
cv::Vec2d apply(const Matrix &H, cv::Vec2d v) {
    return {H(0, 0) * v[0] + H(0, 1) * v[1], H(1, 0) * v[0] + H(1, 1) * v[1]};
}
bool acceptable(const Json &m, const Config &c) {
    return m["structure_ncc"].get<double>() >= c["periodic_min_ncc"].get<double>() &&
           m["edge_alignment"].get<double>() >= c["periodic_min_edge"].get<double>();
}
cv::Mat candidate_fill(const cv::Mat &mapped, const cv::Mat &ann, const cv::Mat &target,
                       const cv::Mat &labels, const cv::Mat &known) {
    cv::Mat eligible = ann & target, unknown = eligible & neg(known), fill = unknown.clone(),
            seeds = unknown & binary(mapped > 0);
    std::set<int> ids;
    for (int y = 0; y < mapped.rows; y++)
        for (int x = 0; x < mapped.cols; x++)
            if (seeds.at<uchar>(y, x))
                ids.insert(mapped.at<uint16_t>(y, x));
    for (int id : ids) {
        cv::Mat cc;
        int n = cv::connectedComponents(eq(mapped, id) & eligible, cc);
        std::vector<bool> seeded(n, false);
        for (int y = 0; y < mapped.rows; y++)
            for (int x = 0; x < mapped.cols; x++)
                if (seeds.at<uchar>(y, x) && mapped.at<uint16_t>(y, x) == id)
                    seeded[cc.at<int>(y, x)] = true;
        for (int y = 0; y < mapped.rows; y++)
            for (int x = 0; x < mapped.cols; x++)
                if (seeded[cc.at<int>(y, x)] && cc.at<int>(y, x) > 0 && known.at<uchar>(y, x) &&
                    labels.at<uint16_t>(y, x) == 0)
                    fill.at<uchar>(y, x) = 1;
    }
    return fill;
}
} // namespace
std::vector<cv::Vec2d> anchor_vectors(const cv::Mat &labels, const cv::Mat &valid, int class_id) {
    cv::Mat cc, stats, centers;
    int n =
        cv::connectedComponentsWithStats(eq(labels, class_id) & binary(valid), cc, stats, centers);
    std::vector<cv::Vec2d> pts, out;
    for (int i = 1; i < n; i++)
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= 16)
            pts.emplace_back(centers.at<double>(i, 0), centers.at<double>(i, 1));
    for (int axis = 0; axis < 2; axis++) {
        std::vector<cv::Vec2d> d;
        double minimum = 1e100;
        for (auto a : pts)
            for (auto b : pts) {
                auto v = b - a;
                if (v[axis] >= 12 && std::abs(v[1 - axis]) <= .15 * v[axis]) {
                    d.push_back(v);
                    minimum = std::min(minimum, v[axis]);
                }
            }
        std::vector<double> x, y;
        for (auto v : d)
            if (v[axis] <= minimum * 1.15) {
                x.push_back(v[0]);
                y.push_back(v[1]);
            }
        if (x.size() >= 2)
            out.emplace_back(percentile(x, 50), percentile(y, 50));
    }
    return out;
}
void complete_periodic(MatchResult &r, const cv::Mat &labels, const cv::Mat &annotation_valid,
                       const cv::Mat &source_valid, const cv::Mat &target_valid,
                       const Config &cfg) {
    const auto &source = r.template_image;
    const auto &target = r.input_image;
    auto H = result_matrix(r);
    int w = target.cols, h = target.rows;
    const Json &best = r.registration["matches"][r.registration["ranking"][0].get<int>()];
    double scale = matching_scale(cfg, {source.size(), target.size()});
    cv::Size ss(pyround(source.cols * scale), pyround(source.rows * scale)),
        ts(pyround(w * scale), pyround(h * scale));
    Matrix Da = scale_matrix(source.size(), ss), Db = scale_matrix(target.size(), ts);
    auto image_vectors = [&]() {
        std::vector<cv::Vec2d> v;
        auto ws = best.value("working_target_size", Json::array({ts.width, ts.height}));
        for (int axis = 0; axis < 2; axis++) {
            Json chosen;
            for (auto c : best.value("lattice_candidates", Json::array())) {
                auto a = c["vector_work_px"];
                if (a[axis].get<double>() > 0 && a[1 - axis].get<double>() == 0 &&
                    (chosen.is_null() ||
                     a[axis].get<double>() < chosen["vector_work_px"][axis].get<double>()))
                    chosen = c;
            }
            if (!chosen.is_null()) {
                auto a = chosen["vector_work_px"];
                cv::Vec2d tv(a[0].get<double>() * w / ws[0].get<double>(),
                             a[1].get<double>() * h / ws[1].get<double>());
                cv::Matx22d linear(H(0, 0), H(0, 1), H(1, 0), H(1, 1));
                v.push_back(linear.inv() * tv);
            }
        }
        return v;
    };
    std::vector<cv::Vec2d> vectors;
    for (auto v : cfg["period_vectors"])
        vectors.emplace_back(v[0].get<double>(), v[1].get<double>());
    std::string period_source = "configured";
    if (vectors.empty()) {
        vectors = anchor_vectors(labels, annotation_valid, cfg["periodic_anchor_class"].get<int>());
        period_source = "anchor_class";
    }
    if (vectors.empty()) {
        vectors = image_vectors();
        period_source = "image_autocorrelation";
    }
    r.completed_template = r.warped_template.clone();
    r.completed_valid = r.geometric_valid.clone();
    r.periodic_valid = cv::Mat::zeros(h, w, CV_8U);
    r.label_periodic_valid = cv::Mat::zeros(h, w, CV_8U);
    int total = cv::countNonZero(target_valid), direct = cv::countNonZero(r.label_valid);
    Json info = {{"status", "incomplete"},
                 {"period_source", period_source},
                 {"period_vectors", vector_json(vectors)},
                 {"copies", Json::array()},
                 {"direct_label_pixels", direct},
                 {"target_valid_pixels", total}};
    std::set<int> classes;
    for (int y = 0; y < labels.rows; y++)
        for (int x = 0; x < labels.cols; x++)
            if (labels.at<uint16_t>(y, x) > 0)
                classes.insert(labels.at<uint16_t>(y, x));
    for (auto it = cfg["class_periodicity"].begin(); it != cfg["class_periodicity"].end(); ++it)
        classes.insert(std::stoi(it.key()));
    if (!cfg["ordered_object_class"].is_null())
        classes.erase(cfg["ordered_object_class"].get<int>());
    info["class_periodicity"] = Json::object();
    for (int c : classes)
        info["class_periodicity"][std::to_string(c)] = periodic_axes(cfg, c);
    auto finish = [&](std::string status = "") {
        int valid = cv::countNonZero(r.label_valid),
            added = cv::countNonZero(r.label_periodic_valid),
            missing = cv::countNonZero(target_valid & neg(r.label_valid));
        info["status"] = status.empty() ? (missing == 0 ? "complete" : "incomplete") : status;
        info["label_coverage_roi"] = double(valid) / total;
        info["geometric_coverage_roi"] = double(cv::countNonZero(r.completed_valid)) / total;
        info["added_label_pixels"] = added;
        info["newly_covered_pixels"] = valid - direct;
        info["boundary_foreground_pixels"] = added - valid + direct;
        info["remaining_unknown_pixels"] = missing;
        r.registration["periodic_completion"] = info;
    };
    bool needs = false;
    if (!cfg["ordered_object_class"].is_null()) {
        auto axes = periodic_axes(cfg, cfg["ordered_object_class"].get<int>());
        needs = std::find(axes.begin(), axes.end(), repeat_axis(cfg)) != axes.end();
    }
    if (cv::countNonZero(target_valid & neg(r.label_valid)) == 0 && !needs) {
        finish();
        return;
    }
    if (vectors.empty()) {
        finish("no_period");
        return;
    }
    auto a = resize_matching(gray(source), source_valid, ss),
         b = resize_matching(gray(target), target_valid, ts);
    Features src = features(a.first, a.second), dst = features(b.first, b.second);
    Config local_cfg = cfg;
    local_cfg["min_overlap"] = .2;
    if (period_source == "anchor_class") {
        Json validation = Json::array();
        std::vector<cv::Vec2d> supported;
        for (auto v : vectors) {
            auto t = Matrix::eye();
            auto delta = apply(Da, v);
            t(0, 2) = delta[0];
            t(1, 2) = delta[1];
            auto m = score(src, src, t, local_cfg);
            bool ok = acceptable(m, cfg);
            validation.push_back({{"vector", {v[0], v[1]}}, {"accepted", ok}, {"metrics", m}});
            if (ok)
                supported.push_back(v);
        }
        info["anchor_period_validation"] = validation;
        if (supported.size() != vectors.size()) {
            vectors = supported;
            for (auto v : image_vectors()) {
                bool independent = true;
                for (auto prev : vectors)
                    if (std::abs(v[0] * prev[1] - v[1] * prev[0]) <=
                        .15 * cv::norm(v) * cv::norm(prev))
                        independent = false;
                if (independent)
                    vectors.push_back(v);
                if (vectors.size() == 2)
                    break;
            }
            period_source =
                vectors.size() > supported.size() ? "image_autocorrelation" : "anchor_class";
            info["period_source"] = period_source;
            info["period_vectors"] = vector_json(vectors);
            info["period_fallback_reason"] = "anchor_spacing_not_supported_by_image";
        }
        if (vectors.empty()) {
            finish("no_period");
            return;
        }
    }
    std::vector<std::vector<int>> offsets;
    std::vector<int> tmp(vectors.size());
    int steps = cfg["periodic_max_steps"];
    std::function<void(int)> gen = [&](int i) {
        if (i == int(tmp.size())) {
            if (std::any_of(tmp.begin(), tmp.end(), [](int x) { return x != 0; }))
                offsets.push_back(tmp);
            return;
        }
        for (int n = -steps; n <= steps; n++) {
            tmp[i] = n;
            gen(i + 1);
        }
    };
    gen(0);
    std::sort(offsets.begin(), offsets.end(), [](auto a, auto b) {
        int x = 0, y = 0;
        for (int z : a)
            x += std::abs(z);
        for (int z : b)
            y += std::abs(z);
        return x == y ? a < b : x < y;
    });
    for (auto offset : offsets) {
        cv::Vec2d shift(0, 0);
        for (size_t i = 0; i < vectors.size(); i++)
            shift += vectors[i] * offset[i];
        Matrix P = Matrix::eye();
        P(0, 2) = shift[0];
        P(1, 2) = shift[1];
        Matrix candidate = H * P;
        auto ann = warp_mask(annotation_valid, candidate, target.size());
        if (!cv::countNonZero(ann & target_valid & neg(r.label_valid)) && !needs)
            continue;
        Matrix work = Db * candidate * Da.inv();
        auto initial = score(src, dst, work, local_cfg);
        auto refined = refine(src, dst, work, local_cfg);
        auto rm = score(src, dst, refined.first, local_cfg);
        std::vector<cv::Point2f> points = {{float(ss.width * .2), float(ss.height * .2)},
                                           {float(ss.width * .8), float(ss.height * .8)},
                                           {float(ss.width * .5), float(ss.height * .5)}},
                                 p1, p2;
        cv::perspectiveTransform(points, p1, cv::Mat(work));
        cv::perspectiveTransform(points, p2, cv::Mat(refined.first));
        double displacement = 0, limit = 16;
        for (int i = 0; i < 3; i++)
            displacement = std::max(displacement, cv::norm(p1[i] - p2[i]));
        for (auto v : vectors)
            limit = std::min(limit, cv::norm(apply(Db, apply(H, v))) * .1);
        Json m = initial;
        bool ecc = false;
        if (rm["match_score"].get<double>() >= initial["match_score"].get<double>() &&
            displacement <= limit) {
            candidate = Db.inv() * refined.first * Da;
            m = rm;
            ecc = refined.second;
        }
        bool accepted = acceptable(m, cfg);
        Json record = {
            {"offset", offset},       {"accepted", accepted},
            {"metrics", m},           {"geometry_accepted", accepted},
            {"ecc_converged", ecc},   {"matrix_template_to_target", matrix_json(candidate)},
            {"added_label_pixels", 0}};
        if (accepted) {
            cv::Mat mapped = warp(labels, candidate, target.size(), cv::INTER_NEAREST);
            ann = warp_mask(annotation_valid, candidate, target.size());
            std::vector<cv::Vec2d> active;
            for (size_t i = 0; i < vectors.size(); i++)
                if (offset[i])
                    active.push_back(vectors[i]);
            std::set<int> allowed, blocked;
            for (int c : classes)
                (allows_shift(cfg, c, active) ? allowed : blocked).insert(c);
            record["blocked_class_ids"] = blocked;
            cv::Mat eligible = cv::Mat::zeros(h, w, CV_8U);
            for (int c : allowed)
                eligible |= eq(mapped, c);
            if (blocked.empty())
                eligible |= eq(mapped, 0);
            ann &= eligible;
            auto fill = candidate_fill(mapped, ann, target_valid, r.mapped_labels, r.label_valid);
            cv::Mat components;
            int count = cv::connectedComponents(fill, components);
            cv::Mat approved = cv::Mat::zeros(h, w, CV_8U);
            record["local_regions"] = Json::array();
            for (int c = 1; c < count; c++) {
                cv::Mat region = eq(components, c), local;
                cv::resize(region, local, ts, 0, 0, cv::INTER_NEAREST);
                cv::dilate(local, local, cv::Mat::ones(25, 25, CV_8U));
                local &= dst.mask;
                Features ld = dst;
                ld.mask = local;
                auto lm = score(src, ld, Db * candidate * Da.inv(), local_cfg);
                bool ok = acceptable(lm, cfg);
                record["local_regions"].push_back(
                    {{"accepted", ok}, {"pixels", cv::countNonZero(region)}, {"metrics", lm}});
                if (ok)
                    approved |= region;
            }
            fill = approved;
            record["accepted"] = cv::countNonZero(fill) > 0;
            if (cv::countNonZero(fill)) {
                record["newly_covered_pixels"] = cv::countNonZero(fill & neg(r.label_valid));
                record["boundary_foreground_pixels"] = cv::countNonZero(fill & r.label_valid);
                mapped.copyTo(r.mapped_labels, fill);
                r.label_valid |= fill;
                r.label_periodic_valid |= fill;
                auto geom = warp_mask(source_valid, candidate, target.size());
                cv::Mat fill_image = geom & target_valid & neg(r.completed_valid) & approved;
                warp(source, candidate, target.size()).copyTo(r.completed_template, fill_image);
                r.completed_valid |= fill_image;
                r.periodic_valid |= fill_image;
                record["added_label_pixels"] = cv::countNonZero(fill);
            }
        }
        info["copies"].push_back(record);
        if (accepted && record["accepted"].get<bool>() &&
            cv::countNonZero(target_valid & neg(r.label_valid)) == 0)
            break;
    }
    finish();
}
} // namespace ptm
