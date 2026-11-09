#include "ptm/core.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <set>

namespace ptm {
Config object_transpose_config(Config);
Json object_transpose_diagnostics(const Json &);
ObjectPrediction object_transpose_prediction(const ObjectPrediction &);
namespace {
cv::Mat trans(const cv::Mat &m) {
    cv::Mat o;
    cv::transpose(m, o);
    return o;
}
cv::Mat dilated(const cv::Mat &m, int n) {
    cv::Mat o;
    cv::dilate(m, o, cv::Mat::ones(n, n, CV_8U));
    return o;
}
cv::Size scaled(cv::Size s, double f) {
    return {pyround(s.width * f), pyround(s.height * f)};
}
cv::Rect bounds(const cv::Mat &m) {
    std::vector<cv::Point> p;
    cv::findNonZero(m, p);
    return p.empty() ? cv::Rect() : cv::boundingRect(p);
}
double stddev(const std::vector<double> &v) {
    if (v.empty())
        return 0;
    double s = 0;
    for (auto a : v)
        s += a;
    double mean = s / v.size();
    s = 0;
    for (auto a : v)
        s += (a - mean) * (a - mean);
    return std::sqrt(s / v.size());
}
cv::Mat white_occlusions(const cv::Mat &g, const cv::Mat &valid) {
    cv::Mat w = binary(g >= 230) & valid, components, stats, centers;
    int n = cv::connectedComponentsWithStats(w, components, stats, centers);
    cv::Mat result = cv::Mat::zeros(g.size(), CV_8U);
    for (int i = 1; i < n; ++i)
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= 64)
            result |= binary(components == i);
    return result;
}
cv::Mat candidate_fill(const cv::Mat &mapped, const cv::Mat &annotation, const cv::Mat &target,
                       const cv::Mat &labels, const cv::Mat &known) {
    cv::Mat eligible = annotation & target, unknown = eligible & (1 - known),
            fill = unknown.clone(), positive = binary(mapped), seeds = unknown & positive,
            components;
    int n = cv::connectedComponents(positive & eligible, components);
    for (int i = 1; i < n; ++i) {
        cv::Mat component = binary(components == i);
        if (cv::countNonZero(component & seeds))
            fill |= component & known & (1 - binary(labels));
    }
    return fill;
}
} // namespace
void complete_object_support(ObjectPrediction &prediction, const std::vector<cv::Mat> &images,
                             const std::vector<std::vector<ObjectRef>> &references,
                             const std::vector<cv::Mat> &source_valids, const cv::Mat &target,
                             const cv::Mat &target_valid,
                             const std::vector<cv::Vec2d> &target_vectors, const Config &cfg) {
    if (references.size() != images.size() || source_valids.size() != images.size())
        throw std::invalid_argument("One reference list and valid mask per image required");
    if (cfg.value("object_order_axis", std::string("x")) == "y") {
        auto p = object_transpose_prediction(prediction);
        std::vector<cv::Mat> ims, vs;
        auto refs = references;
        auto vectors = target_vectors;
        for (size_t i = 0; i < images.size(); ++i) {
            ims.push_back(trans(images[i]));
            vs.push_back(trans(source_valids[i]));
            for (auto &r : refs[i])
                r.mask = trans(r.mask);
        }
        for (auto &v : vectors)
            std::swap(v[0], v[1]);
        complete_object_support(p, ims, refs, vs, trans(target), trans(target_valid), vectors,
                                object_transpose_config(cfg));
        prediction = object_transpose_prediction(p);
        return;
    }
    auto status = prediction.diagnostics.value("status", std::string());
    if (status != "accepted" && status != "partial_reference")
        return;
    int class_id = cfg.at("ordered_object_class").get<int>();
    std::vector<cv::Vec2d> vectors;
    for (auto v : target_vectors)
        if (std::abs(v[1]) >= 12 && std::abs(v[0]) <= .15 * std::abs(v[1]) &&
            allows_shift(cfg, class_id, {v}))
            vectors.push_back(v);
    prediction.diagnostics["support_completion"] = {{"added_valid_pixels", 0},
                                                    {"added_foreground_pixels", 0},
                                                    {"copies", Json::array()},
                                                    {"repeat_axes", periodic_axes(cfg, class_id)}};
    auto &info = prediction.diagnostics["support_completion"];
    if (vectors.empty() || cv::countNonZero(target_valid & (1 - prediction.valid)) == 0)
        return;
    cv::Size shape = target.size();
    std::vector<cv::Size> sizes{shape};
    for (auto &i : images)
        sizes.push_back(i.size());
    double scale = matching_scale(cfg, sizes);
    auto ts = scaled(shape, scale);
    Matrix db = scale_matrix(shape, ts);
    auto tg = gray(target), occluded = white_occlusions(tg, target_valid);
    info["occluded_pixels"] = cv::countNonZero(occluded);
    cv::Mat ev = target_valid & (1 - occluded);
    if (!cv::countNonZero(ev))
        return;
    auto t = resize_matching(tg, ev, ts);
    auto dst = features(t.first, t.second);
    std::map<int, std::pair<Features, Matrix>> cache;
    Config local_cfg = cfg;
    local_cfg["min_overlap"] = .2;
    std::vector<std::string> cycle = cfg.at("object_cycle").get<std::vector<std::string>>();
    for (auto &obj : prediction.diagnostics["objects"]) {
        int ti = obj["template_index"], ri = obj["reference_index"];
        if (obj.value("reference_partial", false))
            continue;
        auto &ref = references[ti][ri];
        cv::Mat raw = binary(ref.mask) & source_valids[ti];
        if (!cache.count(ti)) {
            auto ss = scaled(images[ti].size(), scale);
            auto s = resize_matching(gray(images[ti]), source_valids[ti], ss);
            cache[ti] = {features(s.first, s.second), scale_matrix(images[ti].size(), ss)};
        }
        auto &src = cache[ti].first;
        auto da = cache[ti].second;
        Matrix H = matrix_from_json(obj["matrix_template_to_target"]);
        std::string name = obj["identity"];
        int identity = int(std::find(cycle.begin(), cycle.end(), name) - cycle.begin() + 1);
        for (auto vector : vectors)
            for (int n = 1; n <= cfg.value("periodic_max_steps", 3); ++n)
                for (int sign : {1, -1}) {
                    int step = n * sign;
                    Matrix shift = Matrix::eye();
                    shift(0, 2) = step * vector[0];
                    shift(1, 2) = step * vector[1];
                    Matrix candidate = shift * H;
                    cv::Mat mapped;
                    warp_mask(raw, candidate, shape).convertTo(mapped, CV_16U, class_id);
                    cv::Mat own = binary(prediction.identities == identity) | binary(mapped);
                    auto r = bounds(own);
                    if (r.empty())
                        continue;
                    cv::Mat band = cv::Mat::zeros(shape, CV_8U);
                    band.colRange(r.x, r.x + r.width).setTo(1);
                    cv::Mat annotation = warp_mask(source_valids[ti], candidate, shape) & band;
                    if (!ref.background_known)
                        annotation &= binary(mapped);
                    if (!cv::countNonZero(annotation & target_valid & (1 - prediction.valid)))
                        continue;
                    cv::Mat fill = candidate_fill(mapped, annotation, target_valid,
                                                  prediction.labels, prediction.valid) &
                                   (1 - binary(prediction.labels)),
                            components;
                    int count = cv::connectedComponents(fill, components);
                    auto observed_source = warp(gray(images[ti]), candidate, shape);
                    Matrix work = db * candidate * da.inv();
                    auto work_source = warp(src.norm, work, ts),
                         work_support = warp_mask(src.mask, work, ts);
                    Json record = {{"identity", name},
                                   {"step", step},
                                   {"target_vector", Json::array({vector[0], vector[1]})},
                                   {"matrix_template_to_target", matrix_json(candidate)},
                                   {"regions", Json::array()}};
                    cv::Mat approved = cv::Mat::zeros(shape, CV_8U),
                            approved_labels = cv::Mat::zeros(shape, CV_16U);
                    std::vector<std::pair<cv::Mat, bool>> pending;
                    for (int i = 1; i < count; ++i)
                        pending.emplace_back(binary(components == i), false);
                    int tile_size = std::max(16, pyround(32 / scale));
                    while (!pending.empty()) {
                        auto item = pending.back();
                        pending.pop_back();
                        cv::Mat region = item.first;
                        bool is_tile = item.second;
                        cv::Mat local;
                        cv::resize(region, local, ts, 0, 0, cv::INTER_NEAREST);
                        local = dilated(local, 25) & dst.mask;
                        Features dlocal = dst;
                        dlocal.mask = local;
                        Json metrics = score(src, dlocal, work, local_cfg);
                        cv::Mat evidence = region & (1 - occluded);
                        auto a = selected(observed_source, evidence), b = selected(tg, evidence);
                        int pixels = cv::countNonZero(region),
                            evidence_pixels = cv::countNonZero(evidence);
                        bool evidence_ok = a.size() >= std::max(64., .5 * pixels) &&
                                           stddev(a) >= 2 && stddev(b) >= 2;
                        double region_ncc = evidence_ok ? ncc(a, b) : 0.;
                        bool accepted = metrics["structure_ncc"].get<double>() >=
                                            cfg.value("periodic_min_ncc", .75) &&
                                        metrics["edge_alignment"].get<double>() >=
                                            cfg.value("periodic_min_edge", .6) &&
                                        region_ncc >= cfg.value("periodic_min_ncc", .75);
                        cv::Mat region_mapped = mapped;
                        Json corrected = nullptr;
                        cv::Mat ecc_mask = local & work_support;
                        if (!accepted && evidence_ok && cv::countNonZero(ecc_mask) >= 1000) {
                            cv::Mat w = cv::Mat::eye(2, 3, CV_32F);
                            try {
                                cv::findTransformECC(
                                    dst.norm, work_source, w, cv::MOTION_TRANSLATION,
                                    cv::TermCriteria(cv::TermCriteria::COUNT |
                                                         cv::TermCriteria::EPS,
                                                     cfg.value("ecc_iterations", 70), 1e-5),
                                    ecc_mask * 255, 5);
                                if (std::hypot(w.at<float>(0, 2), w.at<float>(1, 2)) <= 6) {
                                    Matrix wm = Matrix::eye();
                                    for (int y = 0; y < 2; ++y)
                                        for (int x = 0; x < 3; ++x)
                                            wm(y, x) = w.at<float>(y, x);
                                    Matrix refined = db.inv() * wm.inv() * db * candidate;
                                    auto support = warp_mask(source_valids[ti], refined, shape);
                                    if (cv::countNonZero(region & (1 - support)) == 0) {
                                        auto observed = warp(gray(images[ti]), refined, shape);
                                        auto values = selected(observed, evidence);
                                        double rn = stddev(values) >= 2 ? ncc(values, b) : 0;
                                        auto rm =
                                            score(src, dlocal, db * refined * da.inv(), local_cfg);
                                        if (rn >= cfg.value("periodic_min_ncc", .75) &&
                                            rm["structure_ncc"].get<double>() >=
                                                cfg.value("periodic_min_ncc", .75) &&
                                            rm["edge_alignment"].get<double>() >=
                                                cfg.value("periodic_min_edge", .6)) {
                                            cv::Mat corrected_mapped;
                                            warp_mask(raw, refined, shape)
                                                .convertTo(corrected_mapped, CV_16U, class_id);
                                            region_mapped = corrected_mapped;
                                            if (!ref.background_known &&
                                                cv::countNonZero(region &
                                                                 (1 - binary(region_mapped))))
                                                continue;
                                            accepted = true;
                                            corrected = matrix_json(refined);
                                            metrics = rm;
                                            region_ncc = rn;
                                        }
                                    }
                                }
                            } catch (const cv::Exception &) {
                            }
                        }
                        record["regions"].push_back(
                            {{"accepted", accepted},
                             {"pixels", pixels},
                             {"evidence_pixels", evidence_pixels},
                             {"occluded_pixels", cv::countNonZero(region & occluded)},
                             {"region_ncc", region_ncc},
                             {"local_tile", is_tile},
                             {"metrics", metrics},
                             {"corrected_matrix", corrected}});
                        if (accepted) {
                            approved |= region;
                            region_mapped.copyTo(approved_labels, region);
                            if (!corrected.is_null()) {
                                if (!obj.contains("supporting_transforms"))
                                    obj["supporting_transforms"] = Json::array();
                                obj["supporting_transforms"].push_back(corrected);
                            }
                        } else if (!is_tile) {
                            auto rb = bounds(region);
                            if (std::max(rb.width, rb.height) > tile_size)
                                for (int y = rb.y; y < rb.y + rb.height; y += tile_size)
                                    for (int x = rb.x; x < rb.x + rb.width; x += tile_size) {
                                        cv::Rect rr(x, y, std::min(tile_size, shape.width - x),
                                                    std::min(tile_size, shape.height - y));
                                        if (!cv::countNonZero(region(rr)))
                                            continue;
                                        cv::Mat tile = cv::Mat::zeros(shape, CV_8U);
                                        region(rr).copyTo(tile(rr));
                                        pending.emplace_back(tile, true);
                                    }
                        }
                    }
                    bool any = cv::countNonZero(approved) > 0;
                    record["accepted"] = any;
                    info["copies"].push_back(record);
                    if (!any)
                        continue;
                    cv::Mat positive = approved & binary(approved_labels);
                    info["added_valid_pixels"] =
                        info["added_valid_pixels"].get<int>() +
                        cv::countNonZero(approved & (1 - prediction.valid));
                    info["added_foreground_pixels"] =
                        info["added_foreground_pixels"].get<int>() + cv::countNonZero(positive);
                    approved_labels.copyTo(prediction.labels, approved);
                    prediction.identities.setTo(identity, positive);
                    prediction.valid |= approved;
                    if (!obj.contains("supporting_transforms"))
                        obj["supporting_transforms"] = Json::array();
                    obj["supporting_transforms"].push_back(matrix_json(candidate));
                }
    }
    prediction.diagnostics["valid_coverage_roi"] =
        double(cv::countNonZero(prediction.valid & target_valid)) /
        std::max(cv::countNonZero(target_valid), 1);
}
} // namespace ptm
