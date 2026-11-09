#include "ptm/core.hpp"
#include <opencv2/imgproc.hpp>

namespace ptm {
namespace {
std::pair<cv::Mat, cv::Mat> transfer(const Inputs &in, MatchResult &out, int index, const Matrix &H,
                                     const Config &cfg) {
    cv::Mat labels = in.labels.at(index).clone();
    cv::Mat source_roi = in.template_valids.at(index).clone();
    if (!in.annotation_valids.at(index).empty())
        source_roi = source_roi.mul(binary(in.annotation_valids[index]));
    if (in.has_objects)
        labels.setTo(0, labels == cfg.at("ordered_object_class").get<int>());
    out.mapped_labels = warp(labels, H, in.target.size(), cv::INTER_NEAREST);
    out.label_valid = warp_mask(source_roi, H, in.target.size()).mul(in.target_valid);
    out.mapped_labels.setTo(0, 1 - out.label_valid);
    return {labels, source_roi};
}
void supplemental(const Inputs &in, MatchResult &result, int selected_index, const Config &cfg) {
    Json records = Json::array();
    for (auto &entry : result.registration["ranking"]) {
        if (cv::countNonZero(in.target_valid.mul(1 - result.label_valid)) == 0)
            break;
        int index = entry;
        const auto &candidate = result.registration["matches"][index];
        if (index == selected_index || in.labels[index].empty() ||
            !candidate.value("geometry_accepted", false) ||
            !candidate.contains("matrix_template_to_target") ||
            candidate["matrix_template_to_target"].is_null())
            continue;
        Matrix H = matrix_from_json(candidate["matrix_template_to_target"]);
        MatchResult other;
        other.registration = result.registration;
        other.registration["ranking"] = {index};
        other.registration["best_template_index"] = index;
        other.registration.erase("periodic_completion");
        other.input_image = in.target;
        other.template_image = in.templates[index];
        other.warped_template = warp(other.template_image, H, in.target.size());
        other.geometric_valid =
            warp_mask(in.template_valids[index], H, in.target.size()).mul(in.target_valid);
        auto transferred = transfer(in, other, index, H, cfg);
        complete_periodic(other, transferred.first, transferred.second, in.template_valids[index],
                          in.target_valid, cfg);
        cv::Mat fill = other.label_valid.mul(in.target_valid).mul(1 - result.label_valid);
        other.mapped_labels.copyTo(result.mapped_labels, fill);
        result.label_valid |= fill;
        result.label_periodic_valid |= other.label_periodic_valid.mul(fill);
        other.completed_template.copyTo(result.completed_template, fill);
        result.completed_valid |= fill.mul(other.completed_valid);
        result.periodic_valid |= fill.mul(other.periodic_valid);
        records.push_back({{"template_index", index},
                           {"added_label_pixels", cv::countNonZero(fill)},
                           {"periodic_completion", other.registration["periodic_completion"]}});
    }
    if (!records.empty()) {
        int missing = cv::countNonZero(in.target_valid.mul(1 - result.label_valid));
        auto &info = result.registration["periodic_completion"];
        info["supplemental_templates"] = records;
        info["remaining_unknown_pixels"] = missing;
        info["label_coverage_roi"] =
            double(cv::countNonZero(result.label_valid)) / cv::countNonZero(in.target_valid);
        info["status"] = missing == 0 ? "complete" : "incomplete";
    }
}
void fuse(const Inputs &in, MatchResult &r, ObjectPrediction &p, const Config &cfg) {
    r.base_labels = r.mapped_labels.clone();
    r.base_valid = r.label_valid.clone();
    cv::Mat foreground = r.base_valid.mul(binary(r.base_labels));
    cv::Mat overlap = binary(p.labels).mul(foreground);
    p.diagnostics["foreground_conflict_pixels"] = cv::countNonZero(overlap);
    bool verified =
        !cfg["object_visible_count"].is_null() &&
        (p.diagnostics["status"] == "accepted" || p.diagnostics["status"] == "partial_reference");
    cv::Mat protected_mask = verified ? cv::Mat::zeros(foreground.size(), CV_8U) : foreground;
    cv::Mat positive = binary(p.labels).mul(1 - protected_mask).mul(in.target_valid);
    p.diagnostics["instance_priority_pixels"] = cv::countNonZero(positive.mul(foreground));
    p.labels.copyTo(r.mapped_labels, positive);
    r.object_identities = cv::Mat::zeros(in.target.size(), CV_16U);
    p.identities.copyTo(r.object_identities, positive);
    r.object_valid = p.valid;
    r.label_valid = ((r.label_valid & p.valid) | positive | foreground).mul(in.target_valid);
    r.mapped_labels.setTo(0, 1 - r.label_valid);
    if (!r.label_periodic_valid.empty())
        r.label_periodic_valid = r.label_periodic_valid.mul(r.label_valid).mul(1 - positive);
    if (r.registration.contains("periodic_completion")) {
        int missing = cv::countNonZero(in.target_valid.mul(1 - r.label_valid));
        auto &info = r.registration["periodic_completion"];
        info["status"] = missing == 0 ? "complete" : "incomplete";
        info["remaining_unknown_pixels"] = missing;
        info["label_coverage_roi"] =
            double(cv::countNonZero(r.label_valid)) / cv::countNonZero(in.target_valid);
    }
}
} // namespace
MatchResult match(const Inputs &in, const Config &cfg) {
    validate_config(cfg);
    if (in.templates.empty() || in.labels.size() != in.templates.size() ||
        in.template_valids.size() != in.templates.size() ||
        in.annotation_valids.size() != in.templates.size())
        throw std::invalid_argument("Mismatched prepared input counts");
    Config registration_cfg = cfg;
    registration_cfg["valid_height"] = nullptr;
    MatchResult r;
    r.registration = select_templates(in.templates, in.target, registration_cfg, in.template_valids,
                                      in.target_valid, cfg.at("label_mode"));
    r.registration["config"] = cfg;
    r.registration["valid_regions"] = in.regions;
    int index = r.registration.at("ranking").at(0);
    r.input_image = in.target;
    r.template_image = in.templates[index];
    r.geometric_valid = cv::Mat::zeros(in.target.size(), CV_8U);
    const auto &mj = r.registration["matches"][index]["matrix_template_to_target"];
    std::optional<Matrix> H;
    if (!mj.is_null())
        H = matrix_from_json(mj);
    bool partial = !cfg["object_visible_count"].is_null() && in.has_objects;
    if (!H && !partial)
        return r;
    if (H) {
        r.warped_template = warp(r.template_image, *H, in.target.size());
        r.geometric_valid =
            warp_mask(in.template_valids[index], *H, in.target.size()).mul(in.target_valid);
    }
    std::optional<ObjectPrediction> prediction;
    std::vector<cv::Mat> source_valids;
    if (in.has_objects && cfg["label_mode"] == "semantic") {
        std::vector<std::optional<Matrix>> matrices;
        for (std::size_t i = 0; i < in.templates.size(); i++) {
            source_valids.push_back(
                in.annotation_valids[i].empty()
                    ? in.template_valids[i]
                    : in.template_valids[i].mul(binary(in.annotation_valids[i])));
            const auto &candidate = r.registration["matches"][i];
            if (candidate.value("geometry_accepted", r.registration["accepted"].get<bool>()) &&
                !candidate["matrix_template_to_target"].is_null())
                matrices.push_back(matrix_from_json(candidate["matrix_template_to_target"]));
            else
                matrices.push_back(std::nullopt);
        }
        prediction = match_ordered_objects(in.templates, in.objects, source_valids, in.target,
                                           in.target_valid, matrices, cfg);
        r.registration["ordered_objects"] = prediction->diagnostics;
        if (!r.registration["accepted"].get<bool>() &&
            r.registration["status"] == "templates_too_close" &&
            prediction->diagnostics["status"] == "accepted") {
            r.registration["accepted"] = true;
            r.registration["status"] = "accepted_ordered_objects";
            r.registration["template_selection_ambiguous"] = true;
        }
    }
    if (!r.registration["accepted"].get<bool>() || in.labels[index].empty()) {
        if (partial && prediction &&
            (prediction->diagnostics["status"] == "accepted" ||
             prediction->diagnostics["status"] == "partial_reference")) {
            r.registration["base_registration_accepted"] = r.registration["accepted"];
            r.registration["base_registration_status"] = r.registration["status"];
            r.registration["accepted"] = true;
            r.registration["status"] = "accepted_objects_only";
            r.registration["prediction_scope"] = "ordered_objects";
            r.mapped_labels = prediction->labels.clone();
            r.label_valid = binary(prediction->labels).mul(in.target_valid);
            r.object_identities = prediction->identities.clone();
            r.object_valid = r.label_valid.clone();
            r.base_labels = cv::Mat::zeros(in.target.size(), CV_16U);
            r.base_valid = cv::Mat::zeros(in.target.size(), CV_8U);
        }
        return r;
    }
    if (!H)
        throw std::logic_error("Accepted base match without geometry");
    auto transferred = transfer(in, r, index, *H, cfg);
    if (cfg["periodic_completion"].get<bool>() && cfg["label_mode"] == "semantic") {
        complete_periodic(r, transferred.first, transferred.second, in.template_valids[index],
                          in.target_valid, cfg);
        supplemental(in, r, index, cfg);
    }
    if (prediction) {
        Json periodic = r.registration.value("periodic_completion", Json::object());
        bool approved = false;
        for (auto c : periodic.value("copies", Json::array()))
            if (c.value("geometry_accepted", c.value("accepted", false))) {
                approved = true;
                break;
            }
        if (approved) {
            std::vector<cv::Vec2d> vectors;
            for (auto v : periodic.value("period_vectors", Json::array())) {
                double x = v[0], y = v[1];
                vectors.emplace_back((*H)(0, 0) * x + (*H)(0, 1) * y,
                                     (*H)(1, 0) * x + (*H)(1, 1) * y);
            }
            complete_object_support(*prediction, in.templates, in.objects, source_valids, in.target,
                                    in.target_valid, vectors, cfg);
        }
        fuse(in, r, *prediction, cfg);
        r.registration["ordered_objects"] = prediction->diagnostics;
    }
    return r;
}
} // namespace ptm
