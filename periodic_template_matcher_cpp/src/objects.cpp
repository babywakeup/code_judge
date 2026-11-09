#include "ptm/core.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <opencv2/imgproc.hpp>

namespace ptm {
namespace {
const Matrix SWAP(0, 1, 0, 1, 0, 0, 0, 0, 1);
bool partial(const Config &c) {
    return c.contains("object_visible_count") && !c["object_visible_count"].is_null();
}
cv::Mat trans(const cv::Mat &m) {
    cv::Mat o;
    cv::transpose(m, o);
    return o;
}
cv::Size scaled(cv::Size s, double scale) {
    return {pyround(s.width * scale), pyround(s.height * scale)};
}
cv::Mat dilate(const cv::Mat &a, int n) {
    cv::Mat o;
    cv::dilate(a, o, cv::Mat::ones(n, n, CV_8U));
    return o;
}
cv::Rect bounds(const cv::Mat &a) {
    std::vector<cv::Point> p;
    cv::findNonZero(a, p);
    return p.empty() ? cv::Rect() : cv::boundingRect(p);
}
cv::Point2d point(const Matrix &h, double x, double y) {
    auto p = h * cv::Vec3d(x, y, 1);
    return {p[0] / p[2], p[1] / p[2]};
}
struct Candidate {
    Json j;
    cv::Mat mask, support;
    bool background;
    double quality() const {
        return j["score"].get<double>() - .2 * (1 - j["vertical_fraction"].get<double>()) -
               .025 * (1 - j["horizontal_fraction"].get<double>()) -
               .05 * int(j["reference_partial"].get<bool>());
    }
};
using C = std::shared_ptr<Candidate>;
struct Pool {
    std::vector<C> all, distinct;
};
struct Group {
    double score;
    std::vector<std::string> order;
    std::vector<C> path;
};
std::vector<cv::Vec3d> correlation_peaks(cv::Mat patch, cv::Mat mask, cv::Mat target, cv::Mat valid,
                                         const Config &cfg) {
    int ph = patch.rows, pw = patch.cols;
    patch.convertTo(patch, CV_32F);
    mask.convertTo(mask, CV_32F);
    valid.convertTo(valid, CV_32F);
    target.convertTo(target, CV_32F);
    patch = patch.mul(mask);
    target = target.mul(valid);
    auto corr = [&](const cv::Mat &a, const cv::Mat &b) {
        cv::Mat padded, o;
        cv::copyMakeBorder(a, padded, ph - 1, ph - 1, pw - 1, pw - 1, cv::BORDER_CONSTANT, 0);
        cv::matchTemplate(padded, b, o, cv::TM_CCORR);
        return o;
    };
    cv::Mat count = corr(valid, mask), st = corr(target, mask), ss = corr(valid, patch),
            et = corr(target.mul(target), mask), es = corr(valid, patch.mul(patch)),
            cross = corr(target, patch), scores(count.size(), CV_32F, cv::Scalar(-1));
    double minimum = std::max(64., cfg.value("object_min_visible", .15) * cv::sum(mask)[0]);
    for (int y = 0; y < count.rows; ++y)
        for (int x = 0; x < count.cols; ++x) {
            double n = std::max(0.f, count.at<float>(y, x)), d = std::max(n, 1.),
                   t = st.at<float>(y, x), s = ss.at<float>(y, x),
                   te = std::max(0., et.at<float>(y, x) - t * t / d),
                   se = std::max(0., es.at<float>(y, x) - s * s / d);
            if (n >= minimum && te / d > .0025 && se / d > .0025) {
                double v = (cross.at<float>(y, x) - t * s / d) / std::sqrt(std::max(te * se, 1e-8));
                if (std::isfinite(v))
                    scores.at<float>(y, x) = float(std::clamp(v, -1., 1.));
            }
        }
    std::vector<cv::Vec3d> peaks;
    std::vector<std::pair<int, int>> locations;
    bool pc = partial(cfg);
    int budget = cfg.value("object_candidates", 8) * (pc ? 2 : 1), rx = std::max(6, pw / 3),
        ry = std::max(6, ph / 8);
    for (int i = 0; i < budget; ++i) {
        double value;
        cv::Point p;
        cv::minMaxLoc(scores, nullptr, &value, nullptr, &p);
        if (value < cfg.value("object_min_score", .7))
            break;
        peaks.emplace_back(value, p.x - pw + 1, p.y - ph + 1);
        cv::Rect r(std::max(0, p.x - rx), std::max(0, p.y - ry), 0, 0);
        r.width = std::min(scores.cols, p.x + rx + 1) - r.x;
        r.height = std::min(scores.rows, p.y + ry + 1) - r.y;
        scores(r).setTo(-1);
        if (pc) {
            auto it = std::find_if(locations.begin(), locations.end(),
                                   [&](auto l) { return std::abs(l.first - p.x) <= rx; });
            if (it == locations.end()) {
                locations.emplace_back(p.x, 0);
                it = locations.end() - 1;
            }
            if (++it->second >= 2) {
                int x0 = std::max(0, it->first - rx),
                    x1 = std::min(scores.cols, it->first + rx + 1);
                scores.colRange(x0, x1).setTo(-1);
            }
            if (int(locations.size()) >= cfg.value("object_candidates", 8))
                break;
        }
    }
    return peaks;
}
std::vector<Group> layouts(const std::map<std::string, Pool> &pools,
                           const std::vector<std::string> &cycle,
                           const std::map<std::pair<std::string, std::string>, double> &spacing,
                           const Config &cfg) {
    int n = partial(cfg) ? cfg["object_visible_count"].get<int>() : int(cycle.size());
    std::vector<Group> groups;
    for (size_t phase = 0; phase < cycle.size(); ++phase) {
        std::vector<std::string> order;
        for (int i = 0; i < n; ++i)
            order.push_back(cycle[(phase + i) % cycle.size()]);
        std::vector<Group> paths;
        for (auto c : pools.at(order[0]).distinct)
            paths.push_back({c->quality() / n, order, {c}});
        for (int i = 1; i < n; ++i) {
            std::vector<Group> next;
            for (auto c : pools.at(order[i]).distinct) {
                std::vector<Group> opts;
                for (auto &p : paths) {
                    auto a = p.path.back();
                    double ax = a->j["center_x"], bx = c->j["center_x"];
                    if (ax >= bx)
                        continue;
                    int overlap = cv::countNonZero(a->mask & c->mask);
                    if (overlap >
                        .15 * std::min(cv::countNonZero(a->mask), cv::countNonZero(c->mask)))
                        continue;
                    auto it = spacing.find({a->j["identity"], c->j["identity"]});
                    double cost = it == spacing.end()
                                      ? 0
                                      : cfg.value("object_spacing_weight", .3) *
                                            std::min(std::abs((bx - ax) / it->second - 1), 1.) /
                                            std::max(n - 1, 1);
                    auto q = p;
                    q.score += c->quality() / n - cost;
                    q.path.push_back(c);
                    opts.push_back(q);
                }
                std::stable_sort(opts.begin(), opts.end(),
                                 [](auto &a, auto &b) { return a.score > b.score; });
                for (size_t k = 0; k < std::min(size_t(2), opts.size()); ++k)
                    next.push_back(opts[k]);
            }
            paths = std::move(next);
        }
        std::stable_sort(paths.begin(), paths.end(),
                         [](auto &a, auto &b) { return a.score > b.score; });
        for (size_t k = 0; k < std::min(size_t(n == 1 ? 2 : 1), paths.size()); ++k)
            groups.push_back(paths[k]);
    }
    std::stable_sort(groups.begin(), groups.end(),
                     [](auto &a, auto &b) { return a.score > b.score; });
    return groups;
}
Matrix refine_object(const cv::Mat &image, const cv::Mat &mask, const cv::Mat &sv,
                     const cv::Mat &target, const cv::Mat &tv, const Matrix &H, const Config &cfg) {
    double s = matching_scale(cfg, {target.size(), image.size()});
    auto ss = scaled(image.size(), s), ts = scaled(target.size(), s);
    Matrix da = scale_matrix(image.size(), ss), db = scale_matrix(target.size(), ts);
    auto a = resize_matching(gray(image), sv, ss), b = resize_matching(gray(target), tv, ts);
    auto src = features(a.first, a.second), dst = features(b.first, b.second);
    cv::Mat local;
    cv::resize(warp_mask(mask, H, target.size()), local, ts, 0, 0, cv::INTER_NEAREST);
    dst.mask = dilate(local, 9) & b.second;
    Matrix work = db * H * da.inv();
    Config c = cfg;
    c["min_overlap"] = .2;
    auto before = score(src, dst, work, c);
    auto result = refine(src, dst, work, c);
    if (!result.second)
        return H;
    auto after = score(src, dst, result.first, c);
    Matrix candidate = db.inv() * result.first * da;
    auto r = bounds(mask);
    double distance = 0;
    for (auto p : std::vector<cv::Point2d>{{double(r.x), double(r.y)},
                                           {double(r.x + r.width - 1), double(r.y)},
                                           {double(r.x + r.width - 1), double(r.y + r.height - 1)},
                                           {double(r.x), double(r.y + r.height - 1)}})
        distance = std::max(distance, cv::norm(point(candidate, p.x, p.y) - point(H, p.x, p.y)));
    if (distance * s <= 6 &&
        after["structure_ncc"].get<double>() > before["structure_ncc"].get<double>() + .001 &&
        after["edge_alignment"].get<double>() >= before["edge_alignment"].get<double>() &&
        after["match_score"].get<double>() > before["match_score"].get<double>())
        return candidate;
    return H;
}
} // namespace

// Shared by the completion implementation; transposition recursively preserves all exported poses.
Config object_transpose_config(Config c) {
    c["object_order_axis"] = c.value("object_order_axis", std::string("x")) == "x" ? "y" : "x";
    if (c.contains("class_periodicity") && !c["class_periodicity"].is_null())
        for (auto &axes : c["class_periodicity"].items())
            for (auto &a : axes.value())
                a = a == "x" ? "y" : "x";
    if (c.contains("period_vectors"))
        for (auto &v : c["period_vectors"]) {
            Json t = v[0];
            v[0] = v[1];
            v[1] = t;
        }
    return c;
}
Json object_transpose_diagnostics(const Json &v) {
    if (v.is_array()) {
        Json out = Json::array();
        for (auto &i : v)
            out.push_back(object_transpose_diagnostics(i));
        return out;
    }
    if (!v.is_object())
        return v;
    Json out = Json::object();
    std::map<std::string, std::string> rename = {{"center_x", "center_y"},
                                                 {"center_y", "center_x"},
                                                 {"horizontal_fraction", "vertical_fraction"},
                                                 {"vertical_fraction", "horizontal_fraction"},
                                                 {"left", "top"},
                                                 {"top", "left"},
                                                 {"right", "bottom"},
                                                 {"bottom", "right"}};
    for (auto &e : v.items()) {
        auto k = e.key();
        Json a = e.value();
        if ((k == "matrix_template_to_target" || k == "corrected_matrix") && !a.is_null())
            a = matrix_json(SWAP * matrix_from_json(a) * SWAP);
        else if (k == "supporting_transforms") {
            for (auto &m : a)
                m = matrix_json(SWAP * matrix_from_json(m) * SWAP);
        } else if (k == "target_vector")
            a = Json::array({a[1], a[0]});
        else if (k == "repeat_axes") {
            for (auto &x : a)
                x = x == "x" ? "y" : "x";
        } else if (k == "order_axis")
            a = a == "x" ? "y" : "x";
        else
            a = object_transpose_diagnostics(a);
        out[rename.count(k) ? rename[k] : k] = a;
    }
    return out;
}
ObjectPrediction object_transpose_prediction(const ObjectPrediction &p) {
    return {trans(p.labels), trans(p.identities), trans(p.valid),
            object_transpose_diagnostics(p.diagnostics)};
}

ObjectPrediction match_ordered_objects(const std::vector<cv::Mat> &images,
                                       const std::vector<std::vector<ObjectRef>> &references,
                                       const std::vector<cv::Mat> &source_valids,
                                       const cv::Mat &target, const cv::Mat &target_valid,
                                       const std::vector<std::optional<Matrix>> &matrices,
                                       const Config &cfg) {
    if (references.size() != images.size() || source_valids.size() != images.size() ||
        matrices.size() != images.size())
        throw std::invalid_argument("One reference list, valid mask and optional matrix per image required");
    if (cfg.value("object_order_axis", std::string("x")) == "y") {
        std::vector<cv::Mat> ims, vs;
        auto refs = references;
        auto ms = matrices;
        for (size_t i = 0; i < images.size(); ++i) {
            ims.push_back(trans(images[i]));
            vs.push_back(trans(source_valids[i]));
            for (auto &r : refs[i])
                r.mask = trans(r.mask);
            if (ms[i])
                ms[i] = SWAP * (*ms[i]) * SWAP;
        }
        return object_transpose_prediction(match_ordered_objects(
            ims, refs, vs, trans(target), trans(target_valid), ms, object_transpose_config(cfg)));
    }
    cv::Size shape = target.size();
    ObjectPrediction output{cv::Mat::zeros(shape, CV_16U), cv::Mat::zeros(shape, CV_16U),
                            cv::Mat::zeros(shape, CV_8U), Json::object()};
    auto &info = output.diagnostics;
    std::vector<std::string> cycle = cfg.at("object_cycle").get<std::vector<std::string>>();
    info = {{"status", "insufficient_evidence"},
            {"visible_order", ""},
            {"identity_names", Json::object()},
            {"class_id", cfg.at("ordered_object_class")},
            {"order_axis", "x"},
            {"objects", Json::array()},
            {"candidate_counts", Json::object()},
            {"alternatives", Json::array()},
            {"valid_coverage_roi", 0.}};
    for (size_t i = 0; i < cycle.size(); ++i)
        info["identity_names"][std::to_string(i + 1)] = cycle[i];
    std::vector<cv::Size> sizes{shape};
    for (auto &im : images)
        sizes.push_back(im.size());
    double scale = matching_scale(cfg, sizes);
    auto ts = scaled(shape, scale);
    Matrix db = scale_matrix(shape, ts);
    auto tgt = resize_matching(gray(target), target_valid, ts);
    auto tf = features(tgt.first, tgt.second).norm;
    std::map<std::string, Pool> pools;
    for (auto &name : cycle)
        pools[name] = Pool{};
    for (size_t ti = 0; ti < images.size(); ++ti) {
        if (references[ti].empty())
            continue;
        std::vector<Matrix> hypotheses;
        if (matrices[ti])
            hypotheses.push_back(*matrices[ti]);
        else if (partial(cfg))
            for (auto &angle : cfg.at("angles"))
                for (auto &s : cfg.at("scales")) {
                    double theta = angle.get<double>() * CV_PI / 180, sc = s.get<double>();
                    hypotheses.emplace_back(sc * std::cos(theta), sc * std::sin(theta), 0,
                                            -sc * std::sin(theta), sc * std::cos(theta), 0, 0, 0,
                                            1);
                }
        auto ss = scaled(images[ti].size(), scale);
        Matrix da = scale_matrix(images[ti].size(), ss);
        auto source = resize_matching(gray(images[ti]), source_valids[ti], ss);
        auto sf = features(source.first, source.second).norm;
        for (auto initial : hypotheses) {
            Matrix linear = db * initial * da.inv();
            linear(0, 2) = linear(1, 2) = 0;
            double lx = 1e99, ly = 1e99, hx = -1e99, hy = -1e99;
            for (auto p : std::vector<cv::Point>{{0, 0},
                                                 {ss.width - 1, 0},
                                                 {ss.width - 1, ss.height - 1},
                                                 {0, ss.height - 1}}) {
                auto q = point(linear, p.x, p.y);
                lx = std::min(lx, q.x);
                ly = std::min(ly, q.y);
                hx = std::max(hx, q.x);
                hy = std::max(hy, q.y);
            }
            lx = std::floor(lx);
            ly = std::floor(ly);
            hx = std::ceil(hx);
            hy = std::ceil(hy);
            linear(0, 2) -= lx;
            linear(1, 2) -= ly;
            cv::Size canvas(int(hx - lx + 1), int(hy - ly + 1));
            auto aligned = warp(sf, linear, canvas), av = warp_mask(source.second, linear, canvas);
            for (size_t ri = 0; ri < references[ti].size(); ++ri) {
                auto &ref = references[ti][ri];
                cv::Mat raw = binary(ref.mask) & source_valids[ti];
                if (!cv::countNonZero(raw))
                    continue;
                cv::Mat mask;
                cv::resize(raw, mask, ss, 0, 0, cv::INTER_NEAREST);
                mask = warp_mask(mask, linear, canvas);
                cv::Mat context = dilate(mask, 9) & av;
                if (cv::countNonZero(context) < 64)
                    continue;
                auto rect = bounds(context), rb = bounds(raw);
                bool rp = rb.x < 2 || rb.x + rb.width - 1 >= images[ti].cols - 2;
                for (auto peak :
                     correlation_peaks(aligned(rect), context(rect), tf, tgt.second, cfg)) {
                    Matrix shift = Matrix::eye();
                    shift(0, 2) = peak[1] - rect.x;
                    shift(1, 2) = peak[2] - rect.y;
                    Matrix H = db.inv() * shift * linear * da;
                    cv::Mat visible = warp_mask(raw, H, shape) & target_valid;
                    int count = cv::countNonZero(visible);
                    double det = std::abs(H(0, 0) * H(1, 1) - H(0, 1) * H(1, 0)),
                           fraction =
                               std::min(1., count / std::max(cv::countNonZero(raw) * det, 1.));
                    if (fraction < cfg.value("object_min_visible", .15) || count < 64)
                        continue;
                    int rows = 0;
                    for (int y = 0; y < visible.rows; ++y)
                        if (cv::countNonZero(visible.row(y)))
                            ++rows;
                    double vf =
                        std::min(1., rows / std::max(rb.height * std::hypot(H(0, 1), H(1, 1)), 1.));
                    double cx = point(H, rb.x + (rb.width - 1) / 2., rb.y + (rb.height - 1) / 2.).x;
                    auto c = std::make_shared<Candidate>();
                    c->j = {{"identity", ref.identity},
                            {"template_index", ti},
                            {"reference_index", ri},
                            {"shape_index", ref.shape_index < 0 ? int(ri) : ref.shape_index},
                            {"score", peak[0]},
                            {"reference_partial", rp},
                            {"visible_fraction", fraction},
                            {"partial", fraction < .95},
                            {"center_x", cx},
                            {"vertical_fraction", vf},
                            {"horizontal_fraction", std::min(1., fraction / std::max(vf, 1e-6))},
                            {"matrix_template_to_target", matrix_json(H)}};
                    c->mask = visible;
                    c->support = warp_mask(source_valids[ti], H, shape);
                    c->background = ref.background_known;
                    pools[ref.identity].all.push_back(c);
                }
            }
        }
    }
    auto spacing_ms = matrices;
    if (partial(cfg))
        for (size_t ti = 0; ti < images.size(); ++ti) {
            C best;
            for (auto &pool : pools)
                for (auto c : pool.second.all)
                    if (c->j["template_index"] == ti && (!best || c->quality() > best->quality()))
                        best = c;
            if (best)
                spacing_ms[ti] = matrix_from_json(best->j["matrix_template_to_target"]);
        }
    info["reference_counts"] = Json::object();
    info["candidate_counts_by_template"] = Json::object();
    for (size_t ti = 0; ti < images.size(); ++ti) {
        int n = 0;
        for (auto &pool : pools)
            for (auto c : pool.second.all)
                if (c->j["template_index"] == ti)
                    ++n;
        info["reference_counts"][std::to_string(ti)] = references[ti].size();
        info["candidate_counts_by_template"][std::to_string(ti)] = n;
    }
    for (auto &name : cycle) {
        auto &pool = pools[name];
        std::stable_sort(pool.all.begin(), pool.all.end(),
                         [](auto &a, auto &b) { return a->quality() > b->quality(); });
        for (auto c : pool.all) {
            bool distinct = true;
            for (auto p : pool.distinct)
                if (std::abs(c->j["center_x"].get<double>() - p->j["center_x"].get<double>()) <=
                    std::max(8., 12 / scale))
                    distinct = false;
            if (distinct)
                pool.distinct.push_back(c);
            if (int(pool.distinct.size()) >= cfg.value("object_candidates", 8))
                break;
        }
        info["candidate_counts"][name] = pool.all.size();
    }
    std::map<std::pair<std::string, std::string>, std::vector<double>> observations;
    for (size_t ti = 0; ti < images.size(); ++ti) {
        if (!spacing_ms[ti])
            continue;
        std::vector<std::tuple<double, std::string, double>> centers;
        for (auto &r : references[ti]) {
            auto b = bounds(binary(r.mask));
            if (b.empty() || b.x < 2 || b.x + b.width - 1 >= images[ti].cols - 2)
                continue;
            double x = b.x + (b.width - 1) / 2., y = b.y + (b.height - 1) / 2.;
            auto p = (*spacing_ms[ti]) * cv::Vec3d(x, y, 1);
            centers.emplace_back(x, r.identity, p[0]);
        }
        std::sort(centers.begin(), centers.end());
        for (size_t i = 1; i < centers.size(); ++i) {
            auto a = std::get<1>(centers[i - 1]), b = std::get<1>(centers[i]);
            auto it = std::find(cycle.begin(), cycle.end(), a);
            double x = std::get<2>(centers[i - 1]), y = std::get<2>(centers[i]);
            if (it != cycle.end() && b == cycle[(it - cycle.begin() + 1) % cycle.size()] && y > x)
                observations[{a, b}].push_back(y - x);
        }
    }
    std::map<std::pair<std::string, std::string>, double> spacing;
    info["layout_spacings"] = Json::array();
    for (auto &e : observations) {
        double v = percentile(e.second, 50);
        spacing[e.first] = v;
        info["layout_spacings"].push_back(
            {{"left", e.first.first}, {"right", e.first.second}, {"target_px", v}});
    }
    info["cycle_size"] = cycle.size();
    info["instance_count"] = partial(cfg) ? cfg["object_visible_count"] : Json(cycle.size());
    auto groups = layouts(pools, cycle, spacing, cfg);
    if (groups.empty())
        return output;
    auto &best = groups[0];
    Json margin = nullptr;
    for (size_t i = 1; i < groups.size(); ++i)
        if (groups[i].order != best.order || cycle.size() == 1) {
            margin = best.score - groups[i].score;
            break;
        }
    std::string order;
    for (auto &name : best.order)
        order += name;
    info["visible_order"] = order;
    info["visible_identities"] = best.order;
    info["score"] = best.score;
    info["order_margin"] = margin;
    for (size_t i = 0; i < std::min(size_t(6), groups.size()); ++i) {
        std::string o;
        for (auto &n : groups[i].order)
            o += n;
        info["alternatives"].push_back({{"order", o}, {"score", groups[i].score}});
    }
    info["proposed_objects"] = Json::array();
    for (auto c : best.path)
        info["proposed_objects"].push_back(c->j);
    if (!margin.is_null() && margin.get<double>() < cfg.value("object_min_gap", .03)) {
        info["status"] = "ambiguous_identity";
        return output;
    }
    info["status"] = "accepted";
    output.valid = target_valid.clone();
    for (auto chosen : best.path) {
        std::string name = chosen->j["identity"];
        std::vector<C> selected;
        double cx = chosen->j["center_x"], sc = chosen->j["score"];
        for (auto c : pools[name].all)
            if (std::abs(c->j["center_x"].get<double>() - cx) < 8 / scale &&
                c->j["score"].get<double>() >= sc - .025)
                selected.push_back(c);
        cv::Mat combined = cv::Mat::zeros(shape, CV_8U), support = combined.clone();
        bool all_partial = true;
        for (auto c : selected) {
            if (partial(cfg)) {
                int ti = c->j["template_index"], ri = c->j["reference_index"];
                cv::Mat raw = binary(references[ti][ri].mask) & source_valids[ti];
                Matrix initial = matrix_from_json(c->j["matrix_template_to_target"]),
                       H = refine_object(images[ti], raw, source_valids[ti], target, target_valid,
                                         initial, cfg);
                c->j["local_refined"] = cv::norm(cv::Mat(H), cv::Mat(initial)) != 0;
                c->j["matrix_template_to_target"] = matrix_json(H);
                auto r = bounds(raw);
                c->j["center_x"] = point(H, r.x + (r.width - 1) / 2., r.y + (r.height - 1) / 2.).x;
                c->mask = warp_mask(raw, H, shape) & target_valid;
                c->support = warp_mask(source_valids[ti], H, shape);
            }
            combined |= c->mask;
            support |= c->support;
            all_partial = all_partial && c->j["reference_partial"].get<bool>();
        }
        auto r = bounds(combined);
        if (r.empty())
            continue;
        cv::Mat band = cv::Mat::zeros(shape, CV_8U);
        band.colRange(r.x, r.x + r.width).setTo(1);
        output.valid &= (1 - (band & (1 - support)));
        if (!chosen->background)
            output.valid &= combined;
        if (all_partial) {
            output.valid &= combined;
            info["status"] = "partial_reference";
        }
        output.labels.setTo(cfg["ordered_object_class"].get<int>(), combined);
        output.identities.setTo(
            int(std::find(cycle.begin(), cycle.end(), name) - cycle.begin() + 1), combined);
        Json record = chosen->j;
        record["supporting_transforms"] = Json::array();
        record["supporting_references"] = Json::array();
        for (auto c : selected) {
            record["supporting_transforms"].push_back(c->j["matrix_template_to_target"]);
            record["supporting_references"].push_back(
                {{"template_index", c->j["template_index"]},
                 {"reference_index", c->j["reference_index"]},
                 {"matrix_template_to_target", c->j["matrix_template_to_target"]}});
        }
        info["objects"].push_back(record);
    }
    output.valid |= binary(output.labels);
    output.valid &= target_valid;
    info["valid_coverage_roi"] =
        double(cv::countNonZero(output.valid)) / std::max(cv::countNonZero(target_valid), 1);
    return output;
}
} // namespace ptm
