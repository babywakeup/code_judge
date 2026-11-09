#include "ptm/core.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/video/tracking.hpp>
#include <stdexcept>
#include <thread>
#include <type_traits>
// NumPy compiles pocketfft without implicit contraction. Spectrum multiplication
// below explicitly uses fma, as NumPy's X86_V3 complex multiply ufunc does.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif
#define POCKETFFT_NO_MULTITHREADING
#include <pocketfft/pocketfft_hdronly.h>
namespace ptm {
namespace {
template <typename T>
std::vector<std::complex<T>> real_spectrum(const cv::Mat &input, cv::Size size) {
    // NumPy 2.5 passes the Python integer fct=1 to each forward FFT gufunc.
    // Its resolver selects the double loop even for float32/complex64 input,
    // then casts into the complex64 output after EACH axis. Inverse FFTs use
    // a typed reciprocal and therefore retain the native float32 loop.
    cv::Mat converted, padded = cv::Mat::zeros(size, CV_64F);
    input.convertTo(converted, CV_64F);
    const cv::Rect area(0, 0, std::min(input.cols, size.width), std::min(input.rows, size.height));
    converted(area).copyTo(padded(area));
    const pocketfft::shape_t shape{size_t(size.height), size_t(size.width)};
    const ptrdiff_t width = size.width / 2 + 1;
    const pocketfft::stride_t real_stride{ptrdiff_t(padded.step), ptrdiff_t(sizeof(double))};
    const pocketfft::stride_t complex_stride{width * ptrdiff_t(sizeof(std::complex<double>)),
                                             ptrdiff_t(sizeof(std::complex<double>))};
    std::vector<std::complex<double>> work(size_t(size.height) * size_t(width));
    pocketfft::r2c(shape, real_stride, complex_stride, pocketfft::shape_t{1}, pocketfft::FORWARD,
                   padded.ptr<double>(), work.data(), 1.);
    if constexpr (std::is_same_v<T, float>)
        for (auto &z : work)
            z = std::complex<float>(z);
    const pocketfft::shape_t half_shape{size_t(size.height), size_t(width)};
    pocketfft::c2c(half_shape, complex_stride, complex_stride, pocketfft::shape_t{0},
                   pocketfft::FORWARD, work.data(), work.data(), 1.);
    std::vector<std::complex<T>> spectrum(work.begin(), work.end());
    return spectrum;
}
template <typename A, typename B>
cv::Mat fft_correlate(const cv::Mat &a, const cv::Mat &b, cv::Size size) {
    using T = std::common_type_t<A, B>;
    auto fa = real_spectrum<A>(a, size);
    auto fb = real_spectrum<B>(b, size);
    std::vector<std::complex<T>> product(fa.size());
    for (size_t i = 0; i < fa.size(); ++i) {
        const T ar = fa[i].real(), ai = fa[i].imag(), br = fb[i].real(), bi = -T(fb[i].imag());
        // Cast each spectrum only AFTER its own FFT; mixed float32/integer input
        // multiplies a rounded complex64 spectrum by a complex128 spectrum.
        const T ii = ai * bi, ir = ai * br;
        product[i] = {std::fma(ar, br, -ii), std::fma(ar, bi, ir)};
    }
    const pocketfft::shape_t shape{size_t(size.height), size_t(size.width)},
        half_shape{size_t(size.height), size_t(size.width / 2 + 1)};
    const pocketfft::stride_t complex_stride{ptrdiff_t(size.width / 2 + 1) *
                                                 ptrdiff_t(sizeof(std::complex<T>)),
                                             ptrdiff_t(sizeof(std::complex<T>))};
    cv::Mat result(size, std::is_same_v<T, float> ? CV_32F : CV_64F);
    const pocketfft::stride_t real_stride{ptrdiff_t(result.step), ptrdiff_t(sizeof(T))};
    // np.fft.irfft2 performs ifft(axis=0, norm=1/h), then irfft(axis=1,
    // norm=1/w). Combining these factors changes float32 rounding and ties.
    pocketfft::c2c(half_shape, complex_stride, complex_stride, pocketfft::shape_t{0},
                   pocketfft::BACKWARD, product.data(), product.data(), T(1) / T(size.height));
    pocketfft::c2r(shape, complex_stride, real_stride, pocketfft::shape_t{1}, pocketfft::BACKWARD,
                   product.data(), result.ptr<T>(), T(1) / T(size.width));
    return result;
}
double real_value(const cv::Mat &a, int y, int x) {
    return a.depth() == CV_32F ? double(a.at<float>(y, x)) : a.at<double>(y, x);
}
// Match the float32 reduction of NumPy's contiguous boolean-indexed array.
// Summing in double and then casting is not equivalent, especially with a DC
// offset. Keep eight independent accumulators and a balanced reduction tree.
template <typename T> T pairwise_sum(const T *values, size_t count) {
    if (count < 8) {
        T sum = -T(0);
        for (size_t i = 0; i < count; ++i)
            sum += values[i];
        return sum;
    }
    if (count <= 128) {
        T sums[8];
        std::copy_n(values, 8, sums);
        size_t i = 8;
        for (; i < count - count % 8; i += 8)
            for (size_t j = 0; j < 8; ++j)
                sums[j] += values[i + j];
        T sum = ((sums[0] + sums[1]) + (sums[2] + sums[3])) +
                ((sums[4] + sums[5]) + (sums[6] + sums[7]));
        for (; i < count; ++i)
            sum += values[i];
        return sum;
    }
    size_t middle = (count / 2) / 8 * 8;
    return pairwise_sum(values, middle) + pairwise_sum(values + middle, count - middle);
}
template <typename T>
void center_feature(cv::Mat &feature, const cv::Mat &mask, bool promote = false) {
    std::vector<T> values;
    values.reserve(feature.total());
    for (int y = 0; y < feature.rows; ++y)
        for (int x = 0; x < feature.cols; ++x)
            if (real_value(mask, y, x) > 0)
                values.push_back(feature.at<T>(y, x));
    const T mean = values.empty()
                       ? std::numeric_limits<T>::quiet_NaN()
                       : pairwise_sum(values.data(), values.size()) / static_cast<T>(values.size());
    cv::Mat centered(feature.size(), promote ? CV_64F : feature.depth());
    for (int y = 0; y < feature.rows; ++y)
        for (int x = 0; x < feature.cols; ++x)
            if (promote)
                centered.at<double>(y, x) =
                    double(T(feature.at<T>(y, x) - mean)) * real_value(mask, y, x);
            else
                centered.at<T>(y, x) = (feature.at<T>(y, x) - mean) * T(real_value(mask, y, x));
    feature = centered;
}
} // namespace
Features features(const cv::Mat &input, const cv::Mat &mask) {
    cv::Mat g = input.clone(), x;
    if (cv::countNonZero(mask) != mask.total()) {
        cv::Mat excluded = (1 - mask) * 255;
        cv::inpaint(g, excluded, g, 3, cv::INPAINT_TELEA);
    }
    g.convertTo(x, CV_32F);
    // Python divides float32 pixels by 255.0. convertTo(..., 1./255)
    // multiplies by a rounded reciprocal instead, perturbing ECC's local optimum.
    for (int y = 0; y < x.rows; ++y)
        for (int z = 0; z < x.cols; ++z)
            x.at<float>(y, z) /= 255.f;
    cv::GaussianBlur(x, x, {}, .8);
    cv::Mat mean, var, norm, dx, dy, grad, safe, edges;
    cv::GaussianBlur(x, mean, {}, 9);
    cv::GaussianBlur(x.mul(x), var, {}, 9);
    var -= mean.mul(mean);
    cv::max(var, 0, var);
    cv::sqrt(var + .0025, var);
    cv::divide(x - mean, var, norm);
    cv::max(norm, -3, norm);
    cv::min(norm, 3, norm);
    cv::Scharr(x, dx, CV_32F, 1, 0);
    cv::Scharr(x, dy, CV_32F, 0, 1);
    cv::magnitude(dx, dy, grad);
    double cap = std::max(percentile(selected(grad, mask), 95),
                          1e-5); // NumPy float32 division rounds the scalar cap to float32 first;
    // OpenCV scalar /= uses reciprocal multiplication, which differs by an ULP.
    const float cap32 = static_cast<float>(cap);
    for (int y = 0; y < grad.rows; ++y)
        for (int z = 0; z < grad.cols; ++z)
            grad.at<float>(y, z) = std::min(grad.at<float>(y, z) / cap32, 1.f);
    cv::erode(mask, safe, cv::Mat::ones(7, 7, CV_8U), {-1, -1}, 1, cv::BORDER_CONSTANT, 0);
    cv::Canny(g, edges, 35, 100);
    edges = binary(edges).mul(safe);
    cv::Mat sf;
    safe.convertTo(sf, CV_32F);
    return {norm.mul(sf), grad.mul(sf), edges, safe};
}
double ncc(const std::vector<double> &x, const std::vector<double> &y) {
    if (x.size() < 32)
        return 0;
    double mx = std::accumulate(x.begin(), x.end(), 0.) / x.size(),
           my = std::accumulate(y.begin(), y.end(), 0.) / y.size(), a = 0, b = 0, c = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        double u = x[i] - mx, v = y[i] - my;
        a += u * v;
        b += u * u;
        c += v * v;
    }
    return std::clamp(a / std::max(std::sqrt(b * c), 1e-9), 0., 1.);
}
Json score(const Features &s, const Features &t, const Matrix &H, const Config &cfg) {
    cv::Mat valid = warp_mask(s.mask, H, t.norm.size()).mul(t.mask);
    cv::erode(valid, valid, cv::Mat::ones(3, 3, CV_8U));
    double coverage = cv::sum(valid)[0] / std::max(cv::sum(t.mask)[0], 1.);
    Json empty = {{"match_score", 0.},           {"structure_ncc", 0.},
                  {"edge_alignment", 0.},        {"local_consistency", 0.},
                  {"target_coverage", coverage}, {"edge_distance_p95_work_px", nullptr}};
    if (coverage < cfg.value("min_overlap", .45))
        return empty;
    cv::Mat wn = warp(s.norm, H, t.norm.size()),
            we = warp_mask(s.edges, H, t.norm.size()).mul(valid), et = t.edges.mul(valid);
    double structure = ncc(selected(wn, valid), selected(t.norm, valid));
    if (cv::sum(we)[0] < 40 || cv::sum(et)[0] < 40)
        return empty;
    cv::Mat dt, ds;
    cv::distanceTransform(1 - et, dt, cv::DIST_L2, 5);
    cv::distanceTransform(1 - we, ds, cv::DIST_L2, 5);
    auto distances = selected(dt, we), other = selected(ds, et);
    distances.insert(distances.end(), other.begin(), other.end());
    double edge = 0;
    for (double d : distances)
        edge += std::exp(-d / cfg.value("edge_tolerance_px", 2.));
    edge /= distances.size();
    std::vector<double> tiles;
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            cv::Rect r(x * t.norm.cols / 4, y * t.norm.rows / 4,
                       (x + 1) * t.norm.cols / 4 - x * t.norm.cols / 4,
                       (y + 1) * t.norm.rows / 4 - y * t.norm.rows / 4);
            auto v = valid(r);
            if (cv::sum(v)[0] >= std::max(100., v.total() * .4)) {
                cv::Scalar mean, sd;
                cv::meanStdDev(t.norm(r), mean, sd, v);
                if (sd[0] > .05)
                    tiles.push_back(ncc(selected(wn(r), v), selected(t.norm(r), v)));
            }
        }
    double local = tiles.empty() ? 0 : percentile(tiles, 25);
    return {{"match_score",
             std::clamp(100 * (.55 * structure + .30 * edge + .15 * local) * std::sqrt(coverage),
                        0., 100.)},
            {"structure_ncc", structure},
            {"edge_alignment", edge},
            {"local_consistency", local},
            {"target_coverage", coverage},
            {"edge_distance_p95_work_px", percentile(distances, 95)}};
}
std::pair<Matrix, bool> refine(const Features &s, const Features &t, const Matrix &H,
                               const Config &cfg) {
    Matrix current = H;
    bool converged = false;
    for (double factor : {.5, 1.}) {
        cv::Size size(std::max(32, pyround(t.norm.cols * factor)),
                      std::max(32, pyround(t.norm.rows * factor)));
        Matrix D(size.width / double(t.norm.cols), 0, 0, 0, size.height / double(t.norm.rows), 0, 0,
                 0, 1);
        Matrix hs = D * current;
        cv::Mat warped = warp(s.norm, hs, size), valid = warp_mask(s.mask, hs, size), tm, target;
        cv::resize(t.mask, tm, size, 0, 0, cv::INTER_NEAREST);
        valid = valid.mul(tm);
        cv::erode(valid, valid, cv::Mat::ones(5, 5, CV_8U));
        if (cv::sum(valid)[0] < 1000)
            continue;
        cv::resize(t.norm, target, size, 0, 0, cv::INTER_AREA);
        cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
        try {
            cv::findTransformECC(target, warped, W, cv::MOTION_AFFINE,
                                 {cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                  cfg.value("ecc_iterations", 70), 1e-5},
                                 valid * 255, 5);
            Matrix M = Matrix::eye();
            for (int y = 0; y < 2; y++)
                for (int x = 0; x < 3; x++)
                    M(y, x) = W.at<float>(y, x);
            Matrix updated = D.inv() * M.inv() * D * current;
            cv::Mat A = (cv::Mat(updated))(cv::Rect(0, 0, 2, 2)).clone(), sv;
            cv::SVD::compute(A, sv);
            if (cv::determinant(A) > 0 && sv.at<double>(1) > .65 && sv.at<double>(0) < 1.5) {
                current = updated;
                converged = true;
            }
        } catch (const cv::Exception &) {
        }
    }
    return {current, converged};
}
cv::Mat correlate(const cv::Mat &a, const cv::Mat &b, cv::Size size) {
    if (a.channels() != 1 || b.channels() != 1 || size.width < 1 || size.height < 1)
        throw std::invalid_argument(
            "Correlation requires real single-channel arrays and a positive FFT shape");
    if (a.depth() == CV_32F) {
        if (b.depth() == CV_32F)
            return fft_correlate<float, float>(a, b, size);
        return fft_correlate<float, double>(a, b, size);
    }
    if (b.depth() == CV_32F)
        return fft_correlate<double, float>(a, b, size);
    return fft_correlate<double, double>(a, b, size);
}
std::vector<cv::Vec3d> translation_peaks(const cv::Mat &source, const cv::Mat &target,
                                         const cv::Mat &sm0, const cv::Mat &tm0,
                                         const Config &cfg) {
    const bool opencv = cfg.value("backend", std::string("fft")) == "opencv";
    cv::Mat sm = sm0, tm = tm0, s = source, t = target;
    if (opencv) {
        sm = binary(sm0);
        tm = binary(tm0);
        sm.convertTo(sm, CV_32F);
        tm.convertTo(tm, CV_32F);
        source.convertTo(s, CV_32F);
        target.convertTo(t, CV_32F);
        s = s.mul(sm);
        t = t.mul(tm);
        if (!cv::countNonZero(sm) || !cv::countNonZero(tm))
            return {};
    }
    const cv::Size fs(cv::getOptimalDFTSize(s.cols + t.cols - 1),
                      cv::getOptimalDFTSize(s.rows + t.rows - 1));
    auto corr = [&](const cv::Mat &a, const cv::Mat &b) {
        if (!opencv)
            return correlate(a, b, fs);
        cv::Mat pad, out, d;
        cv::copyMakeBorder(a, pad, s.rows - 1, s.rows - 1, s.cols - 1, s.cols - 1,
                           cv::BORDER_CONSTANT, 0);
        cv::matchTemplate(pad, b, out, cv::TM_CCORR);
        out.convertTo(d, CV_64F);
        return d;
    };
    auto ov = corr(tm, sm), num = corr(t, s), e1 = corr(t.mul(t), sm), e2 = corr(tm, s.mul(s));
    cv::Mat merit(t.rows + s.rows - 1, t.cols + s.cols - 1, CV_64F);
    const double ts = cv::sum(tm)[0], ss = cv::sum(sm)[0], minimum = cfg.value("min_overlap", .45);
    for (int y = 0; y < merit.rows; ++y)
        for (int x = 0; x < merit.cols; ++x) {
            const int yy = opencv ? y : (y - s.rows + 1 + fs.height) % fs.height,
                      xx = opencv ? x : (x - s.cols + 1 + fs.width) % fs.width;
            double overlap = real_value(ov, yy, xx);
            if (opencv)
                overlap = std::nearbyint(std::max(overlap, 0.));
            double cov = std::max(overlap, 0.) / std::max(ts, 1.);
            if (ov.depth() == CV_32F)
                cov = float(std::max(float(overlap), 0.f) / float(std::max(ts, 1.)));
            double en = real_value(e1, yy, xx) * real_value(e2, yy, xx);
            const bool float_energy = e1.depth() == CV_32F && e2.depth() == CV_32F;
            if (float_energy)
                en = float(float(real_value(e1, yy, xx)) * float(real_value(e2, yy, xx)));
            if (opencv)
                en = std::max(real_value(e1, yy, xx), 0.) * std::max(real_value(e2, yy, xx), 0.);
            double den = float_energy ? double(std::sqrt(std::max(float(en), 1e-8f)))
                                      : std::sqrt(std::max(en, 1e-8));
            double v = real_value(num, yy, xx) / den;
            const bool float_corr = num.depth() == CV_32F && float_energy;
            if (float_corr)
                v = float(float(real_value(num, yy, xx)) / float(den));
            v = std::clamp(v, 0., 1.);
            double coverage_factor = std::sqrt(std::min(cov, 1.));
            if (ov.depth() == CV_32F)
                coverage_factor = std::sqrt(std::min(float(cov), 1.f));
            v *= coverage_factor;
            if (float_corr && ov.depth() == CV_32F)
                v = float(v);
            merit.at<double>(y, x) = (cov >= minimum && overlap >= minimum * ss &&
                                      (!opencv || en > 1e-8) && std::isfinite(v))
                                         ? v
                                         : -1;
        }
    std::vector<cv::Vec3d> out;
    int radius = std::max(8, std::min({s.rows, s.cols, t.rows, t.cols}) / 20);
    for (int i = 0; i < cfg.value("peaks_per_hypothesis", 4); ++i) {
        double v;
        cv::Point p;
        cv::minMaxLoc(merit, nullptr, &v, nullptr, &p);
        if (v < 0)
            break;
        out.push_back({v, double(p.x - s.cols + 1), double(p.y - s.rows + 1)});
        cv::Rect r(std::max(0, p.x - radius), std::max(0, p.y - radius), 0, 0);
        r.width = std::min(merit.cols, p.x + radius + 1) - r.x;
        r.height = std::min(merit.rows, p.y + radius + 1) - r.y;
        merit(r).setTo(-1);
    }
    return out;
}
Json periods(const cv::Mat &feature, const cv::Mat &mask) {
    cv::Mat f, m;
    feature.convertTo(f, feature.depth() == CV_32F ? CV_32F : CV_64F);
    mask.convertTo(m, CV_64F);
    if (f.depth() == CV_32F)
        center_feature<float>(f, m, mask.depth() == CV_64F || mask.depth() == CV_32S);
    else
        center_feature<double>(f, m);
    const cv::Size sz(cv::getOptimalDFTSize(2 * f.cols - 1), cv::getOptimalDFTSize(2 * f.rows - 1));
    auto c = correlate(f, f, sz), e1 = correlate(f.mul(f), mask, sz),
         e2 = correlate(mask, f.mul(f), sz), ov = correlate(mask, mask, sz);
    Json out = Json::array();
    for (int axis = 0; axis < 2; ++axis) {
        int limit = axis ? f.cols : f.rows, start = std::max(12, limit / 18),
            stop = int(limit * .7);
        auto val = [&](int i) {
            int y = axis ? 0 : i, x = axis ? i : 0;
            double en = real_value(e1, y, x) * real_value(e2, y, x);
            const bool single = e1.depth() == CV_32F && e2.depth() == CV_32F;
            if (single)
                en = float(float(real_value(e1, y, x)) * float(real_value(e2, y, x)));
            double den = single ? double(std::sqrt(std::max(float(en), 1e-9f)))
                                : std::sqrt(std::max(en, 1e-9));
            if (single && c.depth() == CV_32F)
                return double(float(real_value(c, y, x)) / float(den));
            return real_value(c, y, x) / den;
        };
        std::vector<std::pair<double, int>> candidates;
        for (int i = start + 1; i < stop - 1; ++i)
            if (val(i) > .65 && val(i) >= val(i - 1) && val(i) >= val(i + 1) &&
                real_value(ov, axis ? 0 : i, axis ? i : 0) > cv::sum(mask)[0] * .3)
                candidates.push_back({val(i), i});
        std::sort(candidates.rbegin(), candidates.rend());
        for (size_t i = 0; i < std::min<size_t>(2, candidates.size()); ++i)
            out.push_back({{"vector_work_px",
                            {axis ? candidates[i].second : 0, axis ? 0 : candidates[i].second}},
                           {"autocorrelation", candidates[i].first}});
    }
    return out;
}
namespace {
struct Prepared {
    cv::Size size;
    Matrix to_work;
    double scale;
    std::optional<Features> f;
};
Prepared prepare(const cv::Mat &image, const cv::Mat &valid, const Config &cfg, double scale) {
    cv::Mat g = gray(image);
    std::optional<int> vh;
    if (cfg.contains("valid_height") && !cfg["valid_height"].is_null())
        vh = cfg["valid_height"].get<int>();
    cv::Mat mask = roi_mask(g.size(), valid, vh);
    cv::Size sz(pyround(g.cols * scale), pyround(g.rows * scale));
    if (std::min(sz.width, sz.height) < 32)
        throw std::invalid_argument(
            "Extreme image aspect/size mismatch; use comparable fields of view");
    Matrix D = scale_matrix(g.size(), sz);
    auto resized = resize_matching(g, mask, sz);
    g = resized.first;
    mask = resized.second;
    cv::Scalar mean, sd;
    cv::meanStdDev(g, mean, sd, mask);
    Prepared p{sz, D, scale, {}};
    if (sd[0] >= 2) {
        auto f = features(g, mask);
        if (cv::sum(f.edges)[0] >= 40)
            p.f = f;
    }
    return p;
}
double difference(const Matrix &a, const Matrix &b, cv::Size sz) {
    std::vector<cv::Point2f> pts{{float(sz.width * .2), float(sz.height * .2)},
                                 {float(sz.width * .8), float(sz.height * .2)},
                                 {float(sz.width * .5), float(sz.height * .5)},
                                 {float(sz.width * .2), float(sz.height * .8)},
                                 {float(sz.width * .8), float(sz.height * .8)}},
        u, v;
    cv::perspectiveTransform(pts, u, a);
    cv::perspectiveTransform(pts, v, b);
    std::vector<double> d;
    for (int i = 0; i < 5; i++)
        d.push_back(cv::norm(u[i] - v[i]));
    return percentile(d, 50);
}
Json registration(const Prepared &s, const Prepared &t, const Config &cfg, const Json &lattice) {
    Json fail = {{"match_score", 0.},
                 {"geometry_accepted", false},
                 {"absolute_cell_verified", false},
                 {"periodic_detected", false},
                 {"matrix_template_to_target", nullptr},
                 {"candidates", Json::array()},
                 {"working_scale", t.scale},
                 {"working_target_size", {t.size.width, t.size.height}},
                 {"reasons", {"insufficient_texture_or_overlap"}}};
    if (!s.f || !t.f)
        return fail;
    std::vector<std::pair<double, Matrix>> coarse;
    for (double angle : cfg.value("angles", std::vector<double>{-2, 0, 2}))
        for (double zoom : cfg.value("scales", std::vector<double>{.97, 1, 1.03})) {
            cv::Mat ar = cv::getRotationMatrix2D(
                {float((s.size.width - 1) / 2.), float((s.size.height - 1) / 2.)}, angle, zoom);
            Matrix A = Matrix::eye();
            for (int y = 0; y < 2; y++)
                for (int x = 0; x < 3; x++)
                    A(y, x) = ar.at<double>(y, x);
            double lx = 1e30, ly = 1e30, hx = -1e30, hy = -1e30;
            for (double x : {0., double(s.size.width - 1)})
                for (double y : {0., double(s.size.height - 1)}) {
                    auto p = A * cv::Vec3d(x, y, 1);
                    lx = std::min(lx, p[0]);
                    ly = std::min(ly, p[1]);
                    hx = std::max(hx, p[0]);
                    hy = std::max(hy, p[1]);
                }
            A(0, 2) -= lx;
            A(1, 2) -= ly;
            cv::Size size(int(std::ceil(hx - lx)) + 1, int(std::ceil(hy - ly)) + 1);
            auto sg = warp(s.f->grad, A, size), sm = warp_mask(s.f->mask, A, size);
            for (auto peak : translation_peaks(sg, t.f->grad, sm, t.f->mask, cfg)) {
                Matrix T = Matrix::eye();
                T(0, 2) = peak[1];
                T(1, 2) = peak[2];
                coarse.push_back({peak[0], T * A});
            }
        }
    std::stable_sort(coarse.begin(), coarse.end(),
                     [](auto &a, auto &b) { return a.first > b.first; });
    std::vector<Matrix> shortlist;
    for (auto &item : coarse) {
        bool distinct = true;
        for (auto &prev : shortlist)
            if (difference(item.second, prev, s.size) <= 4)
                distinct = false;
        if (distinct)
            shortlist.push_back(item.second);
        if (shortlist.size() >= size_t(cfg.value("refine_candidates", 5)))
            break;
    }
    Json candidates = Json::array();
    for (auto H : shortlist) {
        Json metrics = score(*s.f, *t.f, H, cfg);
        auto refined = refine(*s.f, *t.f, H, cfg);
        Json rm = score(*s.f, *t.f, refined.first, cfg);
        bool ok = refined.second;
        if (rm["match_score"].get<double>() >= metrics["match_score"].get<double>()) {
            H = refined.first;
            metrics = rm;
        } else
            ok = false;
        bool distinct = true;
        for (auto &c : candidates)
            if (difference(H, matrix_from_json(c["_H_work"]), s.size) <= 3)
                distinct = false;
        if (distinct) {
            metrics["_H_work"] = matrix_json(H);
            metrics["ecc_converged"] = ok;
            metrics["matrix_template_to_target"] = matrix_json(t.to_work.inv() * H * s.to_work);
            candidates.push_back(metrics);
        }
    }
    if (candidates.empty())
        return fail;
    std::stable_sort(candidates.begin(), candidates.end(), [](const Json &a, const Json &b) {
        return a["match_score"].get<double>() > b["match_score"].get<double>();
    });
    Json best = candidates[0];
    double bs = best["match_score"],
           gap = candidates.size() > 1 ? bs - candidates[1]["match_score"].get<double>() : 0;
    bool alias = false;
    for (size_t i = 1; i < candidates.size(); i++)
        if (bs - candidates[i]["match_score"].get<double>() < cfg.value("min_candidate_gap", 2.) &&
            difference(matrix_from_json(best["_H_work"]),
                       matrix_from_json(candidates[i]["_H_work"]),
                       s.size) > std::min(s.size.width, s.size.height) * .12)
            alias = true;
    bool periodic = !lattice.empty() || alias,
         accepted = bs >= cfg.value("min_score", 60.) &&
                    best["target_coverage"].get<double>() >= cfg.value("min_overlap", .45);
    best.erase("_H_work");
    for (auto &c : candidates)
        c.erase("_H_work");
    best.update(Json{
        {"geometry_accepted", accepted},
        {"periodic_detected", periodic},
        {"absolute_cell_verified", accepted && !periodic && candidates.size() > 1 &&
                                       gap >= cfg.value("min_candidate_gap", 2.)},
        {"absolute_identity_status", periodic ? "unverified_periodic" : "visual_evidence_only"},
        {"candidate_score_gap", candidates.size() > 1 ? Json(gap) : Json(nullptr)},
        {"periodic_alias_detected", alias},
        {"lattice_candidates", lattice},
        {"working_scale", t.scale},
        {"working_target_size", {t.size.width, t.size.height}},
        {"candidates", candidates},
        {"reasons",
         accepted ? Json::array() : Json::array({"score_or_coverage_below_threshold"})}});
    if (periodic)
        best["reasons"].push_back("absolute_period_index_requires_external_anchor");
    return best;
}
} // namespace
Json select_templates(const std::vector<cv::Mat> &templates, const cv::Mat &target,
                      const Config &cfg, const std::vector<cv::Mat> &valids,
                      const cv::Mat &target_valid, const std::string &mode) {
    validate_config(cfg);
    if (templates.empty() || (mode != "semantic" && mode != "instance"))
        throw std::invalid_argument("Provide templates and label_mode semantic/instance");
    if (!valids.empty() && valids.size() != templates.size())
        throw std::invalid_argument("One valid mask per template is required");
    cv::setRNGSeed(cfg.value("seed", 7));
    cv::setNumThreads(2);
    std::vector<cv::Size> sizes{target.size()};
    for (auto &im : templates)
        sizes.push_back(im.size());
    double scale = matching_scale(cfg, sizes);
    auto dst = prepare(target, target_valid, cfg, scale);
    Json lattice = dst.f ? periods(dst.f->norm, dst.f->mask) : Json::array(),
         matches = Json::array();
    std::vector<Json> results(templates.size());
    std::atomic<size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto task = [&]() {
        try {
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= templates.size())
                    break;
                auto src =
                    prepare(templates[i], valids.empty() ? cv::Mat() : valids[i], cfg, scale);
                Json result = registration(src, dst, cfg, lattice);
                result["template_index"] = i;
                results[i] = std::move(result);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error)
                error = std::current_exception();
        }
    };
    unsigned requested = cfg.value("template_workers", 0);
    unsigned workers = std::max(
        1u, std::min({unsigned(templates.size()), std::max(1u, std::thread::hardware_concurrency()),
                      requested ? requested : 4u}));
    std::vector<std::thread> pool;
    for (unsigned i = 1; i < workers; ++i)
        pool.emplace_back(task);
    task();
    for (auto &thread : pool)
        thread.join();
    if (error)
        std::rethrow_exception(error);
    for (auto &result : results)
        matches.push_back(std::move(result));
    std::vector<int> ranking(templates.size());
    std::iota(ranking.begin(), ranking.end(), 0);
    std::stable_sort(ranking.begin(), ranking.end(), [&](int a, int b) {
        return matches[a]["match_score"].get<double>() > matches[b]["match_score"].get<double>();
    });
    auto best = matches[ranking[0]];
    double gap = ranking.size() > 1 ? best["match_score"].get<double>() -
                                          matches[ranking[1]]["match_score"].get<double>()
                                    : 0;
    bool chosen = best["geometry_accepted"],
         separated = ranking.size() == 1 || gap >= cfg.value("min_template_gap", 3.),
         visual = chosen && separated,
         absolute = visual && best["absolute_cell_verified"].get<bool>(),
         accepted = mode == "instance" ? absolute : visual;
    return {{"best_template_index", chosen ? Json(ranking[0]) : Json(nullptr)},
            {"accepted", accepted},
            {"status", accepted     ? "accepted"
                       : !chosen    ? "low_match_score"
                       : !separated ? "templates_too_close"
                                    : "absolute_cell_unverified"},
            {"label_mode", mode},
            {"template_score_gap", ranking.size() > 1 ? Json(gap) : Json(nullptr)},
            {"absolute_mapping_accepted", absolute},
            {"ranking", ranking},
            {"matches", matches},
            {"config", cfg},
            {"score_definition", "100*(0.55*structure_ncc+0.30*edge_alignment+0.15*local_"
                                 "consistency)*sqrt(target_coverage)"},
            {"score_is_probability", false}};
}
} // namespace ptm
