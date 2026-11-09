#include "ptm/core.hpp"
#include <algorithm>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <set>
namespace py = pybind11;
using namespace ptm;
namespace {
Json json(py::handle x) {
    return Json::parse(py::module_::import("json").attr("dumps")(x).cast<std::string>());
}
py::object object(const Json &x) {
    return py::module_::import("json").attr("loads")(x.dump());
}
Config config(py::handle x) {
    try {
        auto c = default_config();
        if (!x.is_none())
            c.update(json(x));
        validate_config(c);
        return c;
    } catch (const Json::exception &e) {
        throw py::value_error(e.what());
    }
}
bool sequence(py::handle x) {
    return PySequence_Check(x.ptr()) && !py::isinstance<py::str>(x) &&
           !py::isinstance<py::array>(x);
}
cv::Mat mat(py::handle x) {
    if (x.is_none())
        return {};
    if (!py::isinstance<py::array>(x))
        throw py::value_error("Expected numpy array");
    // Own a deep copy before releasing the GIL: ensure() may create a temporary
    // contiguous owner for sliced/reversed inputs, and callers retain their data.
    auto a = py::array::ensure(x, py::array::c_style);
    if (!a)
        throw py::value_error("Cannot normalize array");
    auto b = a.request();
    if (b.ndim < 2 || b.ndim > 3)
        throw py::value_error("Expected 2D or 3D array");
    int depth = -1;
    if (a.dtype().is(py::dtype::of<bool>()) || a.dtype().is(py::dtype::of<uint8_t>()))
        depth = CV_8U;
    else if (a.dtype().is(py::dtype::of<uint16_t>()))
        depth = CV_16U;
    else if (a.dtype().is(py::dtype::of<int16_t>()))
        depth = CV_16S;
    else if (a.dtype().is(py::dtype::of<int32_t>()))
        depth = CV_32S;
    else if (a.dtype().is(py::dtype::of<float>()))
        depth = CV_32F;
    else if (a.dtype().is(py::dtype::of<double>()))
        depth = CV_64F;
    if (depth < 0) {
        auto d = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(a);
        if (!d)
            throw py::value_error("Array cannot be converted to numeric values");
        return cv::Mat((int)b.shape[0], (int)b.shape[1],
                       CV_MAKETYPE(CV_64F, b.ndim == 3 ? (int)b.shape[2] : 1), d.mutable_data())
            .clone();
    }
    return cv::Mat((int)b.shape[0], (int)b.shape[1],
                   CV_MAKETYPE(depth, b.ndim == 3 ? (int)b.shape[2] : 1), b.ptr)
        .clone();
}
py::object array(const cv::Mat &m, bool boolean = false) {
    if (m.empty())
        return py::none();
    py::dtype dt = boolean               ? py::dtype::of<bool>()
                   : m.depth() == CV_8U  ? py::dtype::of<uint8_t>()
                   : m.depth() == CV_16U ? py::dtype::of<uint16_t>()
                   : m.depth() == CV_16S ? py::dtype::of<int16_t>()
                   : m.depth() == CV_32S ? py::dtype::of<int32_t>()
                   : m.depth() == CV_32F ? py::dtype::of<float>()
                                         : py::dtype::of<double>();
    std::vector<py::ssize_t> shape = {m.rows, m.cols};
    if (m.channels() > 1)
        shape.push_back(m.channels());
    py::array out(dt, shape);
    if (boolean) {
        auto *p = (bool *)out.mutable_data();
        for (int y = 0; y < m.rows; y++)
            for (int x = 0; x < m.cols; x++)
                p[y * m.cols + x] = m.at<uint8_t>(y, x) != 0;
    } else {
        auto c = m.isContinuous() ? m : m.clone();
        std::memcpy(out.mutable_data(), c.data, c.total() * c.elemSize());
    }
    return out;
}
cv::Mat image(py::handle x) {
    if (!py::isinstance<py::array>(x))
        throw py::value_error("Images must be numpy arrays");
    auto a = py::cast<py::array>(x);
    if ((a.ndim() != 2 && a.ndim() != 3) || a.shape(0) < 32 || a.shape(1) < 32 ||
        !a.dtype().is(py::dtype::of<uint8_t>()) ||
        (a.ndim() == 3 && a.shape(2) != 3 && a.shape(2) != 4))
        throw py::value_error("Images must be uint8 gray, BGR or BGRA arrays at least 32 x 32");
    return mat(x);
}
cv::Mat mask(py::handle x, cv::Size size, bool optional = true) {
    if (x.is_none() && optional)
        return {};
    if (!py::isinstance<py::array>(x) || py::cast<py::array>(x).ndim() != 2)
        throw py::value_error("Mask must be 2D");
    auto a = mat(x);
    if (a.size() != size || a.channels() != 1)
        throw py::value_error("Mask must match image height and width");
    return binary(a);
}
cv::Mat labels(py::handle x, cv::Size size) {
    if (x.is_none())
        return {};
    cv::Mat out(size, CV_16U, cv::Scalar(0));
    auto read = [&](py::handle v, bool bin, int id) {
        if (!py::isinstance<py::array>(v))
            throw py::value_error("Labels must be numpy arrays");
        auto a = py::cast<py::array>(v);
        if (a.ndim() != 2)
            throw py::value_error("Masks must be 2D");
        auto kind = py::str(a.dtype().attr("kind")).cast<std::string>();
        if (kind != "b" && kind != "i" && kind != "u")
            throw py::value_error("Masks must contain integer class IDs");
        auto m = mat(v);
        if (m.size() != size || m.channels() != 1)
            throw py::value_error("Masks must match image height and width");
        cv::Mat d;
        m.convertTo(d, CV_64F);
        bool one = true, two = true;
        for (int y = 0; y < d.rows; y++)
            for (int z = 0; z < d.cols; z++) {
                double n = d.at<double>(y, z);
                if (n < 0 || n > 65535)
                    throw py::value_error("Class IDs must be in 0..65535");
                one &= n == 0 || n == 1;
                two &= n == 0 || n == 255;
            }
        if (bin && !one && !two)
            throw py::value_error("Binary masks must use 0/1 or 0/255");
        for (int y = 0; y < d.rows; y++)
            for (int z = 0; z < d.cols; z++) {
                auto n = d.at<double>(y, z);
                if (bin && n > 0) {
                    if (out.at<uint16_t>(y, z))
                        throw py::value_error("Binary class masks overlap");
                    out.at<uint16_t>(y, z) = id;
                } else if (!bin)
                    out.at<uint16_t>(y, z) = (uint16_t)n;
            }
    };
    if (py::isinstance<py::array>(x))
        read(x, false, 0);
    else {
        if (!sequence(x) || py::len(x) > 65535)
            throw py::value_error("Expected class-ID array or binary mask list");
        int id = 1;
        for (auto v : py::reinterpret_borrow<py::sequence>(x))
            read(v, true, id++);
    }
    return out;
}
std::vector<cv::Mat> mats(py::handle x) {
    std::vector<cv::Mat> r;
    for (auto a : py::reinterpret_borrow<py::sequence>(x))
        r.push_back(mat(a));
    return r;
}
Matrix matrix(py::handle x) {
    auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(x);
    if (!a || a.ndim() != 2 || a.shape(0) != 3 || a.shape(1) != 3)
        throw py::value_error("Expected 3x3 matrix");
    Matrix H;
    std::copy(a.data(), a.data() + 9, H.val);
    return H;
}
py::object matrix_array(const Matrix &H) {
    return array(cv::Mat(H));
}
void same_plane(const cv::Mat &a, const cv::Mat &b) {
    if (a.empty() || b.empty() || a.channels() != 1 || b.channels() != 1 || a.size() != b.size())
        throw py::value_error("Expected nonempty same-sized 2D arrays");
}
Features feature(py::handle x) {
    if (!sequence(x))
        throw py::value_error("Features must have four arrays");
    auto s = py::reinterpret_borrow<py::sequence>(x);
    if (py::len(s) != 4)
        throw py::value_error("Features must have four arrays");
    Features f{mat(s[0]), mat(s[1]), mat(s[2]), mat(s[3])};
    same_plane(f.norm, f.grad);
    same_plane(f.norm, f.edges);
    same_plane(f.norm, f.mask);
    if (f.edges.depth() != CV_8U || f.mask.depth() != CV_8U)
        throw py::value_error("Feature edges and mask must be uint8 or bool");
    return f;
}
std::vector<std::vector<ObjectRef>> refs(py::handle x, const std::vector<cv::Mat> &ims) {
    std::vector<std::vector<ObjectRef>> r;
    if (x.is_none() || !sequence(x))
        throw py::value_error("One object reference list per template required");
    if (py::len(x) != ims.size())
        throw py::value_error("One object reference list per template required");
    int i = 0;
    for (auto row : py::reinterpret_borrow<py::sequence>(x)) {
        std::vector<ObjectRef> rr;
        for (auto v : py::reinterpret_borrow<py::sequence>(row)) {
            if (!py::isinstance<py::dict>(v))
                throw py::value_error("Object references must be dictionaries");
            auto d = py::cast<py::dict>(v);
            if (!d.contains("identity") || !py::isinstance<py::str>(d["identity"]) ||
                !d.contains("mask"))
                throw py::value_error("Object references require string identity and mask");
            ObjectRef o;
            o.identity = py::cast<std::string>(d["identity"]);
            o.mask = mask(d["mask"], ims[i].size(), false);
            if (cv::countNonZero(o.mask) == 0)
                throw py::value_error("Object reference mask must be nonempty and source-sized");
            if (d.contains("background_known"))
                o.background_known = py::cast<bool>(d["background_known"]);
            if (d.contains("shape_index"))
                o.shape_index = py::cast<int>(d["shape_index"]);
            rr.push_back(o);
        }
        r.push_back(rr);
        i++;
    }
    return r;
}
#define FIELDS(X)                                                                                  \
    X(input_image, false)                                                                          \
    X(template_image, false) X(warped_template, false) X(geometric_valid, true)                    \
        X(mapped_labels, false) X(label_valid, true) X(completed_template, false)                  \
            X(completed_valid, true) X(periodic_valid, true) X(label_periodic_valid, true)         \
                X(object_identities, false) X(object_valid, true) X(base_labels, false)            \
                    X(base_valid, true)
py::dict result(const MatchResult &r) {
    py::dict d;
    d["registration"] = object(r.registration);
#define OUT(n, b) d[#n] = array(r.n, b);
    FIELDS(OUT)
#undef OUT
    return d;
}
MatchResult result_in(py::dict d) {
    MatchResult r;
    if (d.contains("registration"))
        r.registration = json(d["registration"]);
#define IN(n, b)                                                                                   \
    if (d.contains(#n))                                                                            \
        r.n = mat(d[#n]);
    FIELDS(IN)
#undef IN
    return r;
}
py::dict prediction(const ObjectPrediction &r) {
    py::dict d;
    d["labels"] = array(r.labels);
    d["identities"] = array(r.identities);
    d["valid"] = array(r.valid, true);
    d["diagnostics"] = object(r.diagnostics);
    return d;
}
std::optional<int> height(py::handle x, const Config &c, int h) {
    py::object owned;
    if (x.is_none()) {
        if (c["valid_height"].is_null())
            return {};
        owned = py::int_(c["valid_height"].get<int>());
        x = owned;
    }
    if (!py::isinstance<py::int_>(x) || py::isinstance<py::bool_>(x))
        throw py::value_error("valid_height must be an integer");
    int n = py::cast<int>(x);
    if (n < 1 || n > h)
        throw py::value_error("valid_height outside image");
    return n;
}
Inputs prepare(py::handle ts, py::handle ms, py::handle target, const Config &c, py::handle vs,
               py::handle anns, py::handle tv, py::handle hs, py::handle th, py::handle os) {
    if (!sequence(ts) || py::len(ts) == 0)
        throw py::value_error("templates must be a nonempty list of images");
    size_t n = py::len(ts);
    if (!sequence(ms) || py::len(ms) != n)
        throw py::value_error("masks must have one entry per template");
    for (auto p : {vs, anns, hs})
        if (!p.is_none() && (!sequence(p) || py::len(p) != n))
            throw py::value_error("Expected one entry per template");
    Inputs r;
    r.target = image(target);
    r.regions["templates"] = Json::array();
    auto region = [&](cv::Mat im, py::handle v, py::handle h, Json &info) {
        auto end = height(h, c, im.rows);
        auto m = roi_mask(im.size(), mask(v, im.size()), end);
        int count = cv::countNonZero(m);
        if (count < 1024)
            throw py::value_error("Too few valid image pixels");
        info = {{"image_height", im.rows},
                {"image_width", im.cols},
                {"valid_height", end.value_or(im.rows)},
                {"valid_pixels", count},
                {"height_source", end ? "explicit" : "full_image"}};
        return m;
    };
    // Do not use a conditional expression between py::none and an accessor:
    // pybind11 can then try to cast every non-None value to NoneType.
    auto at = [](py::handle s, size_t i) -> py::object {
        if (s.is_none())
            return py::none();
        return py::reinterpret_borrow<py::sequence>(s)[i];
    };
    for (size_t i = 0; i < n; i++) {
        auto im = image(at(ts, i));
        r.templates.push_back(im);
        r.labels.push_back(labels(at(ms, i), im.size()));
        Json info;
        r.template_valids.push_back(region(im, at(vs, i), at(hs, i), info));
        r.regions["templates"].push_back(info);
        r.annotation_valids.push_back(mask(at(anns, i), im.size()));
    }
    r.target_valid = region(r.target, tv, th, r.regions["target"]);
    if (!c["ordered_object_class"].is_null()) {
        if (os.is_none())
            throw py::value_error("ordered matching requires template_objects for every template");
        r.objects = refs(os, r.templates);
        r.has_objects = true;
        auto cycle = c["object_cycle"].get<std::vector<std::string>>();
        for (size_t i = 0; i < n; i++) {
            auto &row = r.objects[i];
            std::set<std::string> names;
            for (auto &o : row) {
                if (!names.insert(o.identity).second ||
                    std::find(cycle.begin(), cycle.end(), o.identity) == cycle.end())
                    throw py::value_error(
                        "Object references must have unique configured identities");
                o.background_known = true;
                if (!r.annotation_valids[i].empty()) {
                    cv::Mat missing;
                    cv::bitwise_and(r.template_valids[i], 1 - r.annotation_valids[i], missing);
                    o.background_known = cv::countNonZero(missing) == 0;
                }
            }
            if (c["object_visible_count"].is_null()) {
                if (names.size() != cycle.size())
                    throw py::value_error(
                        "Each template must provide each ordered identity exactly once");
            } else if (!row.empty()) {
                std::vector<std::pair<double, std::string>> order;
                bool x = c["object_order_axis"] == "x";
                for (auto &o : row) {
                    std::vector<cv::Point> pts;
                    cv::findNonZero(o.mask, pts);
                    auto box = cv::boundingRect(pts);
                    order.push_back(
                        {x ? box.x + (box.width - 1) / 2. : box.y + (box.height - 1) / 2.,
                         o.identity});
                }
                std::stable_sort(order.begin(), order.end(),
                                 [](auto &a, auto &b) { return a.first < b.first; });
                bool ok = false;
                for (size_t start = 0; start < cycle.size(); start++) {
                    bool valid = true;
                    for (size_t j = 0; j < order.size(); j++)
                        valid &= order[j].second == cycle[(start + j) % cycle.size()];
                    ok |= valid;
                }
                if (!ok)
                    throw py::value_error("Partial template references must form a contiguous "
                                          "cyclic window along order_axis");
            }
        }
    }
    return r;
}
} // namespace
PYBIND11_MODULE(_native, m) {
    m.attr("opencv_version") = CV_VERSION;
    m.def("raw_ecc", [](py::object target, py::object warped, py::object valid, int iterations) {
        auto t = mat(target), w = mat(warped), v = mat(valid);
        cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
        double rho;
        {
            py::gil_scoped_release release;
            rho = cv::findTransformECC(
                t, w, W, cv::MOTION_AFFINE,
                {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, iterations, 1e-5}, v * 255, 5);
        }
        return py::make_tuple(array(W), rho);
    });
    m.def("correlate", [](py::object a, py::object b, std::array<int, 2> shape) {
        auto x = mat(a), y = mat(b);
        cv::Mat r;
        {
            py::gil_scoped_release release;
            r = correlate(x, y, {shape[1], shape[0]});
        }
        return array(r);
    });
    m.def(
        "warp",
        [](py::object a, py::object h, std::array<int, 2> shape, int interpolation) {
            auto x = mat(a);
            auto H = matrix(h);
            cv::Mat r;
            {
                py::gil_scoped_release release;
                r = warp(x, H, {shape[1], shape[0]}, interpolation);
            }
            return array(r);
        },
        py::arg("array"), py::arg("matrix"), py::arg("shape"),
        py::arg("interpolation") = int(cv::INTER_LINEAR));
    m.def("default_config", [] { return object(default_config()); });
    m.def("validate_config", [](py::object c) { config(c); });
    m.def("pyround", &pyround);
    m.def("percentile", &percentile);
    m.def("features", [](py::object g, py::object v) {
        auto a = mat(g), b = mat(v);
        same_plane(a, b);
        if (a.depth() != CV_8U || b.depth() != CV_8U)
            throw py::value_error("Features require uint8 gray and uint8/bool mask");
        Features f;
        {
            py::gil_scoped_release release;
            f = features(a, b);
        }
        return py::make_tuple(array(f.norm), array(f.grad), array(f.edges), array(f.mask));
    });
    m.def("score", [](py::object s, py::object t, py::object h, py::object c) {
        auto a = feature(s), b = feature(t);
        auto H = matrix(h);
        auto cfg = config(c);
        Json r;
        {
            py::gil_scoped_release release;
            r = score(a, b, H, cfg);
        }
        return object(r);
    });
    m.def("refine", [](py::object s, py::object t, py::object h, py::object c) {
        auto a = feature(s), b = feature(t);
        auto H = matrix(h);
        auto cfg = config(c);
        std::pair<Matrix, bool> r;
        {
            py::gil_scoped_release release;
            r = refine(a, b, H, cfg);
        }
        return py::make_tuple(matrix_array(r.first), r.second);
    });
    m.def("translation_peaks",
          [](py::object s, py::object t, py::object sm, py::object tm, py::object c) {
              auto a = mat(s), b = mat(t), av = mat(sm), bv = mat(tm);
              auto cfg = config(c);
              std::vector<cv::Vec3d> r;
              {
                  py::gil_scoped_release release;
                  r = translation_peaks(a, b, av, bv, cfg);
              }
              py::list out;
              for (auto v : r)
                  out.append(py::make_tuple(v[0], v[1], v[2]));
              return out;
          });
    m.def("periods", [](py::object a, py::object b) {
        auto x = mat(a), y = mat(b);
        same_plane(x, y);
        Json r;
        {
            py::gil_scoped_release release;
            r = periods(x, y);
        }
        return object(r);
    });
    m.def(
        "select_templates",
        [](py::object ts, py::object t, py::object c, py::object vs, py::object tv,
           std::string mode) {
            auto ims = mats(ts);
            auto target = mat(t);
            auto cfg = config(c);
            auto valid = vs.is_none() ? std::vector<cv::Mat>(ims.size()) : mats(vs);
            auto targetv = mat(tv);
            Json r;
            {
                py::gil_scoped_release release;
                r = select_templates(ims, target, cfg, valid, targetv, mode);
            }
            return object(r);
        },
        py::arg("templates"), py::arg("target"), py::arg("cfg"), py::arg("valids") = py::none(),
        py::arg("target_valid") = py::none(), py::arg("label_mode") = "semantic");
    m.def(
        "match_ordered_objects",
        [](py::object im, py::object rr, py::object vv, py::object t, py::object tv, py::object hh,
           py::object c) {
            auto ims = mats(im), vs = mats(vv);
            if (vs.size() != ims.size())
                throw py::value_error("One source valid mask per image required");
            auto refs_ = refs(rr, ims);
            auto target = mat(t), targetv = mat(tv);
            auto cfg = config(c);
            std::vector<std::optional<Matrix>> H(ims.size());
            if (!hh.is_none()) {
                if (py::len(hh) != ims.size())
                    throw py::value_error("One matrix per image required");
                size_t i = 0;
                for (auto h : py::reinterpret_borrow<py::sequence>(hh)) {
                    if (!h.is_none())
                        H[i] = matrix(h);
                    ++i;
                }
            }
            ObjectPrediction r;
            {
                py::gil_scoped_release release;
                r = match_ordered_objects(ims, refs_, vs, target, targetv, H, cfg);
            }
            return prediction(r);
        },
        py::arg("images"), py::arg("references"), py::arg("source_valids"), py::arg("target"),
        py::arg("target_valid"), py::arg("matrices"), py::arg("cfg"));
    m.def("complete_object_support",
          [](py::dict p, py::object im, py::object rr, py::object vv, py::object t, py::object tv,
             std::vector<std::array<double, 2>> vec, py::object c) {
              ObjectPrediction r{mat(p["labels"]), mat(p["identities"]), mat(p["valid"]),
                                 json(p["diagnostics"])};
              auto ims = mats(im), vs = mats(vv);
              if (vs.size() != ims.size())
                  throw py::value_error("One source valid mask per image required");
              auto ref = refs(rr, ims);
              auto target = mat(t), targetv = mat(tv);
              auto cfg = config(c);
              std::vector<cv::Vec2d> vectors;
              for (auto v : vec)
                  vectors.emplace_back(v[0], v[1]);
              {
                  py::gil_scoped_release release;
                  complete_object_support(r, ims, ref, vs, target, targetv, vectors, cfg);
              }
              return prediction(r);
          });
    m.def("anchor_vectors", [](py::object l, py::object v, int id) {
        auto a = mat(l), b = mat(v);
        std::vector<cv::Vec2d> r;
        {
            py::gil_scoped_release release;
            r = anchor_vectors(a, b, id);
        }
        py::list out;
        for (auto p : r)
            out.append(py::make_tuple(p[0], p[1]));
        return out;
    });
    m.def("complete_periodic",
          [](py::dict p, py::object l, py::object an, py::object sv, py::object tv, py::object c) {
              auto r = result_in(p);
              auto a = mat(l), b = mat(an), s = mat(sv), t = mat(tv);
              auto cfg = config(c);
              {
                  py::gil_scoped_release release;
                  complete_periodic(r, a, b, s, t, cfg);
              }
              return result(r);
          });
    m.def(
        "evaluate_segmentation",
        [](py::object p, py::object t, py::object pv, py::object tv, bool bg) {
            auto a = mat(p), b = mat(t), av = mat(pv), bv = mat(tv);
            Json r;
            {
                py::gil_scoped_release release;
                r = evaluate_segmentation(a, b, av, bv, bg);
            }
            return object(r);
        },
        py::arg("prediction"), py::arg("truth"), py::arg("prediction_valid") = py::none(),
        py::arg("truth_valid") = py::none(), py::arg("include_background") = false);
    m.def(
        "match",
        [](py::object ts, py::object ms, py::object t, py::object c, py::object vs, py::object an,
           py::object tv, py::object hs, py::object th, py::object os) {
            auto cfg = config(c);
            auto inputs = prepare(ts, ms, t, cfg, vs, an, tv, hs, th, os);
            MatchResult r;
            {
                py::gil_scoped_release release;
                r = match(inputs, cfg);
            }
            return result(r);
        },
        py::arg("templates"), py::arg("masks"), py::arg("target"), py::arg("cfg"),
        py::arg("template_valids") = py::none(), py::arg("annotation_valids") = py::none(),
        py::arg("target_valid") = py::none(), py::arg("template_valid_heights") = py::none(),
        py::arg("target_valid_height") = py::none(), py::arg("template_objects") = py::none());
    m.def(
        "prepare_inputs",
        [](py::object ts, py::object ms, py::object t, py::object c, py::object vs, py::object an,
           py::object tv, py::object hs, py::object th, py::object os) {
            auto cfg = config(c);
            auto r = prepare(ts, ms, t, cfg, vs, an, tv, hs, th, os);
            py::dict d;
            d["target"] = array(r.target);
            d["target_valid"] = array(r.target_valid);
            d["regions"] = object(r.regions);
            auto list = [](const std::vector<cv::Mat> &v) {
                py::list l;
                for (auto &a : v)
                    l.append(array(a));
                return l;
            };
            d["templates"] = list(r.templates);
            d["labels"] = list(r.labels);
            d["valids"] = list(r.template_valids);
            d["annotations"] = list(r.annotation_valids);
            py::list rows;
            for (auto &row : r.objects) {
                py::list refs;
                for (auto &o : row) {
                    py::dict x;
                    x["identity"] = o.identity;
                    x["mask"] = array(o.mask);
                    x["background_known"] = o.background_known;
                    x["shape_index"] = o.shape_index;
                    refs.append(x);
                }
                rows.append(refs);
            }
            d["objects"] = r.has_objects ? py::object(rows) : py::none();
            return d;
        },
        py::arg("templates"), py::arg("masks"), py::arg("target"), py::arg("cfg"),
        py::arg("template_valids") = py::none(), py::arg("annotation_valids") = py::none(),
        py::arg("target_valid") = py::none(), py::arg("template_valid_heights") = py::none(),
        py::arg("target_valid_height") = py::none(), py::arg("template_objects") = py::none());
}
