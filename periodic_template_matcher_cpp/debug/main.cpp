#include "ptm/core.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>

namespace {
cv::Mat read_array(const ptm::Json &j, const std::filesystem::path &base) {
    if (j.is_null())
        return {};
    int rows = j.at("rows"), cols = j.at("cols"), type = j.at("type");
    if (rows <= 0 || cols <= 0 || CV_MAT_DEPTH(type) > CV_64F)
        throw std::runtime_error("Invalid fixture array");
    cv::Mat m(rows, cols, type);
    std::ifstream f(base / j.at("file").get<std::string>(), std::ios::binary);
    f.read(reinterpret_cast<char *>(m.data), std::streamsize(m.total() * m.elemSize()));
    if (!f)
        throw std::runtime_error("Cannot read fixture array");
    return m;
}
} // namespace
int main(int argc, char **argv) {
    try {
        ptm::Inputs inputs;
        auto cfg = ptm::default_config();
        if (argc > 1) {
            std::filesystem::path path(argv[1]);
            std::ifstream stream(path);
            ptm::Json fixture;
            stream >> fixture;
            cfg.update(fixture.at("config"));
            const auto &j = fixture.at("inputs");
            auto base = path.parent_path();
            inputs.target = read_array(j.at("target"), base);
            inputs.target_valid = read_array(j.at("target_valid"), base);
            inputs.regions = j.at("regions");
            for (auto x : j.at("templates"))
                inputs.templates.push_back(read_array(x, base));
            for (auto x : j.at("labels"))
                inputs.labels.push_back(read_array(x, base));
            for (auto x : j.at("valids"))
                inputs.template_valids.push_back(read_array(x, base));
            for (auto x : j.at("annotations"))
                inputs.annotation_valids.push_back(read_array(x, base));
            inputs.has_objects = !j.at("objects").is_null();
            if (inputs.has_objects)
                for (auto row : j["objects"]) {
                    std::vector<ptm::ObjectRef> refs;
                    for (auto ref : row) {
                        ptm::ObjectRef r;
                        r.identity = ref["identity"];
                        r.mask = read_array(ref["mask"], base);
                        r.background_known = ref.value("background_known", true);
                        r.shape_index = ref.value("shape_index", -1);
                        refs.push_back(r);
                    }
                    inputs.objects.push_back(refs);
                }
        } else {
            cv::Mat image(96, 112, CV_8U);
            cv::RNG(42).fill(image, cv::RNG::UNIFORM, 0, 256);
            inputs.templates = {image};
            inputs.target = image.clone();
            cv::Mat labels = cv::Mat::zeros(image.size(), CV_16U);
            labels(cv::Rect(30, 30, 20, 20)).setTo(2);
            inputs.labels = {labels};
            inputs.template_valids = {cv::Mat::ones(image.size(), CV_8U)};
            inputs.annotation_valids = {cv::Mat()};
            inputs.target_valid = inputs.template_valids[0].clone();
            cfg["angles"] = {0.};
            cfg["scales"] = {1.};
            cfg["template_workers"] = 1;
        }
        // Set a source breakpoint here, or inside ptm::match/features/score.
        const auto result = ptm::match(inputs, cfg);
        std::cout << result.registration.dump(2) << std::endl;
        if (argc == 1 && !result.registration.at("accepted").get<bool>())
            return 2;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Native matcher failed: " << e.what() << std::endl;
        return 1;
    }
}
