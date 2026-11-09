#include "ptm/core.hpp"
#include <iostream>
#include <stdexcept>

template <typename F> void requires_invalid_argument(F call) {
    try {
        call();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("Malformed native inputs were accepted");
}

int main() {
    auto cfg = ptm::default_config();
    cfg["ordered_object_class"] = 2;
    cfg["object_order_axis"] = "y";
    cv::Mat image = cv::Mat::ones(64, 64, CV_8U);
    std::vector<cv::Mat> images{image}, valids{image};
    std::vector<std::vector<ptm::ObjectRef>> refs{{{"A", image, true, 0}}};
    requires_invalid_argument([&] {
        ptm::match_ordered_objects(images, refs, valids, image, image, {}, cfg);
    });
    ptm::ObjectPrediction prediction{cv::Mat::zeros(image.size(), CV_16U),
                                     cv::Mat::zeros(image.size(), CV_16U), image, {}};
    requires_invalid_argument([&] {
        ptm::complete_object_support(prediction, images, {}, valids, image, image, {}, cfg);
    });
    std::cout << "PASS: native object entry points reject inconsistent input counts\n";
}
