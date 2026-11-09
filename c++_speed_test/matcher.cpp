#include <cstddef>
#include <cstdint>
#include <limits>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

// Exhaustive grayscale matching using sum of squared differences (SSD).
// Python owns both contiguous uint8 buffers throughout this call.
EXPORT int match_ssd(const std::uint8_t* image, int height, int width,
                     const std::uint8_t* templ, int th, int tw,
                     int* best_x, int* best_y, double* best_score) {
    if (!image || !templ || !best_x || !best_y || !best_score ||
        height <= 0 || width <= 0 || th <= 0 || tw <= 0 ||
        th > height || tw > width) return 1;
    std::uint64_t best = std::numeric_limits<std::uint64_t>::max();
    for (int y = 0; y <= height - th; ++y) {
        for (int x = 0; x <= width - tw; ++x) {
            std::uint64_t score = 0;
            for (int ty = 0; ty < th; ++ty) {
                for (int tx = 0; tx < tw; ++tx) {
                    const int diff = int(image[(std::size_t(y) + ty) * width + x + tx])
                                   - int(templ[std::size_t(ty) * tw + tx]);
                    score += diff * diff;
                }
            }
            if (score < best) {
                best = score;
                *best_x = x;
                *best_y = y;
            }
        }
    }
    *best_score = static_cast<double>(best);
    return 0;
}
