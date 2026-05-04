#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <array>
#include <cstdint>
#include <winsock2.h>

float cubic_hermite(float a, float b, float c, float d, float t) {
    float a0 = -0.5f*a + 1.5f*b - 1.5f*c + 0.5f*d;
    float a1 =  1.0f*a - 2.5f*b + 2.0f*c - 0.5f*d;
    float a2 = -0.5f*a + 0.5f*c;
    float a3 =  b;
    return a0*t*t*t + a1*t*t + a2*t + a3;
}

float sample(const float* src, int src_size, int x, int y) {
    x = std::max(0, std::min(src_size - 1, x));
    y = std::max(0, std::min(src_size - 1, y));
    return src[y * src_size + x];
}

void bicubic_resize(const float* src, int src_size, float* dst, int dst_size) {
    float scale = (float)(src_size) / (float)(dst_size);
    for (int dy = 0; dy < dst_size; dy++) {
        for (int dx = 0; dx < dst_size; dx++) {
            float sx = (dx + 0.5f) * scale - 0.5f;
            float sy = (dy + 0.5f) * scale - 0.5f;
            int ix = (int)floor(sx);
            int iy = (int)floor(sy);
            float fx = sx - ix;
            float fy = sy - iy;

            float rows[4];
            for (int row = -1; row <= 2; row++) {
                float p[4];
                for (int col = -1; col <= 2; col++) {
                    p[col + 1] = sample(src, src_size, ix + col, iy + row);
                }
                rows[row + 1] = cubic_hermite(p[0], p[1], p[2], p[3], fx);
            }
            float val = cubic_hermite(rows[0], rows[1], rows[2], rows[3], fy);
            val = std::max(0.0f, std::min(1.0f, val));
            dst[dy * dst_size + dx] = val;
        }
    }
}

void normalize_digit(const float* src16, float* dst28) {
    // Step 1: bicubic resize 16x16 -> 20x20
    float img20[400] = {};
    bicubic_resize(src16, 16, img20, 20);

    // Step 2: compute center of mass of the 20x20 image
    float cm_x = 0.0f, cm_y = 0.0f, cm_sum = 0.0f;
    for (int y = 0; y < 20; y++) {
        for (int x = 0; x < 20; x++) {
            float v = img20[y * 20 + x];
            cm_sum += v;
            cm_x += v * x;
            cm_y += v * y;
        }
    }
    if (cm_sum > 0.0f) { cm_x /= cm_sum; cm_y /= cm_sum; }
    else { cm_x = 10.0f; cm_y = 10.0f; }

    // Step 3: place onto 28x28 canvas, offset so center of mass lands at (14,14)
    std::fill(dst28, dst28 + 784, 0.0f);
    int offset_x = (int)std::round(14.0f - cm_x);
    int offset_y = (int)std::round(14.0f - cm_y);

    for (int y = 0; y < 20; y++) {
        for (int x = 0; x < 20; x++) {
            int px = x + offset_x;
            int py = y + offset_y;
            if (px >= 0 && px < 28 && py >= 0 && py < 28)
                dst28[py * 28 + px] = img20[y * 20 + x];
        }
    }
}

void write_uint32_be(std::ofstream& f, uint32_t val) {
    val = htonl(val);
    f.write((char*)&val, sizeof(val));
}

void convert_usps(const std::string& usps_path,
                  const std::string& out_images_path,
                  const std::string& out_labels_path) {
    std::ifstream in(usps_path);
    if (!in.is_open()) { std::cerr << "Failed to open USPS file\n"; return; }

    std::vector<uint8_t> labels;
    std::vector<std::array<float, 784>> images;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);

        // Read label
        float label_f;
        ss >> label_f;
        uint8_t label = ((uint8_t)std::round(label_f) - 1) % 10;
        labels.push_back(label);

        // Read index:value pairs
        float src[256] = {};
        std::string token;
        while (ss >> token) {
            int colon = token.find(':');
            if (colon == std::string::npos) continue;
            int idx = std::stoi(token.substr(0, colon)) - 1;
            float val = std::stof(token.substr(colon + 1));
            if (idx >= 0 && idx < 256)
                src[idx] = (val + 1.0f) / 2.0f;
        }

        std::array<float, 784> dst;
        normalize_digit(src, dst.data());
        images.push_back(dst);
    }

    uint32_t n = images.size();
    std::cout << "Converting " << n << " USPS images...\n";

    // Write images file
    std::ofstream img_out(out_images_path, std::ios::binary);
    write_uint32_be(img_out, 2051);
    write_uint32_be(img_out, n);
    write_uint32_be(img_out, 28);
    write_uint32_be(img_out, 28);
    for (auto& img : images) {
        for (float v : img) {
            uint8_t pixel = (uint8_t)(v * 255.0f);
            img_out.write((char*)&pixel, 1);
        }
    }

    // Write labels file
    std::ofstream lbl_out(out_labels_path, std::ios::binary);
    write_uint32_be(lbl_out, 2049);
    write_uint32_be(lbl_out, n);
    for (uint8_t l : labels) {
        lbl_out.write((char*)&l, 1);
    }

    std::cout << "Done. Written to:\n  " << out_images_path << "\n  " << out_labels_path << "\n";
}

int main() {
    convert_usps("usps_raw/usps",   "../data/usps/usps_images.bin",   "../data/usps/usps_labels.bin");
    convert_usps("usps_raw/usps_test", "../data/usps/usps_t_images.bin", "../data/usps/usps_t_labels.bin");
    return 0;
}