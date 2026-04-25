#pragma once

// Needed for ntohl because MNIST is in network byte order
#ifdef _WIN32
  #include <winsock2.h>
#else
  #include <arpa/inet.h>
#endif

#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace load_data {
    struct Dataset {
        std::vector<Eigen::VectorXf> images;
        std::vector<uint8_t> labels;
    };
    Dataset read_input(const std::string& images_path, const std::string& labels_path);
    std::vector<Eigen::VectorXf> read_images(const std::string& path);
    std::vector<uint8_t> read_labels(const std::string& path);
}
