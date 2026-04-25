#pragma once

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>

namespace activation_function {
    // float relu(float);
    // float relu_back(float);
    Eigen::VectorXf relu(const Eigen::VectorXf& input);
    Eigen::VectorXf relu_back(const Eigen::VectorXf& input);
    Eigen::VectorXf softmax(const Eigen::VectorXf& input);
}