#pragma once

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>

namespace activation_function {
    float relu(float);
    float relu_back(float);
    Eigen::VectorXf softmax(const Eigen::VectorXf& input);
}