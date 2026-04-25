#pragma once

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>

namespace activation_function {
    float relu(float);
    float relu_back(float);
    Eigen::VectorXf relu(Eigen::VectorXf);
    Eigen::VectorXf relu_back(Eigen::VectorXf);
    Eigen::VectorXf softmax(const Eigen::VectorXf& input);
}