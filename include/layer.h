#pragma once
#include <Eigen/Dense>
#include <functional>
#include <iostream>

class Layer {
    public:
        Layer(int inputs, int outputs, 
              std::function<Eigen::VectorXf(const Eigen::VectorXf&)> act,
              std::function<Eigen::VectorXf(const Eigen::VectorXf&)> act_derivative 
                = [](const Eigen::VectorXf& x) { return Eigen::VectorXf::Ones(x.size()); }); //set a default for output layer (might be an issue since it could just square everything, might need to just return 1)
        Eigen::VectorXf forward(const Eigen::VectorXf& input);
        Eigen::VectorXf backward(const Eigen::VectorXf& gradient, float learning_rate);

    private:
        Eigen::MatrixXf W; //weights
        Eigen::VectorXf b; //biases
        Eigen::VectorXf x; //stored input
        Eigen::VectorXf z; //stored pre-activation
        std::function<Eigen::VectorXf(const Eigen::VectorXf&)> activation;
        std::function<Eigen::VectorXf(const Eigen::VectorXf&)> activation_derivative;
};