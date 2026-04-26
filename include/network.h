#pragma once
#include "layer.h"
#include <vector>

class Network {
    public:
        Network() {};
        void add_layer(int inputs, int outputs,
                    std::function<Eigen::VectorXf(const Eigen::VectorXf&)> act,
                    std::function<Eigen::VectorXf(const Eigen::VectorXf&)> act_derivative
                        = [](const Eigen::VectorXf& x) { return Eigen::VectorXf::Ones(x.size()); });
        Eigen::VectorXf forward(const  Eigen::VectorXf& input);
        void backward(const Eigen::VectorXf& gradient, float learning_rate);
        void train(const Eigen::VectorXf& input, uint8_t label, float learning_rate);
        int predict(const Eigen::VectorXf& input);
        float eval(const std::vector<Eigen::VectorXf>& images, const std::vector<uint8_t>& labels);
        void train_all(const std::vector<Eigen::VectorXf>& images, const std::vector<uint8_t>& labels, const std::vector<int>& index, float learning_rate);
    private:
        std::vector<Layer> layers;
};