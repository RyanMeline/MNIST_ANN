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

        Eigen::VectorXf backward_batch(const Eigen::VectorXf& gradient);
        void clear_weights(); //used to clear out weights between batches
        void update_weights(float learning_rate, int batch_size); //applies weights after each batch

    private:
        Eigen::MatrixXf W; //weights
        Eigen::VectorXf b; //biases
        Eigen::VectorXf x; //stored input
        Eigen::VectorXf z; //stored pre-activation

        //Used to store weight changes over the batch
        Eigen::MatrixXf dLdW_batch;
        Eigen::VectorXf dLdb_batch;

        std::function<Eigen::VectorXf(const Eigen::VectorXf&)> activation;
        std::function<Eigen::VectorXf(const Eigen::VectorXf&)> activation_derivative;

};