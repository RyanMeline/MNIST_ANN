/*

Wright matrix W (outputs x inputs) and bias vector b

forward pass ->
        z = W*x + b;
        a = activation_function(z);

Also store data for backprop
Need to cache:
        z -> pre-activation_function
        a -> post-activation_function

W and b update via gradiant descent (handled in network.cpp)
*/
#include "../include/layer.h"

Layer::Layer(int inputs, int outputs,
             std::function<Eigen::VectorXf(const Eigen::VectorXf&)> act,
             std::function<Eigen::VectorXf(const Eigen::VectorXf&)> act_deriv)
    : activation(act), activation_derivative(act_deriv) {
        //He initilization for ReLU
    W = (Eigen::MatrixXf::Random(outputs, inputs) * std::sqrt(2.0f / inputs));
    b = Eigen::VectorXf::Zero(outputs);
}

Eigen::VectorXf Layer::forward(const Eigen::VectorXf& input) {
    x = input;
    z = W*x + b; //Eigen handles the matrix x vector math
    return activation(z);
}

Eigen::VectorXf Layer::backward(const Eigen::VectorXf& gradient, float learning_rate) {
    Eigen::VectorXf dLdz = gradient.array() * activation_derivative(z).array(); //Make the activation derivative function
    Eigen::MatrixXf dLdW = dLdz.matrix() * x.matrix().transpose(); //outer product, flips the x matrix
    Eigen::VectorXf dLdb = dLdz.matrix();
    Eigen::VectorXf dLdx = W.transpose() * dLdz.matrix();

    //Update weights
    W = W - learning_rate * dLdW;
    b = b - learning_rate * dLdb;

    return dLdx;
}