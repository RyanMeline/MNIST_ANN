/*

Vector of layers
Start forward pass (output of one layer into the next)
loss calculation
backpropogation (chain rule walking backwards through layers)

*/
#include "../include/network.h"


void Network::add_layer(int inputs, int outputs, std::function<Eigen::VectorXf(const Eigen::VectorXf&)> act, std::function<Eigen::VectorXf(const Eigen::VectorXf&)> act_derivative) {
    layers.emplace_back(inputs, outputs, act, act_derivative);
}
//  Input -> Hidden -> Output
Eigen::VectorXf Network::forward(const Eigen::VectorXf& input) {
    Eigen::VectorXf temp = input;
    for(auto& layer : layers) {
        temp = layer.forward(temp);
    }
    return temp;
}

//initialize with far right, which is softmax's gradient (predicted - labeled)
//  Input <- Hidden <- Output
void Network::backward(const Eigen::VectorXf& gradient, float learning_rate) {
    Eigen::VectorXf temp = gradient;
    for(int i = static_cast<int>(layers.size())-1; i >= 0; i--) {
        temp = layers[i].backward(temp, learning_rate);
    }
}

//Call in a loop in main to iterate through all 60k images
void Network::train(const Eigen::VectorXf& input, uint8_t label, float learning_rate) {
    Eigen::VectorXf prediction = forward(input);
    Eigen::VectorXf true_label(10); //size 0-9
    true_label.setZero(); //set all to 0
    true_label[label] = 1; //set label index to 1
    backward(prediction - true_label, learning_rate); 
}

int Network::predict(const Eigen::VectorXf& input) {
    Eigen::Index max_index;
    forward(input).maxCoeff(&max_index);
    return static_cast<int>(max_index);
}

float Network::eval(const std::vector<Eigen::VectorXf>& images, const std::vector<uint8_t>& labels) {
    int correct = 0;
    for(size_t i = 0; i < labels.size(); i++){
        if(predict(images[i]) == static_cast<int>(labels[i])) correct++;
        if((i+1)%1000 == 0) { std::cout << "\r Testing Images: [" << i+1 << "/" << images.size() << "]"; }
    }
    std::cout << std::endl;
    return static_cast<float>(correct) / static_cast<float>(labels.size());
}
