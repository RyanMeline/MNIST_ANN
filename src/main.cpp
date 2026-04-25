#include <iostream>

#include <Eigen/Dense>

#include "../include/activation_functions.h"
#include "../include/layer.h"
#include "../include/mnist_loader.h"
#include "../include/network.h"

const std::string train_data_path = "data/mnist/train-images.idx3-ubyte";
const std::string train_label_path = "data/mnist/train-labels.idx1-ubyte";
const std::string test_data_path = "data/mnist/t10k-images.idx3-ubyte";
const std::string test_label_path = "data/mnist/t10k-labels.idx1-ubyte";

int main() {
    std::cout << "Loading training data set\n";
    load_data::Dataset train_data = load_data::read_input(train_data_path, train_label_path);
    std::cout << "Loading test data set\n";
    load_data::Dataset test_data = load_data::read_input(test_data_path, test_label_path);
    return 0;
}