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

//For testing
void print_image(const Eigen::VectorXf& image) {
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            float pixel = image(i * 28 + j);
            if(pixel > 0.75f)      std::cout << "@";
            else if(pixel > 0.5f)  std::cout << "#";
            else if(pixel > 0.25f) std::cout << ".";
            else                   std::cout << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    std::cout <<"\n" <<
    "____________________________________________________\n\n" <<
    " /$$      /$$ /$$   /$$ /$$$$$$  /$$$$$$  /$$$$$$$$" << std::endl <<
    "| $$$    /$$$| $$$ | $$|_  $$_/ /$$__  $$|__  $$__/" << std::endl <<
    "| $$$$  /$$$$| $$$$| $$  | $$  | $$  \\__/   | $$   " << std::endl <<
    "| $$ $$/$$ $$| $$ $$ $$  | $$  |  $$$$$$    | $$   " << std::endl <<
    "| $$  $$$| $$| $$  $$$$  | $$   \\____  $$   | $$   " << std::endl <<
    "| $$\\  $ | $$| $$\\  $$$  | $$   /$$  \\ $$   | $$   " << std::endl <<
    "| $$ \\/  | $$| $$ \\  $$ /$$$$$$|  $$$$$$/   | $$   " << std::endl <<
    "|__/     |__/|__/  \\__/|______/ \\______/    |__/   " << std::endl <<
    "____________________________________________________\n\n";
                                                   
                                                   
                                                   
    std::cout << "Loading training data set\n";
    //load_data::Dataset train_data = load_data::read_input(train_data_path, train_label_path);
    // std::cout << "\nLabel: " << (int)train_data.labels[0];
    // print_image(train_data.images[0]);
    std::cout << "Loading test data set\n";
    //load_data::Dataset test_data = load_data::read_input(test_data_path, test_label_path);
    // std::cout << "\nLabel: " << (int)test_data.labels[0];
    // print_image(test_data.images[0]);


    return 0;
}