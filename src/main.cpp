#include <iostream>

#include <Eigen/Dense>

#include <numeric>
#include <algorithm>
#include <random>

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

using namespace::activation_function;

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
    load_data::Dataset train_data = load_data::read_input(train_data_path, train_label_path);

    std::cout << "Loading test data set\n";
    load_data::Dataset test_data = load_data::read_input(test_data_path, test_label_path);

    Network network;
    network.add_layer(784, 256, relu, relu_back);
    network.add_layer(256, 128, relu, relu_back);
    network.add_layer(128, 64, relu, relu_back);
    network.add_layer(64, 10, softmax);

    float learning_rate = 0.01f;
    float decay_rate = 0.1f;
    std::cout << "Starting Learning rate: " << learning_rate << "\n\n";
    int epochs = 20;

    //For shuffling the training data (need to shuffle indicies rather than the items themselves because labels and images need to stay together)
    std::vector<int> index(train_data.images.size());
    std::iota(index.begin(), index.end(), 0); //fills indecies

    for(int i = 1; i <= epochs; i++) {
        std::cout << "Epoch [" << i << "/" << epochs << "]\n"; 

        std::shuffle(index.begin(), index.end(), std::mt19937{std::random_device{}()}); //randomizes indecies each epoch

        if(i % 5 == 0) {  //Learning Rate Scheduler
            learning_rate *= decay_rate;
            std::cout << "New Learning Rate: " << learning_rate << "\n"; 
        }

        for(size_t j = 0; j < train_data.images.size(); j++) {
            network.train(train_data.images[index[j]], train_data.labels[index[j]], learning_rate);

            if((j+1)%100 == 0) {
                std::cout << "\r Training Images: [" << j+1 << "/" << train_data.images.size() << "]";
            }
        }
        std::cout << std::endl;
        float accuracy = network.eval(test_data.images, test_data.labels) * 100;
        std::cout << "Accuracy: " << accuracy << "%\n\n";
    }

    return 0;
}