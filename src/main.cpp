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

float step_learning_rate(float learning_rate, float decay_rate, float min_learning_rate, int epochs_per_decay, int epoch) {
    if(epoch % epochs_per_decay == 0) {  //Learning Rate Scheduler
        learning_rate = std::max(min_learning_rate, learning_rate * decay_rate);
        std::cout << "New Learning Rate: " << learning_rate << "\n"; 
    }
    return learning_rate;
}

float cosine_learning_rate(float learning_rate, float) {
    return 0;
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

    float learning_rate = 0.1f;
    float decay_rate = 0.5f;
    float min_learning_rate = 0.0001f;
    int epochs_per_decay = 5;
    int batch_size = 32;
    int epochs = 20;
    
    std::cout << "Starting Learning rate: " << learning_rate << "\n\n";

    //For shuffling the training data (need to shuffle indicies rather than the items themselves because labels and images need to stay together)
    std::vector<int> index(train_data.images.size());
    std::iota(index.begin(), index.end(), 0); //fills indecies

    for(int i = 1; i <= epochs; i++) {
        std::cout << "Epoch [" << i << "/" << epochs << "]\n"; 

        std::shuffle(index.begin(), index.end(), std::mt19937{std::random_device{}()}); //randomizes indecies each epoch

        learning_rate = step_learning_rate(learning_rate, decay_rate, min_learning_rate, epochs_per_decay, i);

        network.train_all_batch(train_data.images, train_data.labels, index, learning_rate, batch_size);
        std::cout << std::endl;
        float accuracy = network.eval(test_data.images, test_data.labels) * 100;
        std::cout << "Accuracy: " << accuracy << "%\n\n";
    }

    return 0;
}