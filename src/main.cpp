#include <iostream>

#include <Eigen/Dense>

#include <numeric>
#include <algorithm>
#include <random>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>

#include "../include/extra.h"
#include "../include/activation_functions.h"
#include "../include/layer.h"
#include "../include/mnist_loader.h"
#include "../include/network.h"

//Training Data
const std::string train_data_path = "data/mnist/train-images.idx3-ubyte";
const std::string train_label_path = "data/mnist/train-labels.idx1-ubyte";

//Validation Data
const std::string validation_data_path = "data/mnist/t10k-images.idx3-ubyte";
const std::string validation_label_path = "data/mnist/t10k-labels.idx1-ubyte";

//USPS testing Data
const std::string usps_data = "data/usps/usps_images.bin";
const std::string usps_labels = "data/usps/usps_labels.bin";
const std::string usps_test_data = "data/usps/usps_t_images.bin";
const std::string usps_test_labels = "data/usps/usps_t_labels.bin";

//Family testing Data
const std::string dad_images = "data/family/dad_images.idx3-ubyte";
const std::string dad_labels = "data/family/dad_labels.idx1-ubyte";

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

std::string makeResultsDir() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std:tm* tm = std::localtime(&t);

    std::ostringstream oss;
    oss << "test_results/" << std::put_time(tm, "%Y-%m-%d_%H-%M-%S");

    std::string path = oss.str();
    std::filesystem::create_directories(path);
    return path;
}

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
    printLogo();                                                   
                                                                                            
    std::cout << "Loading training data set\n";
    load_data::Dataset train_data = load_data::read_input(train_data_path, train_label_path);

    std::cout << "Loading validation data set\n";
    load_data::Dataset validation_data = load_data::read_input(validation_data_path, validation_label_path);

    std::cout << "Loading USPS data set\n";
    load_data::Dataset test_data = load_data::read_input(usps_data, usps_labels);

    std::cout << "Loading USPS test data set\n";
    load_data::Dataset test_2_data = load_data::read_input(usps_test_data, usps_test_labels); 

    std::cout << "Loading Dad's handwriting set\n";
    load_data::Dataset dad = load_data::read_input(dad_images, dad_labels);

    Network network;
    network.add_layer(784, 256, relu, relu_back);
    network.add_layer(256, 256, relu, relu_back);
    network.add_layer(256, 64, relu, relu_back);
    network.add_layer(64, 10, softmax);

    // network.add_layer(784, 256, relu, relu_back);
    // network.add_layer(256, 256, relu, relu_back);
    // network.add_layer(256, 10, softmax);

    // network.add_layer(784, 256, relu, relu_back);
    // network.add_layer(256, 10, softmax);

    // network.add_layer(784, 10, softmax);

    float learning_rate = 0.07f; //0.1
    float decay_rate = 0.5f; //0.5
    float min_learning_rate = 0.0001f;
    int epochs_per_decay = 5;
    int batch_size = 28; //32
    int epochs = 15;
    
    bool do_batch_train = true;
    
    std::cout << "Starting Learning rate: " << learning_rate << "\n\n";

    //For shuffling the training data (need to shuffle indicies rather than the items themselves because labels and images need to stay together)
    std::vector<int> index(train_data.images.size());
    std::iota(index.begin(), index.end(), 0); //fills indecies

    for(int i = 1; i <= epochs; i++) {
        std::cout << "Epoch [" << i << "/" << epochs << "]\n"; 

        std::shuffle(index.begin(), index.end(), std::mt19937{std::random_device{}()}); //randomizes indecies each epoch

        learning_rate = step_learning_rate(learning_rate, decay_rate, min_learning_rate, epochs_per_decay, i);

        if(do_batch_train) network.train_all_batch(train_data.images, train_data.labels, index, learning_rate, batch_size);
        else network.train_all(train_data.images, train_data.labels, index, learning_rate);
        
        std::cout << std::endl;
        float accuracy = network.eval(validation_data.images, validation_data.labels) * 100;
        std::cout << "Validation accuracy: " << accuracy << "%\n\n";
    }

    network.results_path = makeResultsDir();
    //network.test(test_data.images, test_data.labels); 
    //network.test(test_2_data.images, test_2_data.labels);
    network.test(dad.images, dad.labels);
    network.save_results();

    return 0;
}