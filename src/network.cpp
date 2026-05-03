/*

Vector of layers
Start forward pass (output of one layer into the next)
loss calculation
backpropogation (chain rule walking backwards through layers)

*/
#include "../include/network.h"
#include <iomanip>
#include <fstream>


void Network::add_layer(int inputs, int outputs, std::function<Eigen::VectorXf(const Eigen::VectorXf&)> act, std::function<Eigen::VectorXf(const Eigen::VectorXf&)> act_derivative) {
    layers.emplace_back(inputs, outputs, act, act_derivative);
}

Eigen::VectorXf Network::forward(const Eigen::VectorXf& input) {
    Eigen::VectorXf temp = input;
    for(auto& layer : layers) {
        temp = layer.forward(temp);
    }
    return temp;
}

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
        if((i+1)%1000 == 0) { std::cout << "\r Validating... [" << i+1 << "/" << images.size() << "]"; }
    }
    std::cout << std::endl;
    return static_cast<float>(correct) / static_cast<float>(labels.size());
}

void Network::train_all(const std::vector<Eigen::VectorXf>& images, const std::vector<uint8_t>& labels, const std::vector<int>& index, float learning_rate) {
    for(size_t j = 0; j < images.size(); j++) {
        train(images[index[j]], labels[index[j]], learning_rate);

        if((j+1)%100 == 0) {
            std::cout << "\r Training..... [" << j+1 << "/" << images.size() << "]";
        }
    }
}

//////////////////////////////////////////////////
//                                              //
//                  Mini-Batch                  //
//                                              //
//////////////////////////////////////////////////

void Network::train_all_batch(const std::vector<Eigen::VectorXf>& images, const std::vector<uint8_t>& labels, const std::vector<int>& index, float learning_rate, int batch_size) {
    for(size_t i = 0; i < images.size(); i++) {
        train_batch(images[index[i]], labels[index[i]], learning_rate);

        if((i+1)%100 == 0) {
            std::cout << "\r Training..... [" << i+1 << "/" << images.size() << "]";
        }

        //At the end of each batch, and at the end
        //i + 1 because if indended batch size is 32, but we start at 0, then when i == 31, that would be the 32nd image
        if((i+1)%batch_size == 0 || (i+1) == images.size()) {
            for(auto& x : layers) {
                x.update_weights(learning_rate, batch_size);
                x.clear_weights();
            }
        }
    }

    //do the same as train_all
    //but call the backwards_batch
    //and when images.size()%batch_size == 0, call update_weights and clear_weights on the layer
    //backwards_batch should be basically the exact same as Network::backward, but calling Layer::backward_batch where weights arent updates. so updates need to happen here
}

void Network::train_batch(const Eigen::VectorXf& input, uint8_t label, float learning_rate) {
    Eigen::VectorXf prediction = forward(input);
    Eigen::VectorXf true_label(10); //size 0-9
    true_label.setZero(); //set all to 0
    true_label[label] = 1; //set label index to 1
    backward_batch(prediction - true_label); 
}

void Network::backward_batch(const Eigen::VectorXf& gradient) {
    Eigen::VectorXf temp = gradient;
    for(int i = static_cast<int>(layers.size())-1; i >= 0; i--) {
        temp = layers[i].backward_batch(temp);
    }
}

/*
 * @brief Prints the Confusion Matrix to the terminal
 * @param conf_matrix 10x10 Confusion Matrix based on test data
 */
void printMatrix(int conf_matrix[10][10]) {
    std::cout << "\nTest Data Set Confusion Matrix:\n\n";
    std::cout << "    |";
    for(int i = 0; i < 10; i++)
        std::cout << std::setw(5) << i << " |";
    std::cout << "\n";

    std::cout << std::string(74, '-') << "\n";

    for(int i = 0; i < 10; i++) {
        std::cout << std::setw(3) << i << " |";
        for(int j = 0; j < 10; j++) {
            std::cout << std::setw(5) << conf_matrix[i][j] << " |";
        }
        std::cout << "\n";
        std::cout << std::string(74, '-') << "\n";
    }
    std::cout << "\n";
}

void saveMatrixCSV(int conf_matrix[10][10], const std::string& path) {
    std::ofstream f(path);
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            f << conf_matrix[i][j];
            if(j < 9) f << ",";
        }
        f << "\n";
    }
}

void Network::test(const std::vector<Eigen::VectorXf>& images, const std::vector<uint8_t>& labels) {
    for(size_t i = 0; i < labels.size(); i++){
        int prediction = predict(images[i]);
        if(prediction == static_cast<int>(labels[i])) correct++;
        confusion_matrix[prediction][labels[i]]++; //labels are 0 through 9, and indexes go 0 through 9, so very nice isnt it
        if((i+1)%1000 == 0) { std::cout << "\rTesting... [" << i+1 << "/" << images.size() << "]"; }
    }
    std::cout << "\rTesting... [" << images.size() << "/" << images.size() << "]\n";

    total_tested += images.size();
}

void Network::save_results() {
    std::cout << std::endl;
    float accuracy = static_cast<float>(correct) / static_cast<float>(total_tested)*100.0f;
    std::cout << "Test accuracy: " << accuracy <<"%\n\n";
    printMatrix(confusion_matrix);
    saveMatrixCSV(confusion_matrix, results_path + "/confusion_matrix.csv");
}