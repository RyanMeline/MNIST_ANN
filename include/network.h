#pragma once
#include "layer.h"
#include <vector>

class Network {
    public:
        Network() {};
        /*
         * @brief Adds layer to network
         * @param inputs Number of input Neurons
         * @param outputs Number of output Neurons
         * @param act Activation function used in layer
         * @param act_derivative Derivative of Activation function used in back prop
         */
        void add_layer(int inputs, int outputs,
                    std::function<Eigen::VectorXf(const Eigen::VectorXf&)> act,
                    std::function<Eigen::VectorXf(const Eigen::VectorXf&)> act_derivative
                        = [](const Eigen::VectorXf& x) { return Eigen::VectorXf::Ones(x.size()); });

        /*
         * @brief Pushes image forward through each layer
         * @param input Vector representing an image
         * @return Eigen::VectorXf - Eigen Vector of floats representing prediction probabilities
         * @note Calls layer::forward() on each layer
         */
        Eigen::VectorXf forward(const  Eigen::VectorXf& input);

        /*
         * @brief Runs backwards through layers, calling back prop each layer
         * @param gradient Vector of probabilities of predicted - actual number(vector of 0's with a 1 in true nums spot)
         * @param learning_rate Scaler for weight change
         */
        void backward(const Eigen::VectorXf& gradient, float learning_rate);

        /*
         * @brief Pass a single image forward, then run back prop
         * @param input Vector representing an image
         * @param label Single int label for image
         * @param learning_rate Scaler used for weight change in back prop
         * @note Calls forward() and backward() internally
         */
        void train(const Eigen::VectorXf& input, uint8_t label, float learning_rate);

        /*
         * @brief Converts vector of probabilities to a single prediction
         * @param input Vector of probabilities of labels
         * @return int - Predicted Label
         */
        int predict(const Eigen::VectorXf& input);
        
        /*
         * @brief Evaluate data sets in neural network
         * @param images Data set's images
         * @param labels Data set's labels
         * @return float - Prediction accuracy from [0,1]
         * @note Calls predict() internally
         */
        float eval(const std::vector<Eigen::VectorXf>& images, const std::vector<uint8_t>& labels);
        
        /*
         * @brief Loops through data set training neural network
         * @param images Data set's images
         * @param labels Data set's labels
         * @param index vector with index count = # of images. Used to reorder dataset while training to randomize each epoch 
         * @param learning_rate Scaler for neuron weight change
         * @note Calls train() internally
         */
        void train_all(const std::vector<Eigen::VectorXf>& images, const std::vector<uint8_t>& labels, const std::vector<int>& index, float learning_rate);
        
        /*
         * @brief Runs test data through network, creates reliability results
         * @param images Data set's images
         * @param labels Data set's labels
         * @note Calls predict(), printMatrix(), saveMatrixCSV() internally
         */
        void Network::test(const std::vector<Eigen::VectorXf>& images, const std::vector<uint8_t>& labels);

        //Mini batch stuff
        
        /*
         * @brief Loops through data set training neural network in batches, updates weights after batch_size images have been trained
         * @param images Data set's images
         * @param labels Data set's labels
         * @param index vector with index count = # of images. Used to reorder dataset while training to randomize each epoch 
         * @param learning_rate Scaler for neuron weight change
         * @param batch_size Number of images per batch - Default value = 32 
         */
        void Network::train_all_batch(const std::vector<Eigen::VectorXf>& images, const std::vector<uint8_t>& labels, const std::vector<int>& index, float learning_rate, int batch_size = 32);
        
        /*
         * @brief Pass a single image forward, then run back prop
         * @param input Vector representing an image
         * @param label Single int label for image
         * @param learning_rate Scaler used for weight change in back prop
         * @note Calls forward() and backward_batch() internally
         */
        void Network::train_batch(const Eigen::VectorXf& input, uint8_t label, float learning_rate);
        
        /*
         * @brief Runs backwards through layers, calling back prop each layer
         * @param gradient Vector of probabilities of predicted - actual number(vector of 0's with a 1 in true nums spot)
         * @note calls layer::backward_batch() internally
         */
        void Network::backward_batch(const Eigen::VectorXf& gradient);
        //Stores file path for results
        std::string results_path;

        private:
        //Vector pointing to layers
        std::vector<Layer> layers;
};