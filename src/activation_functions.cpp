/*

Put ReLU, Sigmoid, Softmax in here
and their derivatives for back prop
Should be simple

*/
#include "../include/activation_functions.h"

namespace activation_function {
    float relu(float x) {
        return(std::max(0.0f, x));
    }

    float relu_back(float x) {
        return x > 0.0f ? 1.0f : 0.0f;
    }

    

    // Equation:
    //  S(y_i) = (e^{y_i}) / sum(e^{y})

    Eigen::VectorXf softmax(const Eigen::VectorXf& input) {
        //Shift all inputs so that 0 is the biggest number (e^0 = 1 | e^{-x} = small number)
        //Stops exploding nums
        Eigen::VectorXf shifted = input.array() - input.maxCoeff();

        //Raises e^x
        Eigen::VectorXf e = shifted.array().exp();

        //Divides each variable by the sum of all
        return e / e.sum();
    }
    //                  First iteration, did this to learn the math 
    //                  before making it with Eigen Vectors

    // std::vector<float> softmax(std::vector<float>& input) {
    //     std::vector<float> probabilities;
    //     int input_size = input.size();
    //     probabilities.resize(input_size);
    //     //fuckin idk
    //     float summation = 0.0f;
    //     float max = 0.0f;
    //     for(float x : input) {
    //         if(x > max) max = x;
    //     }
    //     for(int i = 0; i < input_size; i++) {
    //         float e = expf(input[i] - max);
    //         probabilities[i] = e;
    //         summation += e;
    //     }
    //     for(int i = 0; i < input_size; i++) {
    //         probabilities[i] = probabilities[i]/summation;
    //     }
    // }
    
}

