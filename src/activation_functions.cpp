#include "../include/activation_functions.h"

namespace activation_function {

    Eigen::VectorXf relu(const Eigen::VectorXf& input) {
        return input.array().max(0.0f);
    }

    Eigen::VectorXf relu_back(const Eigen::VectorXf& input) {
        //unaryExpr applies the passed in function or whatever it is to each element in the array
        //ReLU derivative is return 1 if x > 0, return 0 if x < 0, if x = 0, its undefined, but do either (I'm returning 0)
        return input.array().unaryExpr([](float x) { return x > 0.0f ? 1.0f : 0.0f; });
    }

    // Equation:
    //  S(y_i) = (e^{y_i}) / sum(e^{y})

    Eigen::VectorXf softmax(const Eigen::VectorXf& input) {
        //Shift all inputs so that 0 is the biggest number (e^0 = 1 | e^{-x} = small number)  
        Eigen::VectorXf shifted = input.array() - input.maxCoeff(); //Stops exploding nums
        Eigen::VectorXf e = shifted.array().exp(); //Raises e^x  
        return e / e.sum(); //Divides each variable by the sum of all
    }
    //                  SOFTMAX WITHOUT EIGEN
    // std::vector<float> softmax(std::vector<float>& input) {
    //     std::vector<float> probabilities;
    //     int input_size = input.size();
    //     probabilities.resize(input_size);
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
    //     return probabilities;
    // }
}

