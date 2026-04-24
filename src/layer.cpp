/*

Wright matrix W (outputs x inputs) and bias vector b

forward pass ->
        z = W*x + b;
        a = activation_function(z);

Also store data for backprop
Need to cache:
        z -> pre-activation_function
        a -> post-activation_function

W and b update via gradiant descent (handled in network.cpp)
*/