#ifndef AI_LEARNING_NETWORK_H
#define AI_LEARNING_NETWORK_H

#include <vector>

#include <random>

#include "util.h"

//c

using namespace std;
using namespace util;

struct Sample {
    vector<float> x;
    vector<float> y;
};

class Network {
public:
    Network(const vector<int>& sizes);
    void ResetWeightsToRand(const vector<int> &sizes);

    vector<float> feed_forward(vector<float> a);

    //trains the weights and biases in the neruons
    void train(vector<Sample> data, int epochs, int batch_size, float eta, bool& cancel);
private:
    //define weights and biases
    using Biases = vector<vector<float>>;
    using Weights = vector<vector<vector<float>>>;

    //layer sizes
    vector<int> sizes;

    //biases + weights
    Biases biases;
    Weights weights;

    //initalize rng
    std::mt19937 rng{std::random_device{}()};

    //updating batches within the training loop
    void update_mini_batch(vector<Sample>& data, size_t begin, size_t end, float eta);

    //backward gradient for updating hidden layers
    void backprop(Sample& s, Biases& nabla_b, Weights& nabla_w);

    //run the network on a sample
    float eval(vector<Sample>& test);
};


#endif