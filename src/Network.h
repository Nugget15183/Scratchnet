#ifndef AI_LEARNING_NETWORK_H
#define AI_LEARNING_NETWORK_H

#include <vector>

#include <random>

#include <functional>
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

    vector<int> getSizes() {return sizes;}
    vector<float> feed_forward(vector<float> a);

    //trains the weights and biases in the neruons
    void train(vector<Sample> data, int epochs, int batch_size, float eta, int reportEvery, atomic_bool& cancel, std::function<void(int, float)> err_callback=nullptr);

    activationFn activationFor(size_t layer) const {
        return getActivation(layer + 1 == weights.size() ? outputAct : hiddenAct);
    }

    void ResetAdam();

    //activation functions used for hidden and output weights
    activation hiddenAct = activation::Sigmoid;
    activation outputAct = activation::Sigmoid;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    bool useAdam = true;
private:
    //define weights and biases
    using Biases = vector<vector<float>>;
    using Weights = vector<vector<vector<float>>>;

    //layer sizes
    vector<int> sizes;

    //biases + weights
    Biases biases;
    Weights weights;

    //adam's biases and weights
    Biases mb, vb;
    Weights mw, vw;

    //time counter
    long long t = 0;

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