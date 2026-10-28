#include "Network.h"

#include <iostream>
#include <vector>

using namespace util;

void Network::ResetWeightsToRand(const vector<int> &sizes) {
    normal_distribution<float> dist(0.0f, 1.0f);

    //assign random starting values to biases and weights

    biases.clear();
    weights.clear();

    for (size_t l = 0; l + 1 < sizes.size(); l++) {
        biases.emplace_back(sizes[l + 1]);
        weights.emplace_back(sizes[l + 1], vector<float>(sizes[l]));

        for (auto &b : biases[l]) {
            b = dist(rng);
        }

        for (auto &row : weights[l]) {
            for (auto& w : row) w = dist(rng);
        }
    }
}

Network::Network(const vector<int> &sizes) {
    ResetWeightsToRand(sizes);
}

vector<float> Network::feed_forward(vector<float> a) {
    for (size_t l = 0; l < weights.size(); l++) {
        vector<float> next(weights[l].size());

        for (size_t j = 0; j < next.size(); ++j) {
            float z = biases[l][j];

            for (size_t k = 0; k < a.size(); k++) {
                z += weights[l][j][k] * a[k];
            }

            next[j] = sigmoid(z);
        }

        a = std::move(next);
    }

    return a;
}

void Network::train(vector<Sample> data, int epochs, int batch_size, float eta, bool& cancelling) {
    for (int e = 0; e < epochs; e++) {
        if (cancelling) {
            cancelling = false;
            break;
        }

        //shuffle the data to help reduce the chance of overfitting
        shuffle(data.begin(), data.end(), rng);

        //cut the data into batches to speed up testing
        for (size_t i = 0; i < data.size(); i+=batch_size) {
            size_t end = std::min(i + (size_t)batch_size, data.size());
            update_mini_batch(data, i, end, eta);
        }

        //give the output updates on how training is going
        if (e % 500 == 0) {
            cout << "Epoch " << e << " Error: " << eval(data) << endl;
        }
    }
}

float Network::eval(vector<Sample> &test) {
    float err = 0;

    for (auto &s : test) {
        auto out = feed_forward(s.x);

        for (size_t j = 0; j < out.size(); ++j) {
            err += (out[j] - s.y[j]) * (out[j] - s.y[j]);
        }
    }

    //average err
    return err / test.size();
}

void Network::backprop(Sample &s, Biases &nabla_b, Weights &nabla_w) {
    std::vector<std::vector<float>> activations{s.x}, zs;

    for (size_t l = 0; l < weights.size(); ++l) {
        std::vector<float> z(biases[l].size()), a(biases[l].size());

        for (size_t j = 0; j < z.size(); ++j) {
            z[j] = biases[l][j];

            for (size_t k = 0; k < activations.back().size(); ++k) {
                z[j] += weights[l][j][k] * activations.back()[k];
            }

            a[j] = sigmoid(z[j]);
        }

        zs.push_back(z);
        activations.push_back(a);
    }

    // output layer err
    size_t L = weights.size() - 1;
    std::vector<float> delta(zs[L].size());

    for (size_t j = 0; j < delta.size(); ++j) {
        delta[j] = (activations.back()[j] - s.y[j]) * sigmoid_derivative(zs[L][j]);
    }

    //walk back through layers
    for (size_t l = L + 1; l-- > 0;) {
        for (size_t j = 0; j < delta.size(); ++j) {
            nabla_b[l][j] += delta[j];

            for (size_t k = 0; k < activations[l].size(); ++k) {
                nabla_w[l][j][k] += delta[j] * activations[l][k];
            }
        }

        if (l > 0) {
            std::vector<float> prev(zs[l - 1].size(), 0.0f);

            for (size_t k = 0; k < prev.size(); ++k) {
                for (size_t j = 0; j < delta.size(); ++j) {
                    prev[k] += weights[l][j][k] * delta[j];
                }

                prev[k] *= sigmoid_derivative(zs[l - 1][k]);
            }

            delta = std::move(prev);
        }
    }
}

void Network::update_mini_batch(vector<Sample> &data, size_t begin, size_t end, float eta) {
    Biases nb = biases;
    Weights nw = weights;

    for (auto& v : nb) {
        std::fill(v.begin(), v.end(), 0.0f);
    }

    for (auto& m : nw) {
        for (auto& v : m) {
            std::fill(v.begin(), v.end(), 0.0f);
        }
    }

    for (size_t i = begin; i < end; ++i) {
        backprop(data[i], nb, nw);
    }

    float scale = eta / (end - begin);

    for (size_t l = 0; l < weights.size(); ++l) {
        for (size_t j = 0; j < weights[l].size(); ++j) {
            biases[l][j] -= scale * nb[l][j];

            for (size_t k = 0; k < weights[l][j].size(); ++k) {
                weights[l][j][k] -= scale * nw[l][j][k];
            }
        }
    }
}
