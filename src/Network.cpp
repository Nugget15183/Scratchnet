#include "Network.h"

#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace util;

void Network::ResetAdam() {
    //reset adam's biases and weights to the default
    mb = biases;
    vb = biases;
    mw = weights;
    vw = weights;

    for (auto& v : mb) {
        fill(v.begin(), v.end(), 0.0f);
    }

    for (auto& v : vb) {
        fill(v.begin(), v.end(), 0.0f);
    }

    for (auto& m : mw) {
        for (auto& v : m) {
            fill(v.begin(), v.end(), 0.0f);
        }
    }

    for (auto& m : vw) {
        for (auto& v : m) {
            fill(v.begin(), v.end(), 0.0f);
        }
    }

    //reset the time to 0
    t = 0;
}


void Network::ResetWeightsToRand(const vector<int> &newSizes) {
    sizes = newSizes;

    //assign random starting values to biases and weights

    biases.clear();
    weights.clear();

    for (size_t l = 0; l + 1 < sizes.size(); l++) {
        auto activationf = (l + 2 == sizes.size()) ? outputAct : hiddenAct;
        bool relu = (activationf == activation::ReLU || activationf == activation::LeakyReLU);

        float distrib = relu ? sqrt(2.0f / sizes[l]) : 1.0f / sqrt((float)sizes[l]);

        normal_distribution<float> dist(0.0f, distrib);

        biases.emplace_back(sizes[l + 1]);
        weights.emplace_back(sizes[l + 1], vector<float>(sizes[l]));

        for (auto &b : biases[l]) {
            b = dist(rng);
        }

        for (auto &row : weights[l]) {
            for (auto& w : row) w = dist(rng);
        }
    }

    ResetAdam();
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

            next[j] = activationFor(l).f(z);
        }

        a = std::move(next);
    }

    return a;
}

void Network::train(vector<Sample> data, int epochs, int batch_size, float eta, int reportEvery, atomic_bool& cancelling, std::function<void(int, float)> err_callback) {
    //shuffle the idx list instead of the data itself
    vector<size_t> order(data.size());
    iota(order.begin(), order.end(), 0);

    for (int e = 0; e < epochs; e++) {
        if (cancelling) break;

        //shuffle the order to help reduce the chance of overfitting
        shuffle(order.begin(), order.end(), rng);

        //cut the data into batches to speed up testing
        for (size_t i = 0; i < data.size(); i+=batch_size) {
            if (cancelling) break;

            size_t end = std::min(i + (size_t)batch_size, order.size());
            update_mini_batch(data, order, i, end, eta);
        }

        //give the output updates on how training is going
        if (e % reportEvery == 0) {
            float err = eval(data);

            if (err_callback != nullptr) {
                err_callback(e, err);
            }
        }
    }
}

float Network::eval(vector<Sample> &test) {
    size_t n = min<size_t>(test.size(), 2000);
    float err = 0;

    for (size_t i = 0; i < n; i++) {
        auto out = feed_forward(test[i].x);

        for (size_t j = 0; j < out.size(); ++j) {
            err += (out[j] - test[i].y[j]) * (out[j] - test[i].y[j]);
        }
    }

    //average err
    return err / n;
}

float Network::accuracy(vector<Sample> test) {
    if (test.empty()) return 0;

    auto argmax = [](const vector<float>& v) {
        return max_element(v.begin(), v.end()) - v.begin();
    };

    size_t right = 0;

    for (auto& s : test) {
        if (argmax(feed_forward(s.x)) == argmax(s.y)) {
            right++;
        }
    }

    return (float)right / test.size();
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

            a[j] = activationFor(l).f(z[j]);
        }

        zs.push_back(z);
        activations.push_back(a);
    }

    // output layer err
    size_t L = weights.size() - 1;
    std::vector<float> delta(zs[L].size());

    for (size_t j = 0; j < delta.size(); ++j) {
        delta[j] = (activations.back()[j] - s.y[j]) * activationFor(L).df(zs[L][j]);
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

                prev[k] *= activationFor(l - 1).df(zs[l - 1][k]);
            }

            delta = std::move(prev);
        }
    }
}

void Network::update_mini_batch(vector<Sample> &data, vector<size_t>& order, size_t begin, size_t end, float eta) {
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
        backprop(data[order[i]], nb, nw);
    }

    //update the weights
    float invN = 1.0f / (end - begin);

    if (!useAdam) {
        //non-adam update loop
        float scale = eta * invN;

        for (size_t l = 0; l < weights.size(); ++l) {
            for (size_t j = 0; j < weights[l].size(); ++j) {
                biases[l][j] -= scale * nb[l][j];

                for (size_t k = 0; k < weights[l][j].size(); ++k) {
                    weights[l][j][k] -= scale * nw[l][j][k];
                }
            }
        }
        return;
    }

    //adam optimizer, much faster then sgd
    t++;
    float bc1 = 1.0f - pow(beta1, t);
    float bc2 = 1.0f - pow(beta2, t);

    for (size_t l = 0; l < weights.size(); ++l) {
        for (size_t j = 0; j < weights[l].size(); ++j) {
            //update biases
            float g = nb[l][j] * invN;

            mb[l][j] = beta1 * mb[l][j] + (1.0f - beta1) * g;
            vb[l][j] = beta2 * vb[l][j] + (1.0f - beta2) * g * g;

            biases[l][j] -= eta * (mb[l][j] / bc1) / (sqrt(vb[l][j] / bc2) + epsilon);

            for (size_t k = 0; k < weights[l][j].size(); ++k) {
                //update weights
                float gw = nw[l][j][k] * invN;

                mw[l][j][k] = beta1 * mw[l][j][k] + (1.0f - beta1) * gw;
                vw[l][j][k] = beta2 * vw[l][j][k] + (1.0f - beta2) * gw * gw;

                weights[l][j][k] -= eta * (mw[l][j][k] / bc1) / (sqrt(vw[l][j][k] / bc2) + epsilon);
            }
        }
    }
}
