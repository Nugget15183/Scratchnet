//
// Created by conte on 10/26/2026.
//

#ifndef AI_LEARNING_UTIL_H
#define AI_LEARNING_UTIL_H

#include <cmath>



using namespace std;

namespace util {
    inline float sigmoid(float z) {
        return 1.0f / (1.0f + std::exp(-z));
    }

    inline float sigmoid_derivative(float z) {
        float s = sigmoid(z); return s * (1.0f - s);
    }
}

#endif //AI_LEARNING_UTIL_H