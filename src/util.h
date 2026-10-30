#ifndef AI_LEARNING_UTIL_H
#define AI_LEARNING_UTIL_H

#include <cmath>

using namespace std;

enum class activation { Sigmoid, Tanh, ReLU, LeakyReLU };

struct activationFn {
    float (*f)(float); //f(x)
    float (*df)(float); //derivitive of f(x)
    const char* name;
};

inline float sigmoidF(float z) {
    return 1.0f / (1.0f + expf(-z));
}

inline float sigmoidD(float z) {
    return sigmoidF(z) * (1.0f-sigmoidF(z));
}

inline float tanhF(float z) {
    return tanhf(z);
}

inline float tanhD(float z) {
    return 1.0f - pow(tanf(z), 2.0f);
}

inline float reluF(float z) {
    return z > 0.0f ? z : 0.0f;
}

inline float reluD(float z) {
    return z > 0.0f ? 1.0f : 0.0f;
}

inline float leakyF(float z) {
    return z > 0.0f ? z : 0.01f * z;
}

inline float leakyD(float z) {
    return z > 0.0f ? z : 0.01f;
}

inline activationFn getActivation(activation a) {
    activationFn table[] {
        {sigmoidF, sigmoidD, "Sigmoid"},
        {tanhF, tanhD, "Tanh"},
        {reluF, reluD, "ReLU"},
        {leakyF, leakyD, "Leaky ReLU"},
    };
    return table[(int)a];
}

namespace util {
    inline float sigmoid(float z) {
        return 1.0f / (1.0f + std::exp(-z));
    }

    inline float sigmoid_derivative(float z) {
        float s = sigmoid(z);
        return s * (1.0f - s);
    }
}



#endif //AI_LEARNING_UTIL_H