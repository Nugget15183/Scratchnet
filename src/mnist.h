#pragma once

#ifndef AI_LEARNING_MNIST_H
#define AI_LEARNING_MNIST_H

#include <vector>

#include "Network.h"

namespace mnist {
    std::vector<Sample> load(const std::string& imagesPath, const std::string& labelsPath, size_t limit=0);
};


#endif //AI_LEARNING_MNIST_H