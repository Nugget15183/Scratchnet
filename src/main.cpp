#include <iostream>
#include <vector>

#include "Network.h"

//c

int main() {
    Network net = Network({2, 4, 1});

    vector<Sample> xor_data = {
        //input output pair 1
        {
            {0, 0},
            {0}
        },
        //input output pair 2
        {
            {0, 1},
            {1}
        },
        //input output pair 3
        {
            {1, 0},
            {1}
        },
        //input output pair 4
        {
            {1, 1},
            {0}
        }
    };

    net.train(xor_data, 5000, 4, 3.0f);

    for (auto& s : xor_data) {
        std::cout << s.x[0] << "," << s.x[1] << " -> " << net.feed_forward(s.x)[0] << "\n";
    }


    return 0;
}