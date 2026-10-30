#ifndef AI_LEARNING_WINDOW_H
#define AI_LEARNING_WINDOW_H

#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "Network.h"

class Window {
public:
    Window(Network* net) {this->net = net;}

    void init();
    void render();
    void update();
    void shutdown();

    GLFWwindow* getWindow() { return window; }

    void output(const std::string& msg) { log.push_back(msg); scrollToBottom = true; }
private:
    void drawUI();

    GLFWwindow* window = nullptr;
    glm::vec2 winSize = {1000, 600};

    std::vector<std::string> log;
    bool scrollToBottom = false;

    Network* net;

    atomic_bool cancelling = false;
    atomic_bool training = false;
    atomic_bool beginTraining = false;

    thread trainThread;
    mutex lossMutex;

    //params
    float learning_rate = 0.01f;
    float eta = 3.0f;
    int batch_size = 4;
    int epochs = 5000;
    vector<int> sizes = {2, 4, 1};

    //x is epoch y is loss
    //latest training run
    vector<glm::vec2> latest_history;

    //stores all the training runs
    vector<vector<glm::vec2>> history;

    bool logscale = false;
    int reportEvery = 500;

    //data
    vector<Sample> xorData = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}}
    };

    vector<Sample> data = xorData;
};


#endif //AI_LEARNING_WINDOW_H