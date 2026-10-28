#include <iostream>
#include <vector>

#include "Network.h"
#include "Window.h"


int main() {
    Network net = Network({2, 4, 1});
    Window win = Window(&net);

    win.init();

    while (!glfwWindowShouldClose(win.getWindow())) {
        win.update();
        win.render();
    }

    win.shutdown();

    return 0;
}