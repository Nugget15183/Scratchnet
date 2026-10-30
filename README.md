# Scratchnet
A Customizable nerual network, built from scratch in C++.

## Features

**Network**

- Fully Connected feed-forward network with any number of hidden layers, each with a customizable amount of neurons from 1 to 64.
- Many hidden and output activations to test from
  - Sigmoid
  - Tanh
  - ReLU
  - Leaky ReLU
- Weight initialization matches the selected activation function
  - He (ReLU and Leaky ReLU)
  - Xavier (everything except ReLUs)
- Mini-batch training that gets shuffled each epoch
- Two different optimizers to chose from
  - ADAM (Comes with adjustable B1 and B2)
  - SGD

**GUI**
- Training is run on a background thread, so the ui stays responsive
- Live error graph with a log scale toggle, each run is stored so you can compare them
- Testing your network on various math functions to see how it fairs
- - f(x) = x
- - f(x) = |x|
- - f(x) = x^2
- - f(x) = sin x
- - f(x) = sin x + 0.5 * sin x
- bitwise operation training
- cancel training at any time

## Things to Try
1. Train XOR at ETA 0.5, then again with ADAM at ETA 0.01
2. Teach sin(x) + 0.5 * sin(x) and add layers and change neuron counts as it improves
3. Swap out Sigmoid for ReLU and watch how convergence changes

## Building

**Requirements:** a C++17 compiler, CMake, and any OpenGL 3.3 capable GPU.

Dependencies: [GLFW](https://www.glfw.org/), [GLAD](https://glad.dav1d.de/), [GLM](https://github.com/g-truc/glm), [Dear ImGui](https://github.com/ocornut/imgui), [ImPlot](https://github.com/epezent/implot)

````bash
git clone https://github.com/Nugget15183/Scratchnet
cd Scratchnet
cmake -B build
cmake --build build
./build/ai_learning.exe
````

## Project Structure
`Network. h/cpp` The network itself, handles backprop, the forward pass, and training as a whole \
`Window. h/cpp` Handles the GUI, and allows the user to interact directly with their network \
`util.h` Stores activation function formulas

## How it works
### Training
Training uses mini batch gradient descent (in laymans terms, it splits the data into parts (batches), shuffles them around, and teaches the network on one batch at a time). Gradients are summed over each batch by backpropagation, then averaged and applied.

There are two methods in which we apply those averages, one being SGD which uses the formula `w -= eta * g`. The other being ADAM, which keeps track of a gradient `m` and its square `v`, corrects bias then steps by the formula `eta * m̂ / (√v̂ + ε)` 