# Scratchnet
A Customizable nerual network, built from scratch in C++.

<img width="1920" height="1080" alt="Screenshot (901)" src="https://github.com/user-attachments/assets/01fda28a-47f8-4083-9ab1-d2a557f74fa8" />

## Features

**Network**

- Fully Connected feed-forward network with any number of hidden layers, each hidden layer with a customizable amount of neurons from 1 to 64.
- 4 hidden and output activations to test from
  - Sigmoid
  - Tanh
  - ReLU
  - Leaky ReLU
- Weight initialization matches the selected activation function
  - He (ReLU and Leaky ReLU)
  - Xavier (used foreverything except the ReLUs)
- Two different optimizers to chose from
  - ADAM (Comes with adjustable B1 and B2)
  - SGD
- Compatible with the MNIST dataset for training and testing

**GUI**
- Training is run on a background thread, so the ui stays responsive
- Live error graph with a log scale toggle, each run is stored so you can compare them
- Testing your network on various math functions to see how it fairs
  - f(x) = x
  - f(x) = |x|
  - f(x) = x^2
  - f(x) = sin x
  - f(x) = sin x + 0.5 * sin x
- cancel training at any time
- MNIST image previewer aswell as bar graphs displaying the networks confidence guesses

## Things to Try
1. Train XOR at ETA 0.5, then again with ADAM at ETA 0.01
2. Teach sin(x) + 0.5 * sin(x) and add layers and change neuron counts as it improves
3. Swap out Sigmoid for ReLU and watch how convergence changes
4. Try 7 epochs of MNIST with, and without ADAM

## Downloading
Go to the downloads page and download the latest release, unzip the zipped download file and run the .exe

## Building it yourself

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
`util.h` Stores activation function formulas \
`mnist. h/cpp` Handles the loading of the MNIST dataset

## How it works
### Training
Training uses mini batch gradient descent (in laymans terms, it splits the data into parts (batches), shuffles them around, and teaches the network on one batch at a time). Gradients are summed over each batch by backpropagation, then averaged and applied.

There are two methods in which we apply those averages, one being SGD which uses the formula `w -= eta * g`. The other being ADAM, which keeps track of a gradient `m` and its square `v`, corrects bias then steps by the formula `eta * m̂ / (√v̂ + ε)` 

# Some results from training it on the MNIST dataset
<img width="1324" height="670" alt="Screenshot 2026-10-30 160705" src="https://github.com/user-attachments/assets/ff8f005d-cabf-413c-85dd-40ef291b3716" />
<img width="1328" height="633" alt="Screenshot 2026-10-30 160721" src="https://github.com/user-attachments/assets/9006e0f5-9193-4004-b1a1-7a14340e9df4" />
<img width="1323" height="625" alt="Screenshot 2026-10-30 160711" src="https://github.com/user-attachments/assets/ae12eac8-ebc5-4cf3-8441-f255d8429670" />
