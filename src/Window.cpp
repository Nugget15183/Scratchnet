#include "Window.h"

#include <imgui_internal.h>
#include <iostream>
#include "util.h"

#include "implot.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

using namespace std;

void setupStyle() {
    ImGuiStyle& style = ImGui::GetStyle();

    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.PopupRounding = 4.0f;
    style.GrabRounding = 4.0f;

    style.WindowPadding = ImVec2(12.0f, 12.0f);
    style.FramePadding = ImVec2(6.0f, 4.0f);
    style.ItemSpacing = ImVec2(8.0f, 6.0f);

    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.12f, 1.00f); // Deep dark background
    style.Colors[ImGuiCol_Header] = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);   // Subtly lighter panels
    style.Colors[ImGuiCol_Button] = ImVec4(0.25f, 0.38f, 0.66f, 1.00f);   // Pop accent (e.g., slate blue)
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.30f, 0.45f, 0.78f, 1.00f);
}

void Window::init() {
    std::cout << "Initializing..." << std::endl;

    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(winSize.x, winSize.y, "Scratchnet", NULL, NULL);
    if (!window) {
        std::cout << "Failed to create GLFW window" << std::endl;
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    std::cout << "Loading Glad..." << std::endl;

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = "imgui.ini"; //saves window layout

    ImPlot::CreateContext();

    ImGui::StyleColorsClassic();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    setupStyle();

    std::cout << "Loading Complete" << std::endl;
}

void Window::update() {
    if (beginTraining && !training && !cancelling) {
        if (trainThread.joinable()) trainThread.join();

        beginTraining = false;
        cancelling = false;

        if (resetWeights) {
            net->ResetWeightsToRand(sizes);
        }

        {
            lock_guard lk(lossMutex);
            history.clear();
        }

        training = true;

        trainThread = std::thread([this, d = data, ep = epochs, bs = batch_size, et = eta, re = reportEvery]() -> void {
            net->train(d, ep, bs, et, re, cancelling, [this](int epochs, float err) -> void {
                lock_guard lk(lossMutex);
                history.push_back({epochs, err});
            });
            training=false;
        });

        training = false;
    }
}

void Window::drawUI() {
    ImGui::Begin("Parameters");
    ImGui::SliderFloat("Learning Rate", &learning_rate, 0.000001f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("ETA", &eta, 0.1f, 10.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::InputInt("Epochs", &epochs);
    ImGui::InputInt("Batches", &batch_size);
    ImGui::Checkbox("Reset Weights On Train", &resetWeights);

    if (ImGui::Button("Train")) {
        if (training) {
            cancelling = true;
        }
        beginTraining = true;
    }

    if (ImGui::Button("Test On Current Training Data")) {
        for (auto& s : data) {
            std::cout << s.x[0] << "," << s.x[1] << " -> " << net->feed_forward(s.x)[0] << "\n";
        }
    }

    ImGui::End();

    ImGui::Begin("Error Graph");
    ImGui::Checkbox("Log Scale", &logscale);

    std::vector<glm::vec2> snap;

    {
        std::lock_guard lk(lossMutex);
        snap = history;
    }

    if (ImPlot::BeginPlot("##loss", ImGui::GetContentRegionAvail())) {
        //label the axes
        ImPlot::SetupAxes("Epoch", "Loss", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

        //toggle between log scale and linear
        if (logscale) {
            ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        }

        if (!snap.empty()) {
            ImPlot::PlotLine("Loss", &snap[0].x, &snap[0].y, (int)snap.size(), 0, 0, sizeof(glm::vec2));
        }
        ImPlot::EndPlot();
    }

    ImGui::End();
}


void Window::render() {
    glfwPollEvents();

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);

    glViewport(0, 0, w, h);
    glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    drawUI();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
}

void Window::shutdown() {
    cancelling = true;
    if (trainThread.joinable()) trainThread.join();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    window = nullptr;
}