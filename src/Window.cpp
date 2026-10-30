#include "Window.h"

#include <imgui_internal.h>
#include <iostream>
#include "util.h"

#include "implot.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "mnist.h"

using namespace std;

//stores information about a function the user can attempt to teach its network
struct LearnFn {
    int choice = 2;
    int samples = 64;

    float xMin = -3.14159f;
    float xMax = 3.14159f;

    float yMin = 0.0f;
    float yMax = 1.0f;

    vector<float> examples, targets;
    vector<float> guesses;

    mutex mtx;

    atomic_bool active = false;
};

struct MNISTState {
    shared_ptr<const vector<Sample>> train, test;
    atomic_bool active = false;
    atomic<float> accuracy = 0.0f;
    string status;

    //confidence viewer
    atomic<int> selected = 0;
    mutex mtx;
    vector<float> outputs;
    int shownIndex = -1;
};

namespace {
    LearnFn lf = LearnFn();

    const char* fnNames[] = {
        "x",
        "x^2",
        "sin(x)",
        "|x|",
        "sin(x) + 0.5 sin(3x)"
    };

    //return the results of a given function at x
    float evalFn(int c, float x) {
        switch(c) {
            case 0: return x;
            case 1: return x*x;
            case 2: return sin(x);
            case 3: return abs(x);
            case 4: return sin(x) + 0.5f * sin(3 * x);
        }
    }

    template <class Net>
    void sampleGuess(Net& net) {
        //allow for guesses while live training on the trainer thread
        vector<float> g(lf.examples.size());

        for (size_t i = 0; i < lf.examples.size(); i++) {
            float xn = (lf.examples[i] - lf.xMin) / (lf.xMax - lf.xMin);
            auto out = net.feed_forward(vector<float>{xn});

            g[i] = out[0] * (lf.yMax - lf.yMin) + lf.yMin;
        }

        lock_guard<mutex> lk(lf.mtx);
        lf.guesses = move(g);
    }

    MNISTState mn;
}

template <class Net>
void updatePreview(Net& net) {
    if (!mn.test || mn.test->empty()) return;

    auto s = net.getSizes();
    if (s.front() != 784 || s.back() != 10) {
        lock_guard<mutex> lk(mn.mtx);
        mn.outputs.clear();
        mn.shownIndex = -1;
        return;
    }

    int idx = std::clamp(mn.selected.load(), 0, (int)mn.test->size() - 1);
    auto out = net.feed_forward((*mn.test)[idx].x);

    lock_guard<mutex> lk(mn.mtx);
    mn.outputs = std::move(out);
    mn.shownIndex = idx;
}


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

    sizes = {(int) data[0].x.size(), 4, (int)data[0].y.size()};

    std::cout << "Loading Complete" << std::endl;
}

void Window::update() {
    if (beginTraining && !training) {
        if (trainThread.joinable()) trainThread.join();

        beginTraining = false;
        cancelling = false;

        {
            lock_guard lk(lossMutex);
            if (!latest_history.empty()) {
                history.push_back(std::move(latest_history));
            }
            latest_history.clear();
        }

        training = true;

        trainThread = std::thread([this, set = trainSet, ep = epochs, bs = batch_size, et = eta, re = reportEvery]() -> void {
            net->train(*set, ep, bs, et, re, cancelling, [this](int epochs, float err) -> void {
                {
                    lock_guard lk(lossMutex);
                   latest_history.push_back({epochs, err});
                }

                if (lf.active) {sampleGuess(*net);} //live guesses

                if (mn.active && mn.test) {
                    mn.accuracy = net->accuracy(*mn.test);
                    updatePreview(*net);
                }
            });

            if (lf.active) {
                sampleGuess(*net);
                lf.active = false;
            }

            if (mn.active) {
                updatePreview(*net);
            }

            training=false;
        });
    }
}

void Window::drawUI() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImVec2 origin = viewport->WorkPos;
    ImVec2 area = viewport->WorkSize;

    float leftw = area.x * 0.35f;
    float halfH = area.y * 0.5f;

    auto prepNet = [&]() {
        if (resetOnTrain || sizes != net->getSizes()) {
            net->ResetWeightsToRand(sizes);
        } else {
            net->ResetAdam();
        }
    };

    ImGui::SetNextWindowPos(origin, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(leftw, area.y), ImGuiCond_FirstUseEver);
    ImGui::Begin("Parameters");

    ImGui::SliderFloat("ETA", &eta, 0.0001f, 5.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    if (net->useAdam && eta >= 1.0f) {
        ImGui::TextDisabled("It is recommended that ETA stays < 1.0 While using Adam");
    }

    ImGui::BeginDisabled(training);

    ImGui::Checkbox("Use Adam", &net->useAdam);

    if (net->useAdam) {
        ImGui::SliderFloat("Beta 1", &net->beta1, 0.5f, 0.999f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Beta 2", &net->beta2, 0.9f, 0.9999f, "%.4f", ImGuiSliderFlags_Logarithmic);
    }

    ImGui::EndDisabled();

    ImGui::InputInt("Epochs", &epochs);
    ImGui::InputInt("Batches", &batch_size);

    if (ImGui::Button("Train")) {
        lf.active = false;
        mn.active = false;

        data=xorData;
        trainSet = make_shared<const vector<Sample>>(data);

        sizes.front() = (int)xorData[0].x.size();
        sizes.back() = (int)xorData[0].y.size();
        prepNet();

        if (training) {
            cancelling = true;
        }
        beginTraining = true;
    }

    ImGui::BeginDisabled(training);
    if (ImGui::Button("Test On Current Training Data")) {
        for (auto& s : data) {
            for (size_t i = 0; i < s.x.size(); i++) {
                std::cout << s.x[i] << (i + 1 < s.x.size() ? ", " : "");
            }
            std::cout << " -> " << net->feed_forward(s.x)[0] << "\n";
        }
    }
    ImGui::EndDisabled();

    ImGui::BeginDisabled(training);
    ImGui::SeparatorText("Architecture");
    ImGui::Text("Input: %d Neurons", sizes.front());

    const char* names[] = { "Sigmoid", "Tanh", "ReLU", "Leaky ReLU" };
    int h = (int)net->hiddenAct;
    if (ImGui::Combo("Hidden activation", &h, names, IM_ARRAYSIZE(names))) {
        net->hiddenAct = (activation)h;
        prepNet();
    }

    int removeIdx=-1;
    for (int i=1; i + 1 < (int)sizes.size(); i++) {
        ImGui::PushID(i);
        ImGui::SetNextItemWidth(100);

        //neuron input text input
        ImGui::InputInt("###neurons", &sizes[i]);
        sizes[i] = std::clamp(sizes[i], 1, 64);

        //hidden neurons
        ImGui::SameLine();
        ImGui::Text("Hidden %d", i);

        //remove neurons button
        ImGui::SameLine();
        if (ImGui::SmallButton("Remove")) removeIdx = i;

        ImGui::PopID();
    }

    if (removeIdx != -1) {
        sizes.erase(sizes.begin() + removeIdx);
    }

    if (sizes.size() < 10 && ImGui::Button("Add Hidden Layer")) {
        sizes.insert(sizes.end() - 1, 4);  //before hidden layer
    }

    ImGui::Text("Output: %d neurons", sizes.back());

    if (sizes != net->getSizes()) {
        ImGui::TextDisabled("Architecture changed: weights will be reset on Train");
    }

    ImGui::EndDisabled();
    ImGui::End();


    ImGui::SetNextWindowPos(ImVec2(origin.x + leftw, origin.y), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(halfH, halfH), ImGuiCond_FirstUseEver);

    ImGui::Begin("Error Graph");
    ImGui::Checkbox("Log Scale", &logscale);

    std::vector<glm::vec2> snap;

    {
        std::lock_guard lk(lossMutex);
        snap = latest_history;
    }

    if (ImPlot::BeginPlot("##loss", ImGui::GetContentRegionAvail())) {
        //label the axes
        ImPlot::SetupAxes("Epoch", "Error", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

        //toggle between log scale and linear
        if (logscale) {
            ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        }

        int i=0;
        for (auto a : history) {
            i++;

            ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, 1.0f);
            ImPlot::PlotLine(("Loss " + to_string(i)).c_str(), &a[0].x, &a[0].y, (int)a.size(), 0, 0, sizeof(glm::vec2));
        }

        if (!snap.empty()) {
            ImPlot::PlotLine("Latest Loss", &snap[0].x, &snap[0].y, (int)snap.size(), 0, 0, sizeof(glm::vec2));
        }
        ImPlot::EndPlot();
    }

    ImGui::End();


    ImGui::SetNextWindowPos(ImVec2(origin.x + leftw, origin.y + halfH), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(halfH, halfH), ImGuiCond_FirstUseEver);

    ImGui::Begin("Learn Function");
    ImGui::BeginDisabled(training);

    ImGui::Combo("Target", &lf.choice, fnNames, IM_ARRAYSIZE(fnNames));

    ImGui::DragFloatRange2("X Range", &lf.xMin, &lf.xMax, 0.05f, -20.0f, 20.0f);
    if (lf.xMax - lf.xMin < 0.1f) lf.xMax = lf.xMin + 0.1f;

    ImGui::SliderInt("Training samples", &lf.samples, 8, 256);

    if (ImGui::Button("Teach Function")) {
        //# of points
        int N = 200;
        mn.active=false;

        lf.examples.clear();
        lf.targets.clear();

        //plot the points for teaching
        for (int i = 0; i < N; i++) {
            float x = lf.xMin + (lf.xMax - lf.xMin) * i / (N - 1);

            lf.examples.push_back(x);
            lf.targets.push_back(evalFn(lf.choice, x));
        }

        //normalize the output to sigmoid's range
        auto[mm, mx] = minmax_element(lf.targets.begin(), lf.targets.end());

        lf.yMin = *mm;
        lf.yMax = *mx;

        if (lf.yMax - lf.yMin < 1e-6f) lf.yMax = lf.yMin + 1.0f;

        float pad = 0.1f * (lf.yMax - lf.yMin);
        lf.yMin -= pad;
        lf.yMax += pad;

        //build samples list
        using Sample = decay_t<decltype(data[0])>;
        data.clear();

        for (int i=0; i < lf.samples; i++) {
            float x = lf.xMin + (lf.xMax - lf.xMin) * i / (lf.samples - 1);

            Sample s;

            s.x = {(x - lf.xMin) / (lf.xMax - lf.xMin)};
            s.y = {(evalFn(lf.choice, x) - lf.yMin) / (lf.yMax - lf.yMin)};

            data.push_back(s);
        }

        //fit layer sizes to be compatible with learning functions
        trainSet = make_shared<const vector<Sample>>(data);

        sizes.front() = 1;
        sizes.back() = 1;
        prepNet();

        reportEvery = std::max(1, epochs / 200);
        lf.active = true;

        //show untrained guesses
        sampleGuess(*net);
        beginTraining = true;
    }
    ImGui::EndDisabled();

    if (training) {
        ImGui::SameLine();
        if (ImGui::Button("Stop")) {
            cancelling=true;
        }

        ImGui::SameLine();
        ImGui::TextDisabled("Training...");
    }

    if (!lf.examples.empty()) {
        vector<float> g;
        {
            std::lock_guard lk(lf.mtx);
            g=lf.guesses;
        }

        if (ImPlot::BeginPlot("##fn", ImVec2(-1, ImGui::GetContentRegionAvail().y))) {
            ImPlot::SetupAxes("x", "f(x)");

            ImPlot::PlotLine("Target", lf.examples.data(), lf.targets.data(), (int)lf.examples.size());

            if (g.size() == lf.examples.size()) {
                ImPlot::PlotLine("Network guess", lf.examples.data(), g.data(), (int)g.size());
            }

            ImPlot::EndPlot();
        } else {
            ImGui::TextDisabled("Pick a function and press the button.");
        }
    }

    ImGui::End();


    ImGui::SetNextWindowSize(ImVec2(560, 520), ImGuiCond_FirstUseEver);
    ImGui::Begin("MNIST");

    static char dir[256] = "data/mnist";
    static int trainCount = 10000;

    ImGui::InputText("Folder", dir, sizeof dir);
    ImGui::SliderInt("Training images", &trainCount, 1000, 60000);

    ImGui::BeginDisabled(training);
    if (ImGui::Button("Load Dataset")) {
        string d = dir;

        auto tr = mnist::load(d + "/train-images.idx3-ubyte", d + "/train-labels.idx1-ubyte", trainCount);
        auto te = mnist::load(d + "/t10k-images.idx3-ubyte", d + "/t10k-labels.idx1-ubyte", 2000);

        if (tr.empty() || te.empty()) {
            mn.status = "Couldn't read the MNIST files in " + d;
        } else {
            mn.status = "Loaded " + to_string(tr.size()) + " train / " + to_string(te.size()) + " test images";

            mn.train = make_shared<const vector<Sample>>(std::move(tr));
            mn.test  = make_shared<const vector<Sample>>(std::move(te));

            mn.selected = 0;
            updatePreview(*net);
        }
    }
    ImGui::EndDisabled();

    ImGui::BeginDisabled(training || !mn.train);
    if (ImGui::Button("Train on MNIST")) {
        lf.active = false;
        mn.active = true;
        mn.accuracy = 0.0f;

        trainSet = mn.train;
        sizes.front() = 784;
        sizes.back() = 10;
        prepNet();

        reportEvery = 1;
        updatePreview(*net);
        beginTraining = true;
    }
    ImGui::EndDisabled();

    if (!mn.status.empty()) {
        ImGui::TextDisabled("%s", mn.status.c_str());
    }

    if (mn.test) {
        ImGui::Text("Test accuracy: %.1f%%", mn.accuracy.load() * 100.0f);
    }

    ImGui::SeparatorText("Network Guess");
    if (!mn.test) {
        ImGui::TextDisabled("Load the dataset to see test images");
    } else {
        const int total = (int)mn.test->size();
        int sel = std::clamp(mn.selected.load(), 0, total - 1);
        bool changed = false;

        //pick which test image to look at
        if (ImGui::Button("<")) { sel = (sel + total - 1) % total; changed = true; }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(200);
        if (ImGui::SliderInt("##img", &sel, 0, total - 1, "Image %d")) changed = true;
        ImGui::SameLine();
        if (ImGui::Button(">")) { sel = (sel + 1) % total; changed = true; }
        ImGui::SameLine();
        if (ImGui::Button("Random")) {
            static mt19937 pick{random_device{}()};
            sel = uniform_int_distribution<int>(0, total - 1)(pick);
            changed = true;
        }

        static bool normalize = false;
        ImGui::Checkbox("Normalize to 100%", &normalize);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Each output neuron is an independent sigmoid, so the raw values\ndon't have to add up to 100%%.");
        }

        if (changed) {
            mn.selected = sel;
            if (!training) updatePreview(*net); //while training it refreshes after the next epoch
        }

        //grab a snapshot of the outputs
        vector<float> out;
        int shown;
        {
            lock_guard lk(mn.mtx);
            out = mn.outputs;
            shown = mn.shownIndex;
        }

        const Sample& smp = (*mn.test)[sel];
        auto argmax = [](const vector<float>& v) { return (int)(max_element(v.begin(), v.end()) - v.begin()); };
        const int truth = argmax(smp.y);

        //only show bars that belong to the image on screen
        const bool ready = out.size() == 10 && shown == sel;

        float vals[10] = {};
        int guess = 0;

        if (ready) {
            float sum = 0.0f;
            for (float v : out) sum += v;

            for (int i = 0; i < 10; i++) {
                vals[i] = (normalize && sum > 1e-6f) ? out[i] / sum : out[i];
            }
            guess = argmax(out);

            ImVec4 col = (guess == truth) ? ImVec4(0.3f, 0.9f, 0.4f, 1.0f) : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
            ImGui::TextColored(col, "Guess: %d (%.1f%%)    Actual: %d", guess, vals[guess] * 100.0f, truth);
        } else if (out.size() == 10) {
            ImGui::TextDisabled("Actual: %d    (updates after the next epoch)", truth);
        } else {
            ImGui::TextDisabled("Actual: %d    (train on MNIST to see the network's guess)", truth);
        }

        //draw the 28x28 image
        const float cell = 8.0f;
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList* dl = ImGui::GetWindowDrawList();

        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                int v = (int)(smp.x[r * 28 + c] * 255.0f);

                dl->AddRectFilled(ImVec2(p.x + c * cell, p.y + r * cell),
                                  ImVec2(p.x + (c + 1) * cell, p.y + (r + 1) * cell),
                                  IM_COL32(v, v, v, 255));
            }
        }
        ImGui::Dummy(ImVec2(28 * cell, 28 * cell));

        //confidence bars
        ImGui::SameLine();

        if (ImPlot::BeginPlot("##conf", ImVec2(-1, 28 * cell), ImPlotFlags_NoInputs | ImPlotFlags_NoTitle)) {
            ImPlot::SetupAxes("Digit", "Confidence");
            ImPlot::SetupAxisLimits(ImAxis_X1, -0.7, 9.7, ImPlotCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.15, ImPlotCond_Always);
            ImPlot::SetupAxisTicks(ImAxis_X1, 0, 9, 10);

            if (ready) {
                //split into series so the guess and the right answer get their own colors
                float others[10], guessBar[10] = {}, missed[10] = {};
                for (int i = 0; i < 10; i++) others[i] = vals[i];

                guessBar[guess] = vals[guess];
                others[guess] = 0.0f;

                if (guess != truth) {
                    missed[truth] = vals[truth];
                    others[truth] = 0.0f;
                }

                ImPlot::SetNextFillStyle(ImVec4(0.35f, 0.50f, 0.85f, 1.0f));
                ImPlot::PlotBars("Other digits", others, 10);

                ImPlot::SetNextFillStyle(guess == truth ? ImVec4(0.3f, 0.8f, 0.4f, 1.0f)
                                                        : ImVec4(0.9f, 0.35f, 0.35f, 1.0f));
                ImPlot::PlotBars("Network's guess", guessBar, 10);

                if (guess != truth) {
                    ImPlot::SetNextFillStyle(ImVec4(0.95f, 0.8f, 0.25f, 1.0f));
                    ImPlot::PlotBars("Correct answer", missed, 10);
                }

                //percentage labels above each bar
                for (int i = 0; i < 10; i++) {
                    if (vals[i] < 0.01f) continue;

                    char buf[16];
                    snprintf(buf, sizeof buf, "%.0f%%", vals[i] * 100.0f);
                    ImPlot::PlotText(buf, i, vals[i], ImVec2(0, -10));
                }
            }

            ImPlot::EndPlot();
        }
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