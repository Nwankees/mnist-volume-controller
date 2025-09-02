#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main() {
    // ✅ Use all cores available
    torch::set_num_threads(std::thread::hardware_concurrency());

    try {
        // Load the TorchScript model
        torch::jit::script::Module module = torch::jit::load("mnist_cnn.pt");
        module.eval();

        // Load an example image (grayscale, 28x28)
        cv::Mat img = cv::imread("digit.png", cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Could not read input image!" << std::endl;
            return -1;
        }
        cv::resize(img, img, cv::Size(28, 28));
        img.convertTo(img, CV_32F, 1.0 / 255);

        // Convert to tensor (NCHW format)
        auto input = torch::from_blob(img.data, {1, 1, 28, 28}, torch::kFloat32).clone();

        // ✅ Warm-up (important to get fair timing)
        for (int i = 0; i < 10; i++) {
            auto out = module.forward({input}).toTensor();
        }

        // Benchmark loop
        int runs = 1000;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; i++) {
            auto out = module.forward({input}).toTensor();
        }
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate average latency
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double avg_ms = (double)duration / runs / 1000.0;
        std::cout << "Average inference time: " << avg_ms << " ms" << std::endl;

        // Do a final prediction
        auto out = module.forward({input}).toTensor();
        auto pred = out.argmax(1).item<int>();
        std::cout << "Predicted digit: " << pred << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model or running inference: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
