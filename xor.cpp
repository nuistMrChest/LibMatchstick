#include <cmath>
#include <iomanip>
#include <iostream>

#include "src/matrix.h"
#include "src/activation.h"
#include "src/layer.h"

using LibMatchstick::Matrix;
using LibMatchstick::MLPLayer;

static bool finite_matrix(const Matrix& m) {
    for (size_t i = 0; i < m.getHeight(); ++i) {
        for (size_t j = 0; j < m.getWidth(); ++j) {
            if (!std::isfinite(m.get(i, j))) {
                return false;
            }
        }
    }
    return true;
}

static float mse_one(float pred, float target) {
    const float e = pred - target;
    return 0.5f * e * e;
}

static int cls(float x) {
    return x >= 0.5f ? 1 : 0;
}

static float eval_loss(MLPLayer& hidden, MLPLayer& output, Matrix xs[4], Matrix ys[4]) {
    float loss = 0.0f;
    for (int i = 0; i < 4; ++i) {
        Matrix h = hidden.forward(xs[i]);
        Matrix p = output.forward(h);
        loss += mse_one(p.get(0, 0), ys[i].get(0, 0));
    }
    return loss;
}

static int eval_accuracy(MLPLayer& hidden, MLPLayer& output, Matrix xs[4], Matrix ys[4], bool print) {
    int correct = 0;

    for (int i = 0; i < 4; ++i) {
        Matrix h = hidden.forward(xs[i]);
        Matrix p = output.forward(h);

        const float raw = p.get(0, 0);
        const int got = cls(raw);
        const int expected = static_cast<int>(ys[i].get(0, 0));

        if (got == expected) {
            ++correct;
        }

        if (print) {
            std::cout << xs[i].get(0, 0)
                      << " xor "
                      << xs[i].get(1, 0)
                      << " => raw = "
                      << std::fixed << std::setprecision(6) << raw
                      << ", class = "
                      << got
                      << ", expected = "
                      << expected
                      << '\n';
        }
    }

    return correct;
}

int main() {
    using namespace LibMatchstick::Activation;

    Matrix xs[4] = {
        Matrix{{0.0f}, {0.0f}},
        Matrix{{0.0f}, {1.0f}},
        Matrix{{1.0f}, {0.0f}},
        Matrix{{1.0f}, {1.0f}},
    };

    Matrix ys[4] = {
        Matrix{{0.0f}},
        Matrix{{1.0f}},
        Matrix{{1.0f}},
        Matrix{{0.0f}},
    };

    // 2 -> 2 -> 1
    MLPLayer hidden(2, 2);
    MLPLayer output(2, 1);

    hidden.setActivation(sigmoid, sigmoid_d);
    output.setActivation(sigmoid, sigmoid_d);

    // 不用太大的初始值，避免 sigmoid 早早饱和。
    hidden.loadWeight(Matrix{
        { 0.20f, -0.30f },
        { 0.40f,  0.10f }
    });
    hidden.loadBias(Matrix{
        { 0.00f },
        { 0.00f }
    });

    output.loadWeight(Matrix{
        { 0.30f, -0.20f }
    });
    output.loadBias(Matrix{
        { 0.00f }
    });

    const float step = 0.1f;
    const int epochs = 100000;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float total_loss = 0.0f;
        bool ok = true;

        for (int i = 0; i < 4; ++i) {
            Matrix h = hidden.forward(xs[i]);
            Matrix pred = output.forward(h);

            if (!finite_matrix(h) || !finite_matrix(pred)) {
                std::cout << "[ABORT] non-finite value in forward at epoch "
                          << epoch << ", sample " << i << '\n';
                return 2;
            }

            const float y = pred.get(0, 0);
            const float t = ys[i].get(0, 0);

            total_loss += mse_one(y, t);

            // MSE:
            // L = 1/2 * (y - t)^2
            // dL/da = y - t
            Matrix dl_da{{y - t}};

            Matrix dl_dh = output.backward(dl_da, step);
            if (!finite_matrix(dl_dh)) {
                std::cout << "[ABORT] non-finite gradient from output layer at epoch "
                          << epoch << ", sample " << i << '\n';
                return 3;
            }

            Matrix dl_dx = hidden.backward(dl_dh, step);
            if (!finite_matrix(dl_dx)) {
                std::cout << "[ABORT] non-finite gradient from hidden layer at epoch "
                          << epoch << ", sample " << i << '\n';
                return 4;
            }
        }

        if (!std::isfinite(total_loss)) {
            std::cout << "[ABORT] loss became non-finite at epoch "
                      << epoch << '\n';
            return 5;
        }

        if (epoch == 1 || epoch % 5000 == 0) {
            const int acc = eval_accuracy(hidden, output, xs, ys, false);
            std::cout << "epoch " << std::setw(6) << epoch
                      << " | loss = " << std::fixed << std::setprecision(8)
                      << total_loss
                      << " | acc = " << acc << "/4"
                      << '\n';
        }

        // 提前停止。
        if (epoch % 1000 == 0 && eval_accuracy(hidden, output, xs, ys, false) == 4) {
            std::cout << "early stop at epoch " << epoch << '\n';
            break;
        }
    }

    std::cout << "\n===== XOR result =====\n";
    const int correct = eval_accuracy(hidden, output, xs, ys, true);
    std::cout << "\naccuracy = " << correct << " / 4\n";

    return correct == 4 ? 0 : 1;
}

