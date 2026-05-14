#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include "src/matrix.h"
#include "src/activation.h"
#include "src/layer.h"

using LibMatchstick::Matrix;
using LibMatchstick::MLPLayer;

static bool near(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

static void print_check(const std::string& name, float got, float expected, float eps = 1e-4f) {
    std::cout << std::left << std::setw(38) << name
              << " got = " << std::setw(12) << std::fixed << std::setprecision(6) << got
              << " expected = " << std::setw(12) << expected
              << (near(got, expected, eps) ? " [PASS]" : " [FAIL]")
              << '\n';
}

static void print_matrix(const std::string& name, const Matrix& m) {
    std::cout << name << " shape = "
              << m.getHeight() << "x" << m.getWidth() << '\n';

    for (size_t i = 0; i < m.getHeight(); ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < m.getWidth(); ++j) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(6) << m.get(i, j) << ' ';
        }
        std::cout << '\n';
    }
}

int main() {
    using namespace LibMatchstick::Activation;

    std::cout << "===== Activation derivative probe =====\n";

    Matrix z0{{0.0f}};
    Matrix z2{{2.0f}};

    Matrix sd0 = sigmoid_d(z0);
    Matrix sd2 = sigmoid_d(z2);
    Matrix td0 = tanh_d(z0);
    Matrix td2 = tanh_d(z2);

    // If sigmoid_d takes pre-activation z:
    // sigmoid'(0) = 0.25
    // sigmoid'(2) = sigmoid(2) * (1 - sigmoid(2)) ~= 0.104993
    print_check("sigmoid_d(0)", sd0.get(0, 0), 0.25f);
    print_check("sigmoid_d(2)", sd2.get(0, 0), 0.10499358f);

    // If tanh_d takes pre-activation z:
    // tanh'(0) = 1
    // tanh'(2) = 1 - tanh(2)^2 ~= 0.0706508
    print_check("tanh_d(0)", td0.get(0, 0), 1.0f);
    print_check("tanh_d(2)", td2.get(0, 0), 0.07065082f);

    std::cout << "\n===== MLPLayer backward formula probe: identity =====\n";

    {
        // y = W x + b
        // W = [3, 4], x = [1, 2]^T, b = [0]
        // z = 11
        // dL/dy = [1]
        // identity_d = 1
        // dL/dx = W^T * dL/dz = [3, 4]^T
        MLPLayer layer(2, 1);
        layer.setActivation(identity, identity_d);
        layer.loadWeight(Matrix{{3.0f, 4.0f}});
        layer.loadBias(Matrix{{0.0f}});

        Matrix x{{1.0f}, {2.0f}};
        Matrix out = layer.forward(x);
        print_matrix("identity forward output", out);
        print_check("identity forward value", out.get(0, 0), 11.0f);

        Matrix dx = layer.backward(Matrix{{1.0f}}, 0.0f);
        print_matrix("identity backward dx", dx);
        print_check("identity backward dx[0]", dx.get(0, 0), 3.0f);
        print_check("identity backward dx[1]", dx.get(1, 0), 4.0f);
    }

    std::cout << "\n===== MLPLayer backward formula probe: sigmoid, z = 0 =====\n";

    {
        // z = W x + b = 2*1 + (-1)*2 + 0 = 0
        // a = sigmoid(0) = 0.5
        // dL/da = 1
        // dL/dz = 1 * sigmoid'(0) = 0.25
        // dL/dx = W^T * 0.25 = [0.5, -0.25]^T
        MLPLayer layer(2, 1);
        layer.setActivation(sigmoid, sigmoid_d);
        layer.loadWeight(Matrix{{2.0f, -1.0f}});
        layer.loadBias(Matrix{{0.0f}});

        Matrix x{{1.0f}, {2.0f}};
        Matrix out = layer.forward(x);
        print_matrix("sigmoid z=0 forward output", out);
        print_check("sigmoid z=0 forward value", out.get(0, 0), 0.5f);

        Matrix dx = layer.backward(Matrix{{1.0f}}, 0.0f);
        print_matrix("sigmoid z=0 backward dx", dx);
        print_check("sigmoid z=0 dx[0]", dx.get(0, 0), 0.5f);
        print_check("sigmoid z=0 dx[1]", dx.get(1, 0), -0.25f);
    }

    std::cout << "\n===== MLPLayer backward formula probe: sigmoid, z = 2 =====\n";

    {
        // z = W x + b = 1*1 + 1*1 + 0 = 2
        // a = sigmoid(2) ~= 0.880797
        // dL/da = 1
        // dL/dz = sigmoid'(2) ~= 0.104993
        // dL/dx = W^T * 0.104993 = [0.104993, 0.104993]^T
        //
        // If your sigmoid_d incorrectly computes z * (1 - z),
        // then sigmoid_d(2) becomes -2, and dx becomes [-2, -2].
        MLPLayer layer(2, 1);
        layer.setActivation(sigmoid, sigmoid_d);
        layer.loadWeight(Matrix{{1.0f, 1.0f}});
        layer.loadBias(Matrix{{0.0f}});

        Matrix x{{1.0f}, {1.0f}};
        Matrix out = layer.forward(x);
        print_matrix("sigmoid z=2 forward output", out);
        print_check("sigmoid z=2 forward value", out.get(0, 0), 0.88079708f);

        Matrix dx = layer.backward(Matrix{{1.0f}}, 0.0f);
        print_matrix("sigmoid z=2 backward dx", dx);
        print_check("sigmoid z=2 dx[0]", dx.get(0, 0), 0.10499358f);
        print_check("sigmoid z=2 dx[1]", dx.get(1, 0), 0.10499358f);
    }

    std::cout << "\n===== backward_dz probe =====\n";

    {
        // backward_dz receives dL/dz directly, so no activation derivative should be applied.
        // W = [2, -1], dL/dz = [0.25]
        // dL/dx = W^T * 0.25 = [0.5, -0.25]^T
        MLPLayer layer(2, 1);
        layer.setActivation(sigmoid, sigmoid_d);
        layer.loadWeight(Matrix{{2.0f, -1.0f}});
        layer.loadBias(Matrix{{0.0f}});

        Matrix x{{1.0f}, {2.0f}};
        layer.forward(x);

        Matrix dx = layer.backward_dz(Matrix{{0.25f}}, 0.0f);
        print_matrix("backward_dz dx", dx);
        print_check("backward_dz dx[0]", dx.get(0, 0), 0.5f);
        print_check("backward_dz dx[1]", dx.get(1, 0), -0.25f);
    }

    return 0;
}

