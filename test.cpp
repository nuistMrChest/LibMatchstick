#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "src/matrix.h"
#include "src/activation.h"
#include "src/layer.h"

using LibMatchstick::Matrix;
using LibMatchstick::MLPLayer;

namespace {

int passed = 0;
int failed = 0;

bool nearly_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

bool finite_matrix(const Matrix& m) {
    for (size_t i = 0; i < m.getHeight(); ++i) {
        for (size_t j = 0; j < m.getWidth(); ++j) {
            if (!std::isfinite(m.get(i, j))) {
                return false;
            }
        }
    }
    return true;
}

void check(bool condition, const std::string& name) {
    if (condition) {
        ++passed;
        std::cout << "[PASS] " << name << '\n';
    } else {
        ++failed;
        std::cout << "[FAIL] " << name << '\n';
    }
}

void check_shape(const Matrix& m, size_t h, size_t w, const std::string& name) {
    check(m.getHeight() == h && m.getWidth() == w,
          name + " shape == " + std::to_string(h) + "x" + std::to_string(w));
}

void check_value(const Matrix& m, size_t i, size_t j, float expected,
                 const std::string& name, float eps = 1e-4f) {
    check(nearly_equal(m.get(i, j), expected, eps),
          name + " value [" + std::to_string(i) + "," + std::to_string(j) + "]");
}

void check_matrix(const Matrix& m, const Matrix& expected,
                  const std::string& name, float eps = 1e-4f) {
    check_shape(m, expected.getHeight(), expected.getWidth(), name);
    const size_t h = std::min(m.getHeight(), expected.getHeight());
    const size_t w = std::min(m.getWidth(), expected.getWidth());
    for (size_t i = 0; i < h; ++i) {
        for (size_t j = 0; j < w; ++j) {
            check_value(m, i, j, expected.get(i, j), name, eps);
        }
    }
}

void section(const std::string& title) {
    std::cout << "\n===== " << title << " =====\n";
}

void test_matrix_constructors_accessors() {
    section("Matrix: constructors / accessors");

    Matrix empty;
    check(empty.getHeight() == 0 || empty.getWidth() == 0,
          "default Matrix is empty or zero-sized");

    Matrix a(2, 3);
    check_shape(a, 2, 3, "Matrix(size_t, size_t)");
    a.set(1, 2, 7.5f);
    check_value(a, 1, 2, 7.5f, "set/get [1,2]");

    Matrix b{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    check_shape(b, 2, 3, "initializer_list constructor");
    check_value(b, 0, 1, 2.0f, "initializer_list constructor");

    Matrix copied(b);
    check_value(copied, 0, 0, 1.0f, "copy constructor performs deep copy");
    copied.set(0, 0, 99.0f);
    check_value(b, 0, 0, 1.0f,
                "copied Matrix can be modified independently");

    Matrix assigned;
    assigned = b;
    check_value(assigned, 1, 1, 5.0f,
                "copy assignment performs deep copy");
    assigned.set(1, 1, 88.0f);
    check_value(b, 1, 1, 5.0f,
                "assigned Matrix can be modified independently");

    Matrix resized;
    resized.resize(2, 2);
    resized.set(0, 0, 11.0f);
    resized.set(1, 1, 22.0f);
    check_shape(resized, 2, 2, "resize");
    check_value(resized, 0, 0, 11.0f, "resize then set/get 1");
    check_value(resized, 1, 1, 22.0f, "resize then set/get 2");

    // CUDA 版本的 Matrix::getData() 很可能返回 device pointer。
    // Host 端不能直接 raw[0] 读写，所以这里只测非空。
    float* raw = resized.getData();
    check(raw != nullptr,
          "non-const getData returns non-null pointer for non-empty Matrix");

    const Matrix& cref = resized;
    const float* craw = cref.getData();
    check(craw != nullptr,
          "const getData returns non-null pointer for non-empty Matrix");

    std::ostringstream oss;
    oss << b;
    check(!oss.str().empty(), "operator<< writes something to stream");
}

void test_matrix_operators() {
    section("Matrix: operators");

    Matrix a{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix b{{6.0f, 5.0f, 4.0f}, {3.0f, 2.0f, 1.0f}};

    check_matrix(a + b,
                 Matrix{{7.0f, 7.0f, 7.0f}, {7.0f, 7.0f, 7.0f}},
                 "operator+");
    check_matrix(a - b,
                 Matrix{{-5.0f, -3.0f, -1.0f}, {1.0f, 3.0f, 5.0f}},
                 "operator-");

    Matrix add_assign = a;
    add_assign += b;
    check_matrix(add_assign,
                 Matrix{{7.0f, 7.0f, 7.0f}, {7.0f, 7.0f, 7.0f}},
                 "operator+=");

    Matrix sub_assign = a;
    sub_assign -= b;
    check_matrix(sub_assign,
                 Matrix{{-5.0f, -3.0f, -1.0f}, {1.0f, 3.0f, 5.0f}},
                 "operator-=");

    check_matrix(a.hadamard(b),
                 Matrix{{6.0f, 10.0f, 12.0f}, {12.0f, 10.0f, 6.0f}},
                 "hadamard");

    Matrix c{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
    check_matrix(a * c,
                 Matrix{{22.0f, 28.0f}, {49.0f, 64.0f}},
                 "matrix multiplication operator*");

    check_matrix(a.transpose(),
                 Matrix{{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0f}},
                 "transpose");

    check_matrix(a * 2.0f,
                 Matrix{{2.0f, 4.0f, 6.0f}, {8.0f, 10.0f, 12.0f}},
                 "Matrix * scalar");
    check_matrix(2.0f * a,
                 Matrix{{2.0f, 4.0f, 6.0f}, {8.0f, 10.0f, 12.0f}},
                 "scalar * Matrix");

    Matrix scaled = a;
    Matrix scaled_ret = (scaled *= 3.0f);
    Matrix expected_scaled{{3.0f, 6.0f, 9.0f}, {12.0f, 15.0f, 18.0f}};
    check_matrix(scaled, expected_scaled, "operator*= mutates Matrix");
    check_matrix(scaled_ret, expected_scaled, "operator*= returns scaled Matrix");
}

void test_activation() {
    section("Activation");

    using namespace LibMatchstick::Activation;

    Matrix x{{-1.0f, 0.0f, 2.0f}};

    check_matrix(relu(x), Matrix{{0.0f, 0.0f, 2.0f}}, "relu");
    check_shape(relu_d(x), 1, 3, "relu_d");

    Matrix lr = leaky_relu(x);
    check_shape(lr, 1, 3, "leaky_relu");
    check(finite_matrix(lr), "leaky_relu finite values");
    check_value(lr, 0, 2, 2.0f, "leaky_relu keeps positive value");
    check_shape(leaky_relu_d(x), 1, 3, "leaky_relu_d");

    Matrix zero{{0.0f}};
    check_matrix(sigmoid(zero), Matrix{{0.5f}}, "sigmoid at 0");
    check_shape(sigmoid_d(x), 1, 3, "sigmoid_d");
    check(finite_matrix(sigmoid_d(x)), "sigmoid_d finite values");

    check_matrix(tanh(zero), Matrix{{0.0f}}, "tanh at 0");
    check_shape(tanh_d(x), 1, 3, "tanh_d");
    check(finite_matrix(tanh_d(x)), "tanh_d finite values");

    check_matrix(identity(x), x, "identity");
    check_shape(identity_d(x), 1, 3, "identity_d");
    check(finite_matrix(identity_d(x)), "identity_d finite values");

    Matrix sm = softmax(x);
    check_shape(sm, 1, 3, "softmax");
    check(finite_matrix(sm), "softmax finite values");

    float sum = 0.0f;
    for (size_t j = 0; j < sm.getWidth(); ++j) {
        sum += sm.get(0, j);
    }
    check(nearly_equal(sum, 1.0f, 1e-3f),
          "softmax row sum is approximately 1");
    check(sm.get(0, 2) > sm.get(0, 1) && sm.get(0, 1) > sm.get(0, 0),
          "softmax preserves order for increasing input");

    Matrix smd = softmax_d(x);
    check_shape(smd, 1, 3, "softmax_d");
    check(finite_matrix(smd), "softmax_d finite values");
}

void test_mlp_layer() {
    section("MLPLayer");

    using namespace LibMatchstick::Activation;

    MLPLayer default_layer;
    check(true, "MLPLayer default constructor compiles");

    MLPLayer layer(3, 2);
    layer.init(0.0f, 0.0f);
    check(true, "MLPLayer(size_t, size_t) and init(low, high) compile/run");

    Matrix saved_w = layer.saveWeight();
    Matrix saved_b = layer.saveBias();
    check(saved_w.getHeight() != 0 && saved_w.getWidth() != 0,
          "saveWeight returns non-empty Matrix");
    check(saved_b.getHeight() != 0 && saved_b.getWidth() != 0,
          "saveBias returns non-empty Matrix");
    check(layer.loadWeight(saved_w),
          "loadWeight accepts matrix produced by saveWeight");
    check(layer.loadBias(saved_b),
          "loadBias accepts matrix produced by saveBias");

    Matrix known_w{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix known_b{{0.5f}, {-0.5f}};
    check(layer.loadWeight(known_w),
          "loadWeight accepts 2x3 Matrix for MLPLayer(3,2)");
    check(layer.loadBias(known_b),
          "loadBias accepts 2x1 Matrix for MLPLayer(3,2)");

    Matrix input{{1.0f}, {2.0f}, {3.0f}};
    Matrix out = layer.forward(input);
    check_shape(out, 2, 1, "forward with default identity activation");
    check_value(out, 0, 0, 14.5f, "forward identity output 0");
    check_value(out, 1, 0, 31.5f, "forward identity output 1");

    MLPLayer id_layer(3, 2);
    Matrix zero_w{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    Matrix neg_b{{-2.0f}, {-3.0f}};
    check(id_layer.loadWeight(zero_w), "identity test load zero weight");
    check(id_layer.loadBias(neg_b), "identity test load negative bias");

    Matrix id_out = id_layer.forward(input);
    check_value(id_out, 0, 0, -2.0f,
                "default activation is identity, output keeps negative value 0");
    check_value(id_out, 1, 0, -3.0f,
                "default activation is identity, output keeps negative value 1");

    id_layer.setActivation(relu, relu_d);
    Matrix relu_out = id_layer.forward(input);
    check_value(relu_out, 0, 0, 0.0f,
                "setActivation(relu) clamps negative value 0");
    check_value(relu_out, 1, 0, 0.0f,
                "setActivation(relu) clamps negative value 1");

    id_layer.setActivation(identity, identity_d);
    Matrix id_out_again = id_layer.forward(input);
    check_value(id_out_again, 0, 0, -2.0f,
                "setActivation(identity) restores negative value 0");
    check_value(id_out_again, 1, 0, -3.0f,
                "setActivation(identity) restores negative value 1");

    Matrix grad{{1.0f}, {1.0f}};
    Matrix back = layer.backward(grad, 0.01f);
    check_shape(back, 3, 1, "backward returns gradient shaped like input");
    check(finite_matrix(back), "backward returns finite values");

    Matrix back_dz = layer.backward_dz(grad, 0.01f);
    check_shape(back_dz, 3, 1, "backward_dz returns gradient shaped like input");
    check(finite_matrix(back_dz), "backward_dz returns finite values");
}

} // namespace

int main() {
    try {
        test_matrix_constructors_accessors();
        test_matrix_operators();
        test_activation();
        test_mlp_layer();

        std::cout << "\n===== Summary =====\n";
        std::cout << "Passed: " << passed << '\n';
        std::cout << "Failed: " << failed << '\n';

        return failed == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "\n[ABORT] Uncaught std::exception: " << e.what() << '\n';
        return 2;
    } catch (...) {
        std::cerr << "\n[ABORT] Unknown uncaught exception\n";
        return 3;
    }
}

