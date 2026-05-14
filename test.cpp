#include "src/matrix.h"
#include "src/activation.h"
#include "src/layer.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

using LibMatchstick::Matrix;
using LibMatchstick::MLPLayer;

namespace {

int g_total = 0;
int g_failed = 0;

bool nearly_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

void check(bool cond, const std::string& msg) {
    ++g_total;
    if (cond) {
        std::cout << "[PASS] " << msg << '\n';
    } else {
        ++g_failed;
        std::cout << "[FAIL] " << msg << '\n';
    }
}

void check_shape(const Matrix& m, std::size_t h, std::size_t w, const std::string& msg) {
    check(m.getHeight() == h && m.getWidth() == w,
          msg + " shape == " + std::to_string(h) + "x" + std::to_string(w));
}

void check_value(const Matrix& m, std::size_t i, std::size_t j, float expected,
                 const std::string& msg, float eps = 1e-4f) {
    check(nearly_equal(m.get(i, j), expected, eps),
          msg + " [" + std::to_string(i) + "," + std::to_string(j) + "]");
}

void check_matrix_near(const Matrix& got,
                       std::initializer_list<std::initializer_list<float>> expected,
                       const std::string& msg,
                       float eps = 1e-4f) {
    Matrix exp(expected);
    check_shape(got, exp.getHeight(), exp.getWidth(), msg);

    if (got.getHeight() != exp.getHeight() || got.getWidth() != exp.getWidth()) {
        return;
    }

    for (std::size_t i = 0; i < got.getHeight(); ++i) {
        for (std::size_t j = 0; j < got.getWidth(); ++j) {
            check(nearly_equal(got.get(i, j), exp.get(i, j), eps),
                  msg + " value [" + std::to_string(i) + "," + std::to_string(j) + "]");
        }
    }
}

bool all_finite(const Matrix& m) {
    for (std::size_t i = 0; i < m.getHeight(); ++i) {
        for (std::size_t j = 0; j < m.getWidth(); ++j) {
            if (!std::isfinite(m.get(i, j))) return false;
        }
    }
    return true;
}

void test_matrix_constructors_and_accessors() {
    std::cout << "\n===== Matrix: constructors / accessors =====\n";

    Matrix empty;
    check(empty.getHeight() == 0 || empty.getWidth() == 0,
          "default Matrix is empty or zero-sized");

    Matrix a(2, 3);
    check_shape(a, 2, 3, "Matrix(size_t, size_t)");
    a.set(0, 0, 1.0f);
    a.set(0, 1, 2.0f);
    a.set(0, 2, 3.0f);
    a.set(1, 0, 4.0f);
    a.set(1, 1, 5.0f);
    a.set(1, 2, 6.0f);
    check_value(a, 1, 2, 6.0f, "set/get");

    Matrix b{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    check_shape(b, 2, 3, "initializer_list constructor");
    check_value(b, 0, 1, 2.0f, "initializer_list constructor value");

    Matrix copied(b);
    copied.set(0, 0, 99.0f);
    check_value(b, 0, 0, 1.0f, "copy constructor performs deep copy");
    check_value(copied, 0, 0, 99.0f, "copied Matrix can be modified independently");

    Matrix assigned;
    assigned = b;
    assigned.set(1, 1, 88.0f);
    check_value(b, 1, 1, 5.0f, "copy assignment performs deep copy");
    check_value(assigned, 1, 1, 88.0f, "assigned Matrix can be modified independently");

    Matrix resized;
    resized.resize(2, 2);
    check_shape(resized, 2, 2, "resize");
    resized.set(0, 0, 7.0f);
    resized.set(1, 1, 8.0f);
    check_value(resized, 0, 0, 7.0f, "resize then set/get 1");
    check_value(resized, 1, 1, 8.0f, "resize then set/get 2");

    // CUDA 版本里 getData() 很可能返回的是 device pointer。
    // Host 端不能直接 raw[0] 读写，否则会 segfault。
    float* raw = resized.getData();
    check(raw != nullptr, "non-const getData returns non-null pointer for non-empty Matrix");

    const Matrix& cref = resized;
    const float* craw = cref.getData();
    check(craw != nullptr, "const getData returns non-null pointer for non-empty Matrix");

    std::ostringstream oss;
    oss << b;
    check(!oss.str().empty(), "operator<< writes something to stream");
}

void test_matrix_operators() {
    std::cout << "\n===== Matrix: operators =====\n";

    Matrix a{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix b{{6.0f, 5.0f, 4.0f}, {3.0f, 2.0f, 1.0f}};

    check_matrix_near(a + b, {{7.0f, 7.0f, 7.0f}, {7.0f, 7.0f, 7.0f}}, "operator+");
    check_matrix_near(a - b, {{-5.0f, -3.0f, -1.0f}, {1.0f, 3.0f, 5.0f}}, "operator-");

    Matrix plus_eq = a;
    plus_eq += b;
    check_matrix_near(plus_eq, {{7.0f, 7.0f, 7.0f}, {7.0f, 7.0f, 7.0f}}, "operator+=");

    Matrix minus_eq = a;
    minus_eq -= b;
    check_matrix_near(minus_eq, {{-5.0f, -3.0f, -1.0f}, {1.0f, 3.0f, 5.0f}}, "operator-=");

    check_matrix_near(a.hadamard(b), {{6.0f, 10.0f, 12.0f}, {12.0f, 10.0f, 6.0f}}, "hadamard");

    Matrix m1{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix m2{{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}};
    check_matrix_near(m1 * m2, {{58.0f, 64.0f}, {139.0f, 154.0f}}, "matrix multiplication operator*");

    check_matrix_near(m1.transpose(), {{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0f}}, "transpose");

    check_matrix_near(a * 2.0f, {{2.0f, 4.0f, 6.0f}, {8.0f, 10.0f, 12.0f}}, "Matrix * scalar");
    check_matrix_near(2.0f * a, {{2.0f, 4.0f, 6.0f}, {8.0f, 10.0f, 12.0f}}, "scalar * Matrix");

    Matrix scaled = a;
    Matrix scaled_ret = (scaled *= 3.0f);
    check_matrix_near(scaled, {{3.0f, 6.0f, 9.0f}, {12.0f, 15.0f, 18.0f}}, "operator*= mutates Matrix");
    check_matrix_near(scaled_ret, {{3.0f, 6.0f, 9.0f}, {12.0f, 15.0f, 18.0f}}, "operator*= returns scaled Matrix");
}

void test_activation() {
    std::cout << "\n===== Activation =====\n";

    Matrix x{{-1.0f, 0.0f, 2.0f}};

    check_matrix_near(LibMatchstick::Activation::relu(x), {{0.0f, 0.0f, 2.0f}}, "relu");
    check_shape(LibMatchstick::Activation::relu_d(x), 1, 3, "relu_d");

    Matrix leaky = LibMatchstick::Activation::leaky_relu(x);
    check_shape(leaky, 1, 3, "leaky_relu");
    check(all_finite(leaky), "leaky_relu finite values");
    check(nearly_equal(leaky.get(0, 2), 2.0f), "leaky_relu keeps positive value");
    check_shape(LibMatchstick::Activation::leaky_relu_d(x), 1, 3, "leaky_relu_d");

    Matrix sig0{{0.0f}};
    check_matrix_near(LibMatchstick::Activation::sigmoid(sig0), {{0.5f}}, "sigmoid at 0");
    check_shape(LibMatchstick::Activation::sigmoid_d(x), 1, 3, "sigmoid_d");
    check(all_finite(LibMatchstick::Activation::sigmoid_d(x)), "sigmoid_d finite values");

    Matrix tanh0{{0.0f}};
    check_matrix_near(LibMatchstick::Activation::tanh(tanh0), {{0.0f}}, "tanh at 0");
    check_shape(LibMatchstick::Activation::tanh_d(x), 1, 3, "tanh_d");
    check(all_finite(LibMatchstick::Activation::tanh_d(x)), "tanh_d finite values");

    check_matrix_near(LibMatchstick::Activation::identity(x), {{-1.0f, 0.0f, 2.0f}}, "identity");
    check_shape(LibMatchstick::Activation::identity_d(x), 1, 3, "identity_d");
    check(all_finite(LibMatchstick::Activation::identity_d(x)), "identity_d finite values");

    Matrix sm_in{{1.0f, 2.0f, 3.0f}};
    Matrix sm = LibMatchstick::Activation::softmax(sm_in);
    check_shape(sm, 1, 3, "softmax");
    check(all_finite(sm), "softmax finite values");
    float sm_sum = 0.0f;
    for (std::size_t j = 0; j < sm.getWidth(); ++j) sm_sum += sm.get(0, j);
    check(nearly_equal(sm_sum, 1.0f, 1e-3f), "softmax row sum is approximately 1");
    check(sm.get(0, 2) > sm.get(0, 1) && sm.get(0, 1) > sm.get(0, 0),
          "softmax preserves order for increasing input");

    Matrix smd = LibMatchstick::Activation::softmax_d(sm_in);
    check_shape(smd, sm_in.getHeight(), sm_in.getWidth(), "softmax_d");
    check(all_finite(smd), "softmax_d finite values");
}

Matrix ones(std::size_t h, std::size_t w) {
    Matrix m(h, w);
    for (std::size_t i = 0; i < h; ++i) {
        for (std::size_t j = 0; j < w; ++j) {
            m.set(i, j, 1.0f);
        }
    }
    return m;
}

Matrix make_layer_input_from_weight_shape(const Matrix& w, std::size_t in_size) {
    // 常见写法：W 是 out_size x in_size，输入是 in_size x 1。
    if (w.getWidth() == in_size) {
        Matrix input(in_size, 1);
        for (std::size_t i = 0; i < in_size; ++i) input.set(i, 0, static_cast<float>(i + 1));
        return input;
    }

    // 另一种写法：输入是 1 x in_size，W 是 in_size x out_size。
    if (w.getHeight() == in_size) {
        Matrix input(1, in_size);
        for (std::size_t j = 0; j < in_size; ++j) input.set(0, j, static_cast<float>(j + 1));
        return input;
    }

    // 如果实现采用了别的布局，这里仍然给一个最常见的列向量输入。
    Matrix input(in_size, 1);
    for (std::size_t i = 0; i < in_size; ++i) input.set(i, 0, static_cast<float>(i + 1));
    return input;
}

void test_layer() {
    std::cout << "\n===== MLPLayer =====\n";

    constexpr std::size_t in_size = 2;
    constexpr std::size_t out_size = 3;

    MLPLayer default_layer;
    (void)default_layer;
    check(true, "MLPLayer default constructor compiles");

    MLPLayer layer(in_size, out_size);
    layer.init(-0.1f, 0.1f);
    check(true, "MLPLayer(size_t, size_t) and init(low, high) compile/run");

    Matrix saved_w = layer.saveWeight();
    Matrix saved_b = layer.saveBias();
    check(saved_w.getHeight() > 0 && saved_w.getWidth() > 0, "saveWeight returns non-empty Matrix");
    check(saved_b.getHeight() > 0 && saved_b.getWidth() > 0, "saveBias returns non-empty Matrix");

    bool weight_loaded = layer.loadWeight(saved_w);
    bool bias_loaded = layer.loadBias(saved_b);
    check(weight_loaded, "loadWeight accepts matrix produced by saveWeight");
    check(bias_loaded, "loadBias accepts matrix produced by saveBias");

    Matrix input = make_layer_input_from_weight_shape(saved_w, in_size);
    Matrix output = layer.forward(input);
    check(output.getHeight() > 0 && output.getWidth() > 0, "forward returns non-empty Matrix");
    check(all_finite(output), "forward output finite values");

    Matrix dl_da = ones(output.getHeight(), output.getWidth());
    Matrix dl_prev = layer.backward(dl_da, 0.01f);
    check(dl_prev.getHeight() > 0 && dl_prev.getWidth() > 0, "backward returns non-empty Matrix");
    check(all_finite(dl_prev), "backward output finite values");

    MLPLayer layer2(in_size, out_size);
    layer2.init(-0.1f, 0.1f);
    Matrix saved_w2 = layer2.saveWeight();
    Matrix input2 = make_layer_input_from_weight_shape(saved_w2, in_size);
    Matrix output2 = layer2.forward(input2);
    Matrix dl_dz = ones(output2.getHeight(), output2.getWidth());

    Matrix dl_prev2 = layer2.backward_dz(dl_dz, 0.01f);
    check(dl_prev2.getHeight() > 0 && dl_prev2.getWidth() > 0, "backward_dz returns non-empty Matrix");
    check(all_finite(dl_prev2), "backward_dz output finite values");

    Matrix wrong_w(saved_w.getHeight() + 1, saved_w.getWidth() + 1);
    Matrix wrong_b(saved_b.getHeight() + 1, saved_b.getWidth() + 1);
    check(!layer.loadWeight(wrong_w), "loadWeight rejects obviously wrong shape");
    check(!layer.loadBias(wrong_b), "loadBias rejects obviously wrong shape");
}

} // namespace

int main() {
    test_matrix_constructors_and_accessors();
    test_matrix_operators();
    test_activation();
    test_layer();

    std::cout << "\n===== Summary =====\n";
    std::cout << "Total checks: " << g_total << '\n';
    std::cout << "Failed checks: " << g_failed << '\n';

    if (g_failed == 0) {
        std::cout << "All tests passed.\n";
        return EXIT_SUCCESS;
    }

    std::cout << "Some tests failed.\n";
    return EXIT_FAILURE;
}

