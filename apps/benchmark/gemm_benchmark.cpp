#include "math/Matrix.hpp"
#include "math/clFMatrix.hpp"
#include "clPlatformSelector.hpp"
#include "clWrapper.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace math;

void warmup(utils::clWrapper &wrapper) {
  std::cout << "Warming up...";
  // BLAS warmup
  FloatMatrix A(1024, 1024);
  randomize(A, -1.0f, 1.0f);
  FloatMatrix B(1024, 1024);
  randomize(B, -1.0f, 1.0f);


  // clblast warmup
  auto C = FloatMatrix::mul(false, A, false, B);
  clFMatrix a(A);
  clFMatrix b(B);
  auto c = clFMatrix::gemm(1.0f, false, a, false, b, true);
  std::cout << " done" << std::endl;
}

void printHeader() {
  printf("+");
  for (int i = 0; i < 60; i++) { printf("-"); }
  printf("+\n");
  printf("|");
  printf(" %9s | %12s || %15s | %12s ", "TYPE", "N", "TIME", "GFLOPS");
  printf("|\n");
  printf("+");
  for (int i = 0; i < 60; i++) { printf("-"); }
  printf("+\n");
}

void printResult(const std::string &type, size_t n, std::chrono::nanoseconds time, double gflops) {
  printf("|");
  printf(" %9s | %12ld || %15ld | %12.3f ", type.c_str(), n, time.count(), gflops);
  printf("|\n");
}

std::vector<double> benchmarkBlas(int n_start, int n_end, int step) {
  std::vector<double> res;

  printHeader();

  for (int n = n_start; n <= n_end; n += step) {
    FloatMatrix A(n, n);
    FloatMatrix B(n, n);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 20; i++) { FloatMatrix C = FloatMatrix::mul(false, A, false, B); }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    duration /= 20;
    double nflp = (2.0 * n * n * n);
    double gflops = nflp / (duration.count() * 1e-9) * 1e-9;
    printResult("BLAS", n, duration, gflops);
    res.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(duration).count());
  }
  return res;
}

std::vector<double> benchmarkCLBlast(int n_start, int n_end, int step, utils::clWrapper &wrapper) {
  std::vector<double> res;

  auto &queue = wrapper.getDefaultQueue();

  for (int n = n_start; n <= n_end; n += step) {
    clFMatrix A(n, n);
    clFMatrix B(n, n);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 20; i++) { clFMatrix C = clFMatrix::gemm(1.0f, false, A, false, B, queue); }
    queue.finish();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    duration /= 20;
    double nflp = (2.0 * n * n * n);
    double gflops = nflp / (duration.count() * 1e-9) * 1e-9;
    printResult("CLBLAST", n, duration, gflops);
    res.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(duration).count());
  }
  return res;
}

void outputValues(const std::vector<double> &values, const std::string &filename, int start,
                  int step) {
  std::ofstream file(filename);
  int curr = start;
  for (auto v : values) {
    file << curr * curr << " " << v << std::endl;
    curr += step;
  }
}

int main() {
  std::cout << "Benchmarking Matrix Multiplication with OpenCL and BLAS" << std::endl;
  std::cout << "Matrix Multiplication: A * B = C" << std::endl;

  constexpr int n_start = 2;
  constexpr int n_end = 1096;
  constexpr int step = 2;
  std::cout << "m = [" << n_start << ";" << n_end << "], step = " << step << std::endl;

  auto wrapper = utils::clPlatformSelector::execute();
  warmup(*wrapper);

  auto blas_values = benchmarkBlas(n_start, n_end, step);
  auto clblast_values = benchmarkCLBlast(n_start, n_end, step, *wrapper);

  outputValues(blas_values, "blas_values.txt", n_start, step);
  outputValues(clblast_values, "clblast_values.txt", n_start, step);
  return 0;
}
