#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>

#define NOW()                                                                  \
  std::chrono::duration<double, std::milli>(                                   \
      std::chrono::high_resolution_clock::now().time_since_epoch())            \
      .count()

constexpr float ms_scale = 1.0f / 1000.f;

class Metrics {
private:
  std::unordered_map<std::string_view, float> metrics;
  std::unordered_map<std::string_view, double> start_times;

public:
  static Metrics &get() {
    static Metrics instance;
    return instance;
  }

  void set_value(std::string_view name, float value) { metrics[name] = value; }

  void increment(std::string_view name) { metrics[name] += 1; }

  float read(std::string_view name) { return metrics.find(name)->second; }

  void start_time(std::string_view name) { start_times[name] = NOW(); }

  void end_time(std::string_view name) {
    metrics[name] = static_cast<float>((NOW() - start_times[name]));
  }

  void export_to(const std::string &path) {
    std::ofstream file(path);

    if (!file) {
      std::cout << "Failed to open file" << std::endl;
      return;
    }

    std::map<std::string_view, float> sorted(metrics.begin(), metrics.end());
    for (const auto &[name, value] : sorted) {
      file << name << "," << value << "\n";
    }
  }
};
