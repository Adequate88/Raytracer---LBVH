#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

#define NOW()                                                                  \
  std::chrono::duration<double, std::milli>(                                   \
      std::chrono::high_resolution_clock::now().time_since_epoch())            \
      .count()

constexpr float ms_scale = 1.0f / 1000.f;

class Metrics {
private:
  std::unordered_map<std::string_view, float> metrics;
  std::unordered_map<std::string_view, double> start_times;
  std::unordered_map<std::string_view, std::vector<float>> samples;
  std::unordered_map<std::string_view, float> stddevs;
  bool _sampling = false;

public:
  static Metrics &get() {
    static Metrics instance;
    return instance;
  }

  void set_value(std::string_view name, float value) {
    metrics[name] = value;
    if (_sampling)
      samples[name].push_back(value);
  }

  void increment(std::string_view name) { metrics[name] += 1; }

  float read(std::string_view name) { return metrics.find(name)->second; }

  void start_time(std::string_view name) { start_times[name] = NOW(); }

  void end_time(std::string_view name) {
    float value = static_cast<float>((NOW() - start_times[name]));
    metrics[name] = value;
    if (_sampling)
      samples[name].push_back(value);
  }

  void begin_sampling() {
    samples.clear();
    _sampling = true;
  }

  void end_sampling() {
    _sampling = false;
    for (const auto &[name, values] : samples) {
      if (values.empty())
        continue;

      double sum = 0.0;
      for (float v : values)
        sum += v;
      double mean = sum / values.size();

      double var = 0.0;
      for (float v : values)
        var += (v - mean) * (v - mean);
      var = values.size() > 1 ? var / (values.size() - 1) : 0.0;

      metrics[name] = static_cast<float>(mean);
      stddevs[name] = static_cast<float>(std::sqrt(var));
    }
  }

  void export_to(const std::string &path) {
    std::ofstream file(path);

    if (!file) {
      std::cout << "Failed to open file" << std::endl;
      return;
    }

    std::map<std::string_view, float> sorted(metrics.begin(), metrics.end());
    for (const auto &[name, value] : sorted) {
      auto stddev = stddevs.find(name);
      if (stddev != stddevs.end())
        file << name << "," << value << "," << stddev->second << "\n";
      else
        file << name << "," << value << "\n";
    }
  }
};
