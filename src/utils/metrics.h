#pragma once

#include <unordered_map>
#include <string>
#include <chrono>

#define NOW() std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()

const float ms_scale = 1.0f/1000.f;

class Metrics
{
  private:
    std::unordered_map<std::string_view, float> metrics;
    std::unordered_map<std::string_view, double> start_times;

  public:

    static Metrics& get()
    {
      static Metrics instance;
      return instance;
    }

    void set_value(std::string_view name, float value)
    {
      metrics[name] = value;
    }
    
    void increment(std::string_view name)
    {
      metrics[name] += 1;
    }

    float read(std::string_view name)
    {
      return metrics.find(name)->second;
    }

    void start_time(std::string_view name)
    {
      start_times[name] = NOW();  
    }

    void end_time(std::string_view name)
    {
      metrics[name] = static_cast<float>((NOW() - start_times[name]) * ms_scale);
    }

    //TODO WRITE EXPORT FUNCTION
};
