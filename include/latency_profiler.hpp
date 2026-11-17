#ifndef LATENCY_PROFILER_HPP
#define LATENCY_PROFILER_HPP

#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

class LatencyProfiler {
public:
    using Clock = std::chrono::high_resolution_clock;

    void start() {
        start_time_ = Clock::now();
    }

    double stop() {
        auto end_time = Clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end_time - start_time_).count();
        latencies_us_.push_back(elapsed_us);
        return elapsed_us;
    }

    struct Summary {
        double min_us;
        double mean_us;
        double p50_us;
        double p90_us;
        double p99_us;
        double p99_9_us;
        double max_us;
        double sla_violations_percent;
    };

    Summary get_summary(double sla_threshold_us = 500.0) const {
        if (latencies_us_.empty()) return {};

        std::vector<double> sorted = latencies_us_;
        std::sort(sorted.begin(), sorted.end());

        size_t n = sorted.size();
        double min_us = sorted.front();
        double max_us = sorted.back();
        double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        double mean_us = sum / n;

        double p50 = sorted[static_cast<size_t>(n * 0.50)];
        double p90 = sorted[static_cast<size_t>(n * 0.90)];
        double p99 = sorted[static_cast<size_t>(n * 0.99)];
        double p99_9 = sorted[static_cast<size_t>(n * 0.999)];

        size_t violations = 0;
        for (double l : sorted) {
            if (l > sla_threshold_us) ++violations;
        }
        double violation_rate = (static_cast<double>(violations) / n) * 100.0;

        return {min_us, mean_us, p50, p90, p99, p99_9, max_us, violation_rate};
    }

    const std::vector<double>& raw_latencies() const { return latencies_us_; }

private:
    Clock::time_point start_time_;
    std::vector<double> latencies_us_;
};

#endif // LATENCY_PROFILER_HPP
