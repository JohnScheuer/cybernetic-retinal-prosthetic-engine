#ifndef THERMAL_SAFETY_GUARD_HPP
#define THERMAL_SAFETY_GUARD_HPP

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

// Sub-Dermal Chromium Photonic & Thermal Safety Sentinel (ISO 14708-3 / EN 45502)
class ThermalSafetyGuard {
public:
    static constexpr float MAX_ENERGY_DENSITY_UJ_MM2 = 2.0f;
    static constexpr float MAX_THERMAL_RISE_DEGC = 0.5f;
    static constexpr float TISSUE_SPECIFIC_HEAT = 3.6f; // J / (g * °C)
    static constexpr float TISSUE_DENSITY = 1.05f;       // g / cm^3
    static constexpr size_t SLIDING_WINDOW_SIZE = 100;   // 100ms Sliding Window @ 1000 Hz

    struct SafetyStatus {
        bool is_safe;
        float peak_energy_density_uJ_mm2;
        float sliding_window_thermal_rise_degC;
        float safety_margin_percent;
    };

    ThermalSafetyGuard() : window_idx_(0) {
        energy_window_.assign(SLIDING_WINDOW_SIZE, 0.0f);
    }

    SafetyStatus verify_stimulation_burst(const float* currents, int channels, float duration_us) {
        float max_current_uA = 0.0f;
        float total_power_uW = 0.0f;

        for (int i = 0; i < channels; ++i) {
            float abs_i = std::abs(currents[i]);
            if (abs_i > max_current_uA) {
                max_current_uA = abs_i;
            }
            total_power_uW += (abs_i * abs_i * 1e-12f) * 10000.0f * 1e6f;
        }

        float burst_energy_uJ = total_power_uW * (duration_us * 1e-6f);
        
        energy_window_[window_idx_] = burst_energy_uJ;
        window_idx_ = (window_idx_ + 1) % SLIDING_WINDOW_SIZE;

        float accumulated_energy_uJ = std::accumulate(energy_window_.begin(), energy_window_.end(), 0.0f);

        float electrode_area_mm2 = 0.05f;
        float peak_energy_density = (max_current_uA * max_current_uA * 1e-12f * 10000.0f * duration_us * 1e-6f * 1e6f) / electrode_area_mm2;

        float mass_g = electrode_area_mm2 * 0.1f * 1e-3f * TISSUE_DENSITY;
        float delta_T = (accumulated_energy_uJ * 1e-6f) / (mass_g * TISSUE_SPECIFIC_HEAT * 100.0f);

        bool safe = (peak_energy_density <= MAX_ENERGY_DENSITY_UJ_MM2) && (delta_T <= MAX_THERMAL_RISE_DEGC);
        float margin = ((MAX_THERMAL_RISE_DEGC - delta_T) / MAX_THERMAL_RISE_DEGC) * 100.0f;

        return {safe, peak_energy_density, delta_T, std::max(0.0f, margin)};
    }

private:
    size_t window_idx_;
    std::vector<float> energy_window_;
};

#endif // THERMAL_SAFETY_GUARD_HPP
