#ifndef FOVEAL_SACCADE_CONTROLLER_HPP
#define FOVEAL_SACCADE_CONTROLLER_HPP

#include <cmath>

// Kiroshi Saccadic Glitch Suppressor & Motion Blur Stabilizer
class FovealSaccadeController {
public:
    static constexpr float SACCADE_THRESHOLD_DEG_SEC = 300.0f;

    struct State {
        bool in_saccade;
        float angular_velocity_deg_sec;
        float stimulation_gain;
    };

    State update(float camera_dx_deg, float camera_dy_deg, float dt_sec) {
        float velocity = std::sqrt(camera_dx_deg * camera_dx_deg + camera_dy_deg * camera_dy_deg) / dt_sec;
        bool saccade = velocity >= SACCADE_THRESHOLD_DEG_SEC;
        
        float gain = saccade ? 0.1f : 1.0f; // Dampens current output during fast ocular saccades
        return {saccade, velocity, gain};
    }
};

#endif // FOVEAL_SACCADE_CONTROLLER_HPP
