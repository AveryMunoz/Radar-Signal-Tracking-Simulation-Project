#include "gating.h"

std::vector<int> gate_measurements(const std::vector<Vector>& measurements, const Vector& z_pred, double gate_threshold) {
    
    std::vector<int> inside;

    for (int i = 0; i < measurements.size(); ++i) {
        Vector diff = measurements[i] - z_pred;
        double dist = diff.norm();   // Euclidean distance

        if (dist <= gate_threshold) {
            inside.push_back(i);
        }
    }

    return inside;
}
