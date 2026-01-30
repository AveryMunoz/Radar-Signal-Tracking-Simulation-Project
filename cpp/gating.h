#pragma once
#include <Eigen/Dense>
#include <vector>

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// Return indices of measurements inside the gate
std::vector<int> gate_measurements(const std::vector<Vector>& measurements, const Vector& z_pred, double gate_threshold);

