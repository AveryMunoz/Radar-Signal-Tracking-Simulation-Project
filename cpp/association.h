#pragma once
#include <Eigen/Dense>
#include <vector>

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// Nearest Neighbor association: returns index of closest measurement
int associate_nn(const std::vector<Vector>& measurements, const Vector& z_pred, const Matrix& R);

// PDA association: returns fused measurement and beta weights
struct PDAResult {
    Vector z_fused; 
    std::vector<double> betas;   
};

PDAResult associate_pda(const std::vector<Vector>& measurements, const Vector& z_pred, const Matrix& R);
