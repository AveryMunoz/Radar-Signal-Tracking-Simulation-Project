#include "association.h"
#include <limits>

// Same methodology as python files (utilizing Nearest Neighbor or PDA for state estimation)

int associate_nn(const std::vector<Vector>& measurements, const Vector& z_pred, const Matrix& S) {
    Matrix S_inv = S.inverse();
    double best_d2 = std::numeric_limits<double>::infinity();
    int best_index = -1; 

    for (int i = 0; i < measurements.size(); ++i) {
        Vector diff = measurements[i] - z_pred;
        double d2 = diff.transpose() * S_inv * diff;

        if (d2 < best_d2) {
            best_d2 = d2;
            best_index = i;
        }
    }

    return best_index;
}

PDAResult associate_pda(const std::vector<Vector>& measurements, const Vector& z_pred, const Matrix& R ) {
    PDAResult result;

    int m = static_cast<int>(measurements.size());
    if (m == 0) {
        // Match Python: no measurements
        result.betas.clear();
        result.z_fused = Vector();  // empty
        return result;
    }

    Matrix R_inv = R.inverse();

    std::vector<double> likelihoods(m);
    double sum_likelihoods = 0.0;

    // Compute Mahalanobis distances and likelihoods
    for (int i = 0; i < m; ++i) {
        Vector diff = measurements[i] - z_pred;
        double d2 = (diff.transpose() * R_inv * diff)(0, 0);

        double L = std::exp(-0.5 * d2);
        likelihoods[i] = L;
        sum_likelihoods += L;
    }

    // Compute betas (normalized likelihoods)
    result.betas.resize(m);
    if (sum_likelihoods == 0.0) {
        double equal_beta = 1.0 / static_cast<double>(m);
        for (int i = 0; i < m; ++i) {
            result.betas[i] = equal_beta;
        }
    } else {
        for (int i = 0; i < m; ++i) {
            result.betas[i] = likelihoods[i] / sum_likelihoods;
        }
    }

    // Fused measurement z̄ = Σ βᵢ zᵢ
    result.z_fused = Vector::Zero(z_pred.size());
    for (int i = 0; i < m; ++i) {
        result.z_fused += result.betas[i] * measurements[i];
    }

    return result;
}

