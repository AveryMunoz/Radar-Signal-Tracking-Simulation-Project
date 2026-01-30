#pragma once
#include <Eigen/Dense>

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

void kalman_predict(Eigen::Ref<Matrix> x,
                    Eigen::Ref<Matrix> P,
                    const Eigen::Ref<const Matrix> F,
                    const Eigen::Ref<const Matrix> Q);

void kalman_update(Eigen::Ref<Matrix> x,
                   Eigen::Ref<Matrix> P,
                   const Eigen::Ref<const Matrix> z,
                   const Eigen::Ref<const Matrix> H,
                   const Eigen::Ref<const Matrix> R);
