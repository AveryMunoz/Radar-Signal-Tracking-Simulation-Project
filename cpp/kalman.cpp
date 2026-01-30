#include "kalman.h"

void kalman_predict(Eigen::Ref<Matrix> x,
                    Eigen::Ref<Matrix> P,
                    const Eigen::Ref<const Matrix> F,
                    const Eigen::Ref<const Matrix> Q) {
                        
    x = F * x;
    P = F * P * F.transpose() + Q;
}

void kalman_update(Eigen::Ref<Matrix> x,
                   Eigen::Ref<Matrix> P,
                   const Eigen::Ref<const Matrix> z,
                   const Eigen::Ref<const Matrix> H,
                   const Eigen::Ref<const Matrix> R) {

    Matrix y = z - H * x;
    Matrix S = H * P * H.transpose() + R;
    Matrix K = P * H.transpose() * S.inverse();

    x = x + K * y;

    Matrix I = Matrix::Identity(P.rows(), P.cols());
    P = (I - K * H) * P;
}
