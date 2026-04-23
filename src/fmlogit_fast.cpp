// [[Rcpp::depends(RcppEigen, RcppNumerical)]]

#include <RcppNumerical.h>

#include <algorithm>
#include <cmath>
#include <string>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Numer::Constvec;
using Numer::MFuncGrad;
using Numer::Refvec;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  RowMajorMatrixXd;

class FMLogitObjective : public MFuncGrad {
public:
  FMLogitObjective(const Map<MatrixXd>& X, const Map<MatrixXd>& y)
      : X_(X),
        y_baseline_(y.col(0)),
        y_nonbaseline_(y.rightCols(y.cols() - 1)),
        n_(X.rows()),
        p_(X.cols()),
        j_minus_one_(y.cols() - 1) {}

  double f_grad(Constvec& betas, Refvec grad) {
    Map<const RowMajorMatrixXd> beta_mat(betas.data(), j_minus_one_, p_);
    MatrixXd eta = X_ * beta_mat.transpose();
    MatrixXd prob(n_, j_minus_one_);
    double neg_ll = 0.0;

    for (int i = 0; i < n_; ++i) {
      double max_eta = 0.0;
      if (j_minus_one_ > 0) {
        max_eta = std::max(0.0, eta.row(i).maxCoeff());
      }

      double sum_exp = std::exp(-max_eta);
      for (int m = 0; m < j_minus_one_; ++m) {
        sum_exp += std::exp(eta(i, m) - max_eta);
      }

      const double log_denom = max_eta + std::log(sum_exp);
      neg_ll += y_baseline_[i] * log_denom;

      for (int m = 0; m < j_minus_one_; ++m) {
        const double p_im = std::exp(eta(i, m) - log_denom);
        prob(i, m) = p_im;
        neg_ll += y_nonbaseline_(i, m) * (log_denom - eta(i, m));
      }
    }

    Map<RowMajorMatrixXd> grad_mat(grad.data(), j_minus_one_, p_);
    grad_mat.noalias() = (prob - y_nonbaseline_).transpose() * X_;

    return neg_ll;
  }

private:
  MatrixXd X_;
  VectorXd y_baseline_;
  MatrixXd y_nonbaseline_;
  int n_;
  int p_;
  int j_minus_one_;
};

static inline void validate_fmlogit_inputs(const Eigen::Map<Eigen::MatrixXd>& X,
                                           const Eigen::Map<Eigen::MatrixXd>& y,
                                           const Eigen::VectorXd& beta0) {
  if (X.rows() != y.rows()) {
    Rcpp::stop("X and y must have the same number of rows.");
  }
  if (y.cols() < 2) {
    Rcpp::stop("y must contain at least two choice columns.");
  }

  const int expected_params = (y.cols() - 1) * X.cols();
  if (beta0.size() != expected_params) {
    Rcpp::stop("beta0 has incorrect length.");
  }
}

// [[Rcpp::export]]
Rcpp::NumericVector fmlogit_obs_cpp(const Eigen::Map<Eigen::MatrixXd>& X,
                                    const Eigen::Map<Eigen::MatrixXd>& y,
                                    Eigen::VectorXd beta0) {
  validate_fmlogit_inputs(X, y, beta0);

  const int n = X.rows();
  const int p = X.cols();
  const int j_minus_one = y.cols() - 1;

  Rcpp::NumericVector ll(n);
  Eigen::Map<const RowMajorMatrixXd> beta_mat(beta0.data(), j_minus_one, p);
  MatrixXd eta = X * beta_mat.transpose();

  for (int i = 0; i < n; ++i) {
    double max_eta = 0.0;
    if (j_minus_one > 0) {
      max_eta = std::max(0.0, eta.row(i).maxCoeff());
    }

    double sum_exp = std::exp(-max_eta);
    for (int m = 0; m < j_minus_one; ++m) {
      sum_exp += std::exp(eta(i, m) - max_eta);
    }

    const double log_denom = max_eta + std::log(sum_exp);
    double ll_i = -y(i, 0) * log_denom;

    for (int m = 0; m < j_minus_one; ++m) {
      ll_i += y(i, m + 1) * (eta(i, m) - log_denom);
    }

    ll[i] = ll_i;
  }

  return ll;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix fmlogit_obs_grad_cpp(const Eigen::Map<Eigen::MatrixXd>& X,
                                         const Eigen::Map<Eigen::MatrixXd>& y,
                                         Eigen::VectorXd beta0) {
  validate_fmlogit_inputs(X, y, beta0);

  const int n = X.rows();
  const int p = X.cols();
  const int j_minus_one = y.cols() - 1;

  Rcpp::NumericMatrix grad(n, beta0.size());
  Eigen::Map<const RowMajorMatrixXd> beta_mat(beta0.data(), j_minus_one, p);
  MatrixXd eta = X * beta_mat.transpose();

  for (int i = 0; i < n; ++i) {
    double max_eta = 0.0;
    if (j_minus_one > 0) {
      max_eta = std::max(0.0, eta.row(i).maxCoeff());
    }

    double sum_exp = std::exp(-max_eta);
    for (int m = 0; m < j_minus_one; ++m) {
      sum_exp += std::exp(eta(i, m) - max_eta);
    }

    const double log_denom = max_eta + std::log(sum_exp);
    for (int m = 0; m < j_minus_one; ++m) {
      const double p_im = std::exp(eta(i, m) - log_denom);
      const double resid = y(i, m + 1) - p_im;
      const int offset = m * p;

      for (int r = 0; r < p; ++r) {
        grad(i, offset + r) = X(i, r) * resid;
      }
    }
  }

  return grad;
}

// [[Rcpp::export]]
Rcpp::List fmlogit_fast_cpp(const Eigen::Map<Eigen::MatrixXd>& X,
                            const Eigen::Map<Eigen::MatrixXd>& y,
                            Eigen::VectorXd beta0, int maxit, double abstol,
                            bool verbose = false, double eps_g = 1e-5) {
  validate_fmlogit_inputs(X, y, beta0);
  if (maxit <= 0) {
    Rcpp::stop("maxit must be a positive integer.");
  }
  if (!std::isfinite(abstol) || abstol < 0.0) {
    Rcpp::stop("abstol must be a non-negative finite value.");
  }
  if (!std::isfinite(eps_g) || eps_g < 0.0) {
    Rcpp::stop("eps_g must be a non-negative finite value.");
  }

  FMLogitObjective objective(X, y);
  VectorXd estimate = beta0;
  double fx_opt = NA_REAL;
  int status = 0;
  int niter = NA_INTEGER;
  std::string message = "Converged.";

  try {
    LBFGSpp::LBFGSParam<double> param;
    param.epsilon = eps_g;
    param.epsilon_rel = eps_g;
    param.past = 1;
    param.delta = abstol;
    param.max_iterations = maxit;
    param.max_linesearch = 100;
    param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;

    Numer::LBFGSFun fun(objective);
    LBFGSpp::LBFGSSolver<double> solver(param);
    niter = solver.minimize(fun, estimate, fx_opt);

    const double grad_norm = solver.final_grad_norm();
    const double x_norm = estimate.norm();
    const bool grad_converged =
      (grad_norm <= eps_g) || (grad_norm <= eps_g * x_norm);

    if (maxit > 0 && niter >= maxit && !grad_converged) {
      status = 1;
      message =
        "Reached the iteration limit before satisfying the gradient tolerance.";
    }

    if (verbose) {
      Rcpp::Rcout << "fmlogit_fast_cpp: status=" << status
                  << ", iterations=" << niter
                  << ", objective=" << fx_opt << std::endl;
    }
  } catch (const std::exception& e) {
    status = -1;
    message = e.what();
    if (verbose) {
      Rcpp::Rcout << "fmlogit_fast_cpp: status=" << status
                  << ", message=" << message << std::endl;
    }
  }

  return Rcpp::List::create(Rcpp::Named("estimate") = estimate,
                            Rcpp::Named("fx_opt") = fx_opt,
                            Rcpp::Named("status") = status,
                            Rcpp::Named("niter") = niter,
                            Rcpp::Named("message") = message);
}
