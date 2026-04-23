#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fmlogit)
  library(maxLik)
  library(microbenchmark)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg_value <- function(flag, default = NULL) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (length(hit) == 0) {
    return(default)
  }
  sub(paste0("^", flag, "="), "", hit[[1]])
}

parse_int_arg <- function(flag, default = NULL) {
  value <- get_arg_value(flag, default)
  if (is.null(value) || identical(value, "all") || identical(value, "")) {
    return(NULL)
  }
  as.integer(value)
}

parse_num_arg <- function(flag, default) {
  as.numeric(get_arg_value(flag, default))
}

n_rows <- parse_int_arg("--rows", NULL)
times <- parse_int_arg("--times", 5L)
method <- get_arg_value("--method", "CG")
maxit <- parse_int_arg("--maxit", 2000L)
tol <- parse_num_arg("--tol", 1e-8)
eps_g <- parse_num_arg("--eps-g", 1e-8)

prepare_inputs <- function(n_rows) {
  data(spending, package = "fmlogit")

  X <- spending[, 2:5, drop = FALSE]
  y <- spending[, 6:11, drop = FALSE]
  keep <- complete.cases(cbind(X, y))
  X <- as.matrix(X[keep, , drop = FALSE])
  y <- as.matrix(y[keep, , drop = FALSE])

  if (!is.null(n_rows)) {
    n_rows <- min(as.integer(n_rows), nrow(X))
    X <- X[seq_len(n_rows), , drop = FALSE]
    y <- y[seq_len(n_rows), , drop = FALSE]
  }

  X <- X[, apply(X, 2, function(x) length(unique(x)) != 1), drop = FALSE]
  X <- cbind(X, constant = 1)

  list(
    X = X,
    y = y,
    beta0 = rep(0, ncol(X) * (ncol(y) - 1))
  )
}

inputs <- prepare_inputs(n_rows)
X <- inputs$X
y <- inputs$y
beta0 <- inputs$beta0

QMLE_Obs_R <- function(betas) {
  j <- ncol(y)
  k1 <- ncol(X)
  llf <- rep(0, nrow(X))
  betas <- matrix(betas, nrow = j - 1, byrow = TRUE)
  betamat <- rbind(rep(0, k1), betas)

  for (i in seq_len(j)) {
    L <- y[, i] * ((X %*% betamat[i, ]) - log(rowSums(exp(X %*% t(betamat)))))
    llf <- llf + L
  }

  llf
}

QMLE_Obs_CPP <- function(betas) {
  getFromNamespace("fmlogit_obs_cpp", "fmlogit")(X, y, betas)
}

QMLE_Grad_CPP <- function(betas) {
  getFromNamespace("fmlogit_obs_grad_cpp", "fmlogit")(X, y, betas)
}

run_r_maxLik <- function() {
  maxLik(
    logLik = QMLE_Obs_R,
    start = beta0,
    method = method,
    control = list(iterlim = maxit, tol = tol)
  )
}

run_lbfgs_cpp <- function() {
  getFromNamespace("fmlogit_fast_cpp", "fmlogit")(
    X = X,
    y = y,
    beta0 = beta0,
    maxit = maxit,
    abstol = tol,
    verbose = FALSE,
    eps_g = eps_g
  )
}

run_cpp_maxLik <- function() {
  maxLik(
    logLik = QMLE_Obs_CPP,
    grad = QMLE_Grad_CPP,
    start = beta0,
    method = method,
    control = list(iterlim = maxit, tol = tol)
  )
}

baseline_fit <- run_r_maxLik()
lbfgs_fit <- run_lbfgs_cpp()
cpp_maxlik_fit <- run_cpp_maxLik()

gc(FALSE)

bench <- microbenchmark(
  baseline_r_maxLik = run_r_maxLik(),
  cpp_lbfgs = run_lbfgs_cpp(),
  cpp_maxLik = run_cpp_maxLik(),
  times = times,
  unit = "ms"
)

bench_summary <- summary(bench)[, c("expr", "min", "lq", "mean", "median", "uq", "max")]
baseline_median <- bench_summary$median[bench_summary$expr == "baseline_r_maxLik"]
bench_summary$speedup_vs_baseline <- baseline_median / bench_summary$median

cat("Benchmark configuration\n")
cat("-----------------------\n")
cat("Rows:", nrow(X), "\n")
cat("Choices:", ncol(y), "\n")
cat("Parameters:", length(beta0), "\n")
cat("maxLik method:", method, "\n")
cat("maxit:", maxit, "\n")
cat("tol:", format(tol, scientific = TRUE), "\n")
cat("times:", times, "\n\n")

cat("Fit consistency check\n")
cat("---------------------\n")
cat("baseline likelihood:", format(baseline_fit$maximum, digits = 10), "\n")
cat("cpp_lbfgs likelihood:", format(-lbfgs_fit$fx_opt, digits = 10), "\n")
cat("cpp_maxLik likelihood:", format(cpp_maxlik_fit$maximum, digits = 10), "\n")
cat(
  "max |coef baseline - cpp_lbfgs|:",
  format(max(abs(baseline_fit$estimate - lbfgs_fit$estimate)), digits = 6),
  "\n"
)
cat(
  "max |coef baseline - cpp_maxLik|:",
  format(max(abs(baseline_fit$estimate - cpp_maxlik_fit$estimate)), digits = 6),
  "\n\n"
)

cat("microbenchmark summary (ms)\n")
cat("---------------------------\n")
print(bench_summary, row.names = FALSE)
