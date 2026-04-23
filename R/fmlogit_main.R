#' Estimate Fractional Multinomial Logit Models
#' 
#' Used to estimate fractional multinomial logit models using quasi-maximum
#' likelihood estimations following Papke and Wooldridge(1996).
#' @param y the dependent variable (N*J). Can be a matrix or a named data frame.
#'   The first column of the matrix is automatically treated as the baseline.
#' @param X independent variable (N*K). Can be a matrix or a named data frame.
#'   If there is no intercept term in the X, then an intercept term is
#'   automatically added.
#' @param beta0 Initial value for beta used in optimization. Uses a 1*K(J-1)
#'   vector. Default to a vector of zeros.
#' @param MLEmethod Legacy alias for the \code{maxLik} optimization method used
#'   when \code{engine = "maxLik"}. Choose from "NR", "BFGS", "CG", "BHHH",
#'   "SANN", or "NM". Default to "CG". See Details.
#' @param maxit Maximum number of iterations. Used as the default
#'   \code{iterlim} in \code{engine_control}.
#' @param abstol Tolerence. Used as the default \code{tol} in
#'   \code{engine_control}.
#' @param cluster A vector of cluster to be used for clustered standard error computation. 
#' Default to NULL, no cluster computed. 
#' @param reps Numbers of bootstrap replications to be computed for clustered standard errors.
#' @param engine Estimation engine. Defaults to the compiled
#'   \code{RcppNumerical} L-BFGS implementation. Use \code{"maxLik"} to keep
#'   the legacy \code{maxLik} optimizer path.
#' @param fallback_method Optimization method passed to \code{maxLik()} when
#'   \code{engine = "maxLik"}. Defaults to \code{"CG"}. If omitted, the
#'   existing \code{MLEmethod} argument is still accepted for backward
#'   compatibility.
#' @param engine_control Optional list of engine controls. Defaults to
#'   \code{list(iterlim = maxit, tol = abstol)}. The compiled engine also
#'   accepts \code{eps_g}, and \code{fallback_on_failure = TRUE} can be used to
#'   retry with \code{maxLik} after a non-zero compiled status.
#' @param engine_verbose Logical flag controlling whether the compiled engine
#'   prints optimizer summaries.
#' @param ... additional parameters that goes into \code{maxLik()} when
#'   \code{engine = "maxLik"}
#' @return The function returns an object of class "fmlogit". Use \code{effects}, \code{predict}, 
#'  \code{residual}, \code{fitted} to extract various useful features of the value returned by 
#' \code{fmlogit}. 
#' @return An object of class "fmlogit" contains the following components: 
#' @return \code{estimates}   A list of matrices containing parameter estimates,
#'   standard errors, and hypothesis testing results.
#' @return \code{baseline}    The baseline choice
#' @return \code{likelihood}  The likelihood value
#' @return \code{conv_code}   Convergence diagnostics code. 
#' @return \code{convergence} Convergence messages. 
#' @return \code{count}       Provides dataset information
#' @return \code{y}           The dependent variable data frame.
#' @return \code{X}           The independent variable data frame. Augmented by factor dummy transformation
#' , constant term added. 
#' @return \code{rowNo}       A vector of row numbers from the original X and y that is used for estimation.
#' @return \code{coefficient} Matrix of estimated coefficients. Augmented with the baseline coefficient
#' (which is a vector of zeros). 
#' @return \code{vcov}        A list of matrices containing the robust variance covariance matrix for each choice
#' variable. 
#' @return \code{cluster}     The vector of clusters. 
#' @return \code{reps}        Number of bootstrap replications for clustered standard error
#' @details The fractional multinomial model is the expansion of the multinomial
#' logit to fractional responses. Unlike standard multinomial logit models,
#' which only considers 0-1 respones, fractional multinomial model considers the
#' case where the response variable is fractions that sums up to one. Examples
#' of these type of data are, percentages of budget spent in education, defense,
#' public health; fractions of a population that have middle school, high
#' school, college, or post college education, etc.
#' 
#' This function follows Papke and Wooldridge(1996)'s paper, in which they
#' proposed a quasi-maximum likelihood estimator for fractional response data.
#' The likelihood function used here is a standard multinomial likelihood
#' function, see \url{http://maartenbuis.nl/software/likelihoodFmlogit.pdf} for
#' the likelihood used here. Robust standard errors are provided following Papke
#' and Wooldridge(1996), in which they proposed an asymptotically consistent
#' estimator of variance.
#' 
#' By default, optimization uses a compiled \code{RcppNumerical} L-BFGS engine
#' for faster estimation. Setting \code{engine = "maxLik"} keeps the legacy
#' \code{\link{maxLik}} path, and \code{fallback_method} (or the legacy
#' \code{MLEmethod} argument) controls which \code{maxLik} routine is used.
#' 
#' MLE convergence can still be a problem if the dataset is large with many
#' explanatory variables. For the \code{maxLik} engine, it is recommended to
#' call CG (Conjugate Gradients) or BHHH (Berndt-Hall-Hall-Hausman). Conjugate
#' gradients are usually faster, but could lead to non-convergence under
#' certain scenarios. BHHH is slower, but has better convergence performance.
#' 
#' 
#' @examples 
#' data = spending
#' X = data[,2:5]
#' y = data[,6:11]
#' results1 = fmlogit(y,X)
#' @references Papke, L. E. and Wooldridge, J. M. (1996), Econometric methods
#'   for fractional response variables with an application to 401(k) plan
#'   participation rates. J. Appl. Econ., 11: 619-632.
#' @export fmlogit




fmlogit=function(y, X, beta0 = NULL, MLEmethod = "CG", maxit = 5e+05, 
                          abstol = 1e-05,cluster=NULL,reps=1000,
                          engine = c("RcppNumerical", "maxLik"),
                          fallback_method = "CG",
                          engine_control = list(iterlim = maxit, tol = abstol),
                          engine_verbose = FALSE, ...){
  start.time = proc.time()
  
  if(length(cluster)!=nrow(y) & !is.null(cluster)){
    warning("Length of the cluster does not match the data. Cluster is ignored.")
    cluster = NULL
  }
  Xclass = sapply(X, class)
  Xfac = which(Xclass %in% c("factor", "character"))
  if (length(Xfac) > 0) {
    Xfacnames = colnames(X)[Xfac]
    strformFac = paste(Xfacnames, collapse = "+")
    Xdum = model.matrix(as.formula(paste("~", strformFac, 
                                         sep = "")), data = X)[, -1]
    X = cbind(X, Xdum)
    X = X[, -Xfac]
  }
  Xnames = colnames(X)
  ynames = colnames(y)
  X = as.matrix(X)
  y = as.matrix(y)
  n = dim(X)[1]
  j = dim(y)[2]
  k = dim(X)[2]
  xy = cbind(X, y)
  xy = na.omit(xy)
  row.remain = setdiff(1:n, attr(xy, "na.action"))
  X = xy[, 1:k]
  y = xy[, (k + 1):(k + j)]
  n = dim(y)[1]
  remove(xy)
  # adding in the constant term
  if(k==1){
    # check if the input X is constant
    if(length(unique(X))==1){ # X is constant
      Xnames = "constant"
      X = as.matrix(as.numeric(X),nrow=1)
      colnames(X) = Xnames
      X = as.matrix(X)
      k=0
    }else{ # one single variable of input
      Xnames = "X1"
      X = as.matrix(X)
      k = dim(X)[2]
      X = cbind(X, rep(1, n))
      Xnames = c(Xnames, "constant")
      colnames(X) = Xnames
    }
  }else{ # normal cases
    X = X[, apply(X, 2, function(x) length(unique(x)) != 1)]
    Xnames = colnames(X)
    k = dim(X)[2]
    X = cbind(X, rep(1, n))
    Xnames = c(Xnames, "constant")
    colnames(X) = Xnames
  }
  
  
  testcols <- function(X) {
    m = crossprod(as.matrix(X))
    ee = eigen(m)
    evecs <- split(zapsmall(ee$vectors), col(ee$vectors))
    mapply(function(val, vec) {
      if (val != 0) 
        NULL
      else which(vec != 0)
    }, zapsmall(ee$values), evecs)
  }
  collinear = unique(unlist(testcols(X)))
  while (length(collinear) > 0) {
    if (qr(X)$rank == dim(X)[2]) 
      print("Model may suffer from multicollinearity problems.")
    break
    if ((k + 1) %in% collinear) 
      collinear = collinear[-length(collinear)]
    X = X[, -collinear[length(collinear)]]
    Xnames = colnames(X)
    k = k - 1
    collinear = unique(unlist(testcols(X)))
  }
  engine = match.arg(engine)
  if (!is.list(engine_control)) 
    stop("'engine_control' must be a list.")
  if (!is.logical(engine_verbose) || length(engine_verbose) != 1 || 
      is.na(engine_verbose)) 
    stop("'engine_verbose' must be TRUE or FALSE.")
  if (!missing(fallback_method) && !missing(MLEmethod) && 
      !identical(fallback_method, MLEmethod)) 
    warning("Both fallback_method and MLEmethod were supplied. Using fallback_method for engine = \"maxLik\".")
  maxLik_method = if (missing(fallback_method)) 
    MLEmethod
  else fallback_method
  if (!is.character(maxLik_method) || length(maxLik_method) != 1 || 
      is.na(maxLik_method)) 
    stop("'fallback_method' must be a single character string.")
  engine_control = utils::modifyList(list(iterlim = maxit, tol = abstol), 
                                     engine_control)
  if (!is.numeric(engine_control$iterlim) || length(engine_control$iterlim) != 1 || 
      !is.finite(engine_control$iterlim) || engine_control$iterlim < 1) 
    stop("'engine_control$iterlim' must be a positive, finite scalar.")
  if (!is.numeric(engine_control$tol) || length(engine_control$tol) != 1 || 
      !is.finite(engine_control$tol) || 
      engine_control$tol < 0) 
    stop("'engine_control$tol' must be a non-negative, finite scalar.")
  engine_maxit = as.integer(engine_control$iterlim)
  engine_tol = as.numeric(engine_control$tol)
  engine_eps_g = if (is.null(engine_control$eps_g)) 
    1e-05
  else as.numeric(engine_control$eps_g)
  if (!is.numeric(engine_eps_g) || length(engine_eps_g) != 1 || 
      !is.finite(engine_eps_g) || 
      engine_eps_g < 0) 
    stop("'engine_control$eps_g' must be a non-negative, finite scalar.")
  fallback_on_failure = isTRUE(engine_control$fallback_on_failure)
  maxLik_control = engine_control
  maxLik_control$eps_g = NULL
  maxLik_control$fallback_on_failure = NULL
  QMLE_Obs <- function(betas) {
    betas = matrix(betas, nrow = j - 1, byrow = T)
    betamat = rbind(rep(0, k + 1), betas)
    llf = rep(0, n)
    for (i in 1:j) {
      L = y[, i] * ((X %*% betamat[i, ]) - log(rowSums(exp(X %*% 
                                                             t(betamat)))))
      llf = llf + L
    }
    return(llf)
  }
  if (length(beta0) == 0){
    beta0 = rep(0, (k + 1) * (j - 1))
  }
  if (length(beta0) != (k + 1) * (j - 1)) {
    beta0 = rep(0, (k + 1) * (j - 1))
    warning("Wrong length of beta0 given. Use default setting instead.")
  }
  run_maxLik = function() {
    maxLik(QMLE_Obs, start = beta0, method = maxLik_method, 
           control = maxLik_control, ...)
  }
  run_RcppNumerical = function() {
    fast_opt = tryCatch(fmlogit_fast_cpp(X = X, y = y, beta0 = beta0, 
                                         maxit = engine_maxit, 
                                         abstol = engine_tol, 
                                         verbose = engine_verbose, 
                                         eps_g = engine_eps_g), 
                        error = function(e) {
                          warning(paste("RcppNumerical engine failed:", 
                                        conditionMessage(e), 
                                        "Falling back to maxLik."))
                          run_maxLik()
                        })
    if (inherits(fast_opt, "maxLik")) 
      return(fast_opt)
    if (!is.null(fast_opt$status) && fast_opt$status < 0) {
      warning("RcppNumerical failed inside the compiled optimizer. Falling back to maxLik.")
      return(run_maxLik())
    }
    if (!is.null(fast_opt$status) && fast_opt$status != 0 && fallback_on_failure) {
      warning("RcppNumerical returned a non-zero status. Falling back to maxLik.")
      return(run_maxLik())
    }
    list(estimate = as.numeric(fast_opt$estimate), maximum = -fast_opt$fx_opt, 
         code = fast_opt$status, type = "RcppNumerical L-BFGS", 
         iterations = fast_opt$niter, message = fast_opt$message)
  }
  opt = if (engine == "RcppNumerical") 
    run_RcppNumerical()
  else run_maxLik()
  betamat = matrix(opt$estimate, nrow = j - 1, byrow = T)
  betamat_aug = rbind(rep(0, k + 1), betamat)
  colnames(betamat_aug) = Xnames
  rownames(betamat_aug) = ynames
  sigmat = matrix(nrow = j - 1, ncol = k + 1)
  vcov = list()
  
  ###insert--nonparametric bootstrap procedure (clustered SE and vcov)
  
  if(is.null(cluster)==F){
    cluster = cluster[row.remain]
    clusters <- names(table(cluster))
    for (i in 1:j) {
      # cluster should preferably be coming from a same data frame with the original y and X. 
      sterrs <- matrix(NA, nrow=reps, ncol=k + 1)
      vcov_j_list=list()
      
      b=1
      no_singular_error=c()
      while(b<=reps){
        
        index <- sample(1:length(clusters), length(clusters), replace=TRUE)
        aa <- clusters[index]
        bb <- table(aa)
        bootdat <- NULL
        dat=cbind(y,X)
        for(b1 in 1:max(bb)){
          cc <- dat[cluster %in% names(bb[bb %in% b1]),]
          for(b2 in 1:b1){
            bootdat <- rbind(bootdat, cc)
          }
        }
        
        bootdatX=matrix(bootdat[,(j+1):ncol(bootdat)],nrow=nrow(bootdat))
        bootdaty=bootdat[,1:j]
        
        sum_expxb = rowSums(exp(bootdatX %*% t(betamat_aug)))
        expxb = exp(bootdatX %*% betamat_aug[i, ])
        G = expxb/sum_expxb
        g = (expxb * sum_expxb - expxb^2)/sum_expxb^2
        X_a = bootdatX * as.vector(sqrt(g^2/(G * (1 - G))))
        A = t(X_a) %*% X_a
        mu = bootdaty[, i] - G
        X_b = bootdatX * as.vector(mu * g/G/(1 - G))
        B = t(X_b) %*% X_b
        
        a_solve_error = tryCatch(solve(A),error=function(e){NULL})
        if(is.null(a_solve_error)){
          no_singular_error=c(no_singular_error,b)
          next
        }
        
        Var_b = solve(A) %*% B %*% solve(A)
        std_b = sqrt(diag(Var_b))
        sterrs[b,]=std_b
        vcov_j_list[[b]]=Var_b
        
        b=b+1
      }
      if(length(no_singular_error)>0){warning(paste('Error in solve.default(A) : Lapack routine dgesv: system is exactly singular: U[28,28] = 0" Appeared',length(no_singular_error),'times within cluster bootstrap for outcome #',i))}
      std_b=apply(sterrs,2,mean)
      vcov[[i]] = Reduce("+", vcov_j_list) / length(vcov_j_list)
      if (i > 1) 
        sigmat[i - 1, ] = std_b
    }
  }else{
    for(i in 1:j){
      # start calculation  
      sum_expxb = rowSums(exp(X %*% t(betamat_aug))) # sum of the exp(x'b)s
      expxb = exp(X %*% betamat_aug[i,]) # individual exp(x'b)
      G = expxb / sum_expxb # exp(X'bj) / sum^J(exp(X'bj))
      g = (expxb * sum_expxb - expxb^2) / sum_expxb^2 # derivative of the logit function
      
      # Here the diagonal of A is the 'standard' standard error
      # hat(A) = sum hat(gi)^2 * xi'xi / hat(Gi)(1-hat(Gi))
      # or, Xtilde = X * sqrt(g^2/G(1-G)), A = Xtilde'Xtilde
      X_a = X * as.vector(sqrt(g^2/(G*(1-G))))
      A = t(X_a) %*% X_a
      
      # robust standard error, again following PW(1996)
      mu = y[,i] - G
      X_b = X * as.vector(mu * g / G / (1-G))
      B = t(X_b) %*% X_b
      Var_b = solve(A) %*% B %*% solve(A)
      std_b = sqrt(diag(Var_b))
      # std_b= sqrt(diag(solve(A))) is the "unrobust" standard error. 
      vcov[[i]] = Var_b
      if(i>1) sigmat[i-1,] = std_b
    }
  }
  
  ###end of insert--nonparametric bootstrap procedure (clustered SE and vcov)
  
  listmat = list()
  for (i in 1:(j - 1)) {
    tabout = matrix(ncol = 4, nrow = k + 1)
    tabout[, 1:2] = t(rbind(betamat[i, ], sigmat[i, ]))
    tabout[, 3] = tabout[, 1]/tabout[, 2]
    tabout[, 4] = 2 * (1 - pnorm(abs(tabout[, 3])))
    colnames(tabout) = c("estimate", "std", "z", "p-value")
    if (length(Xnames) > 0) 
      rownames(tabout) = Xnames
    listmat[[i]] = tabout
  }
  if (length(ynames) > 0) 
    names(listmat) = ynames[2:j]
  outlist = list()
  outlist$estimates = listmat
  outlist$baseline = ynames[1]
  outlist$likelihood = opt$maximum
  outlist$conv_code = opt$code
  outlist$convergence = paste(opt$type, paste(as.character(opt$iterations), 
                                              "iterations"), opt$message, sep = ",")
  outlist$count = c(Obs = n, Explanatories = k, Choices = j)
  outlist$y = y
  outlist$X = X
  outlist$rowNo = row.remain
  outlist$coefficient = betamat_aug
  names(vcov) = ynames
  outlist$vcov = vcov
  outlist$cluster = cluster
  outlist$reps=ifelse(is.null(cluster),0,reps)
  
  print(paste("Fractional logit model estimation completed. Time:", 
              round(proc.time()[3] - start.time[3], 1), "seconds"))
  return(structure(outlist, class = "fmlogit"))
}



