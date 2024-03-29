#' @title Learning DTR from Sequential Interventions (Continuous Treatment)
#' @author Junyi Zhou \email{junyzhou@iu.edu}
#' @description This function supports to learn the optimal sequential decision rules from either randomized studies
#'              or observational ones. Multiple treatment options are supported.
#' @param X A list of information available at each stage in order, that is, \code{X[[1]]} represents the baseline information,
#'          and \code{X[[t]]} is the information observed before \eqn{t^{th}} intervention.
#'          The dimensionality of each element \code{X[[i]]} can be different from each other.
#'          Notably, it can includes previous stages' action information \code{A} and outcome/reward information
#'          \code{Y}. User can flexibly manipulate which covariates to use in training.
#'          However, if argument \code{all.inclusive} is \code{TRUE}, all previous stages' \code{X}, \code{A},
#'          and \code{Y} will be used in training. So, in that case, \code{X} should not involve action and reward
#'          information.
#' @param A A list of actions taken during the sequential studies. The order should match with that of \code{X}
#' @param Y A list of outcomes observed in the sequential studies. The order should match with that of \code{X}.
#'          \code{Y[[t]]} is suppose to be driven by the \code{X[[t]]} and action \code{A[[t]]}.
#' @param weights Weights on each stage of rewards. Default is all 1.
#' @param baseLearner Choose one baselearner for meta-learn er algorithms. So far, it supports \code{BART} by
#'                    package \code{dbarts}, \code{RF} (random forests) by \code{ranger}, \code{XGBoost} by
#'                    package \code{xgboost}, and \code{GAM} (generalized additive model)
#'                    through package \code{glmnet}, which can provide variable selection/sparsity by
#'                    various type of regularization. So more in details.
#' @param metaLearners \code{c("S")}. For continuous treatment assignment, only S-learner type is available.
#' @param include.X 0 for no past X included in analysis; 1 for all past X included
#' @param include.A 0 for no past treatment assignment included in analysis; 1 for only last A included; 2 for all past
#'                  A included
#' @param include.Y 0 for no past reward/outcome Y included in analysis; 1 for only last Y included; 2 for all past
#'                  Y included
#' @param parallel A boolean, for whether parallel computing is adopted to speed up the "prediction" steps.
#'                 It might induce certain connection issue with \code{glmnet}. But parallel is not really
#'                 necessary. It does not impact training speed. Also, if a numeric value, it implies the
#'                 number of cores to use. Otherwise, directly use the number from `detectCores()`.
#' @param parallel.package One of c("doMC", "snow", "doParallel"), the parallel package to use. Only \code{BART}
#'                         seems matches better with \code{snow}, while the others, like \code{XGBoost},
#'                         \code{RF}, and \code{GAM} prefers \code{doMC}. So please check which parallel structure
#'                         fits the best on the chosen algorithm before running heavy duty.
#' @param A.box.cnstr Box constraint for A. Default is \code{NULL}, which means no constraint on A
#'                    and the feasible set of A will be implied from observed action values
#' @param A.cnstr.func For other type of constraint applied to A, user can input a function here.
#'                     The basic syntax is \code{function(A, X){...}}, with two arguments: A stands
#'                     for action and X are covariates at that stage. For which covariates to select,
#'                     use the argument \code{x.select}. The function should return a boolean,
#'                     i.e., \code{TRUE} or \code{FALSE}. For example, if constraint is \eqn{3A-log(A)<5},
#'                     then function is \code{function(A, X){3A-log(A)<5}}.
#' @param x.select Covariate names or id (numeric). Default is \code{NULL} means either all variables are
#'                 selected (if X is used in function \code{A.cnstr.func}) or X is not related to A's feasible set
#' @param n.grid Number of grid used to search for the best treatment/action. Large values slow down the algorithm.
#' @param verbose Console print allowed?
#' @param verbose.parallel Verbose specifically for parallel computing. May slow down the computation
#' @param ... Additional arguments that can be passed to \code{dbarts::bart}, \code{ranger::ranger},
#'            \code{params} of \code{xbgoost::xgb.cv}, or \code{glmnet::cv.glmnet}
#' @details This function supports to find the optimal dynamic treatment regime (DTR) for either randomized experiments
#'          or observational studies. Also, thanks to meta-learner structure, S-, T-, and deC-learner can naturally
#'          support multiple action options at any stage. \cr
#'          \cr
#'          For \code{GAM}, the algorithm will not automatically project the covariates \code{X} or outcomes/rewards
#'          \code{Y} onto any bases-spanned spaces. User shall transform the covariates and/or  outcomes/rewards
#'          manually and then input the desired design matrix through inputs \code{X} and/or \code{Y}.\cr
#'          \cr
#'          It is strongly suggested to adopt BART over random forests as baselearner if sample size is small.
#'
#' @return It includes learning results, basically, the trained functions that can be used for predictions. Since
#'         sequential recommendations might require intermediate observations, [learnDTR.cont] will not automatically
#'         provide prediction. But by another function [recommendDTR.cont],
#'         predictions can flexibly been made stage by stage.
#' @examples
#' ## Though ThreeStg_Dat does not have continuous treatment, still we can use it as an example
#' DTRs = learnDTR.cont(X = ThreeStg_Dat$X,
#'                      A = ThreeStg_Dat$A,
#'                      Y = ThreeStg_Dat$Y,
#'                      weights = rep(1, 3),
#'                      baseLearner  = c("BART"),
#'                      metaLearners = c("S"),
#'                      include.X = 1,
#'                      include.A = 2,
#'                      include.Y = 0)
#' @import stats utils dbarts glmnet ranger xgboost pbapply doParallel snow foreach
#' @references
#' Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu.
#' "Metalearners for estimating heterogeneous treatment effects using machine learning."
#' *Proceedings of the national academy of sciences* 116, no. 10 (2019): 4156-4165.
#' \cr\cr
#' Zhou, Junyi, Ying Zhang, and Wanzhu Tu. "A reference-free R-learner for treatment recommendation."
#' *Statistical Methods in Medical Research* (2022)
#' @export
#' @seealso \code{\link{recommendDTR.cont}}

learnDTR.cont <- function(X, A, Y, weights = rep(1, length(X)),
                          baseLearner  = c("RF", "BART", "XGBoost", "GAM"),
                          metaLearners = c("S"),
                          include.X = 0, include.A = 0, include.Y = 0,
                          parallel = FALSE,
                          parallel.package = "doMC",
                          A.box.cnstr = NULL,
                          A.cnstr.func = NULL,
                          x.select = NULL,
                          n.grid = 100,
                          verbose = TRUE,
                          verbose.parallel = FALSE,
                          ...
) {
  n.stage = length(X)
  ######==================== check inputs ====================
  if (length(unique(length(X), length(A), length(Y))) != 1){
    stop("Inconsistent input of X, A, and Y. Please check!\n")
  }



  A <- lapply(A, as.numeric)
  if (is.null(A.box.cnstr)) {
    A.list <- lapply(A, function(x) range(x))
  } else {
    if (is.list(A.box.cnstr)) {
      if (length(A.box.cnstr)!=length(X)) {
        stop("A.box.cnstr does not have proper number of list elements!")
      }
    } else {
      A.list = rep(list(A.box.cnstr), length(X))
    }
  }



  if (parallel == FALSE) {

  } else {
    if (verbose.parallel) {
      # Progress combine function
      f <- function(iterator){
        pb <- txtProgressBar(min = 1, max = iterator - 1, style = 3)
        count <- 0
        function(...) {
          count <<- count + length(list(...)) - 1
          setTxtProgressBar(pb, count)
          flush.console()
          cbind(...) # this can feed into .combine option of foreach
        }
      }
    } else {
      f <- function(x){function(...) cbind(...)}
    }
  }


  # a simply defined function to reflect verbose requirement
  if (verbose) {
    my.sapply <- function(...){pbsapply(...)}
  } else {
    my.sapply <- function(...){sapply(...)}
  }


  ######==================== clean arguments in ... ====================
  if (baseLearner[1] == "XGBoost") {
    params = list(...)
    params.list = params[!names(params) %in% c(names(as.list(args(xgb.cv))), names(as.list(args(xgb.train))))]
    params.cv.tr = params[names(params) %in% c(names(as.list(args(xgb.cv))), names(as.list(args(xgb.train))))]
    # default settings for XGBoost
    params.default <- list(booster = "gbtree", objective = "reg:squarederror",
                           eta=0.2, gamma=0, max_depth=5,
                           min_child_weight=1, subsample=2/(sqrt(5)+1), colsample_bytree=2/(sqrt(5)+1),
                           lambda = 0, alpha = 1)
    params.list = c(params.list, params.default[names(params.default)[!names(params.default) %in% names(params.list)]])
  }




  S.learners <- T.learners <- deC.learners <- NULL
  ######==================== bart/xgboost as baselearner ====================
  if (baseLearner[1] %in% c("BART", "XGBoost")) {
    ######============  S-learner  ============
    if ("S" %in% metaLearners) {
      V.est = 0; S.learners = list()
      for (stage in seq(n.stage, by = -1)) {
        if (verbose) {
          print(paste0(Sys.time(), " @Stage(", baseLearner[1], "/S-): ", stage))
        }

        n.train = nrow(X[[stage]])
        ## construct training dataset
        if (stage == 1) {
          X.tr = data.frame(X = X[[stage]])
        } else {
          ## note the sequence of X, A, Y
          if (include.X == 1) {
            X.tr = data.frame(X = matrix(unlist(X[1:stage]), nrow = n.train, byrow = FALSE))
          } else {
            X.tr = data.frame(X = X[[stage]])
          }
          if (include.A == 1) { # nothing need to do when include.A == 0
            A.tmp = data.frame(A = A[[stage-1]])
            X.tr = cbind(X.tr, A.tmp)
          } else if (include.A == 2) {
            A.tmp = do.call(cbind.data.frame, A[1:(stage-1)])
            colnames(A.tmp) <- paste("A", 1:(stage-1), sep = ".")
            X.tr = cbind(X.tr, A.tmp)
          }
          if (include.Y == 1) { # nothing need to do when include.Y == 0
            Y.tmp = data.frame(Y = Y[[stage-1]]); colnames(Y.tmp) <- "y"
            X.tr = cbind(X.tr, Y.tmp)
          } else if (include.Y == 2) {
            Y.tmp = data.frame(Y = matrix(unlist(Y[1:(stage-1)]), ncol = stage-1, byrow = FALSE)); colnames(Y.tmp) <- paste("y", seq(ncol(Y.tmp)),sep = "_")
            X.tr = cbind(X.tr, Y.tmp)
          }
        }

        A.tr = A[[stage]]
        Y.tr = Y[[stage]]
        A.feasible = seq(min(A.list[[stage]]), max(A.list[[stage]]), length.out = n.grid)
        V.est = V.est + Y.tr*weights[stage]
        ## generate data set for S-learner:
        dat.tmp = data.frame(trt = A.tr, X.tr)
        store.names = colnames(dat.tmp)
        ## train by bart/xgboost:
        if (baseLearner[1] == "BART") {
          S.fit = bart(x.train = stats::model.matrix(~trt*.-1, dat.tmp), y.train = V.est, keeptrainfits = F,
                       keeptrees = TRUE, verbose = FALSE, ...)
          remove(dat.tmp)

          if (verbose) {
            print(paste0(Sys.time(), ": Fitting part done!! Next is doing some estimation jobs for next stage."))
          }

          if (stage>1) {
            if (parallel == FALSE) {
              S.est = my.sapply(1:nrow(X.tr), function(i) {
                # find proper A.range if there is any constraint on A given X/Y
                if (is.function(A.cnstr.func)) {
                  if (is.null(x.select)) {
                    A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i,]) )])
                  } else {
                    A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i, x.select]) )])
                  }
                } else {
                  A.range = A.feasible
                }

                dat.tmp = data.frame(trt = A.range, outer(rep(1, length(A.range)), as.numeric(X.tr[i,]))); colnames(dat.tmp) <- store.names
                return( max(colMeans(predict(S.fit, newdata = stats::model.matrix(~trt*.-1, dat.tmp)))) )
              })
            } else {
              ## register cores
              if (is.numeric(parallel)) {
                # Start a cluster
                if (parallel.package[1] == "doMC"){
                  doMC::registerDoMC(parallel)
                }
                if (parallel.package[1] == "doParallel") {
                  doParallel::registerDoParallel(parallel)
                }
                if (parallel.package[1] == "snow") {
                  cl <- makeCluster(parallel, type='SOCK')
                  registerDoParallel(cl)
                }
              } else {
                # Start a cluster
                if (parallel.package[1] == "doMC"){
                  doMC::registerDoMC(detectCores())
                }
                if (parallel.package[1] == "doParallel") {
                  doParallel::registerDoParallel(detectCores())
                }
                if (parallel.package[1] == "snow") {
                  cl <- makeCluster(detectCores(), type='SOCK')
                  registerDoParallel(cl)
                }
              }

              S.est = foreach(i=1:nrow(X.tr), .packages=c("dbarts"), .combine = f(nrow(X.tr))) %dopar% {
                # find proper A.range if there is any constraint on A given X/Y
                if (is.function(A.cnstr.func)) {
                  if (is.null(x.select)) {
                    A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i,]) )])
                  } else {
                    A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i, x.select]) )])
                  }
                } else {
                  A.range = A.feasible
                }

                dat.tmp = data.frame(trt = A.range, outer(rep(1, length(A.range)), as.numeric(X.tr[i,]))); colnames(dat.tmp) <- store.names
                return( max(colMeans(predict(S.fit, newdata = stats::model.matrix(~trt*.-1, dat.tmp)))) )
              } # should be n.grid x nrow(X.tr)

              ## Stop the cluster
              if (parallel.package[1] == "snow") {stopCluster(cl)}
            }
          }
        }
        if (baseLearner[1] == "XGBoost") {
          dtrain <- xgb.DMatrix(data = stats::model.matrix(~trt*.-1, dat.tmp), label = V.est)
          remove(dat.tmp)
          xgbcv <- xgb.cv( params = params.list, data = dtrain,
                           nrounds = ifelse("nrounds" %in% names(params.cv.tr), params.cv.tr[['nrounds']], 500),
                           nfold = ifelse("nfold" %in% names(params.cv.tr), params.cv.tr[['nfold']], 5),
                           stratified = ifelse("stratified" %in% names(params.cv.tr), params.cv.tr[['stratified']], FALSE),
                           early_stopping_rounds = ifelse("early_stopping_rounds" %in% names(params.cv.tr), params.cv.tr[['early_stopping_rounds']], 10),
                           showsd = FALSE, print_every_n = 10, maximize = F, verbose = FALSE # no use at all
          )
          S.fit = xgb.train( params = params.list, data = dtrain, nrounds = xgbcv$best_iteration,
                             watchlist = list(train=dtrain),
                             eval_metric = ifelse("eval_metric" %in% names(params.cv.tr), params.cv.tr[['eval_metric']], 'rmse'),
                             maximize = F, verbose = FALSE)
          if (verbose) {
            print(paste0(Sys.time(), ": Fitting part done!! Next is doing some estimation jobs for next stage."))
          }

          if (stage>1) {
            if (parallel == FALSE) {
              S.est = my.sapply(1:nrow(X.tr), function(i) {
                # find proper A.range if there is any constraint on A given X/Y
                if (is.function(A.cnstr.func)) {
                  if (is.null(x.select)) {
                    A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i,]) )])
                  } else {
                    A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i, x.select]) )])
                  }
                } else {
                  A.range = A.feasible
                }

                dat.tmp = data.frame(trt = A.range, outer(rep(1, length(A.range)), as.numeric(X.tr[i,]))); colnames(dat.tmp) <- store.names
                return( max(predict( S.fit, xgb.DMatrix(data = stats::model.matrix(~trt*.-1, dat.tmp) ) )) )
              }) # should be n.grid x nrow(X.tr)
            } else {
              ## register cores
              if (is.numeric(parallel)) {
                # Start a cluster
                if (parallel.package[1] == "doMC"){
                  doMC::registerDoMC(parallel)
                }
                if (parallel.package[1] == "doParallel") {
                  doParallel::registerDoParallel(parallel)
                }
                if (parallel.package[1] == "snow") {
                  cl <- makeCluster(parallel, type='SOCK')
                  registerDoParallel(cl)
                }
              } else {
                # Start a cluster
                if (parallel.package[1] == "doMC"){
                  doMC::registerDoMC(detectCores())
                }
                if (parallel.package[1] == "doParallel") {
                  doParallel::registerDoParallel(detectCores())
                }
                if (parallel.package[1] == "snow") {
                  cl <- makeCluster(detectCores(), type='SOCK')
                  registerDoParallel(cl)
                }
              }

              S.est = foreach(i=1:nrow(X.tr), .packages=c("xgboost"), .combine = f(nrow(X.tr))) %dopar% {
                # find proper A.range if there is any constraint on A given X/Y
                if (is.function(A.cnstr.func)) {
                  if (is.null(x.select)) {
                    A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i,]) )])
                  } else {
                    A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i, x.select]) )])
                  }
                } else {
                  A.range = A.feasible
                }
                dat.tmp = data.frame(trt = A.range, outer(rep(1, length(A.range)), as.numeric(X.tr[i,]))); colnames(dat.tmp) <- store.names
                return( max(predict( S.fit, xgb.DMatrix(data = stats::model.matrix(~trt*.-1, dat.tmp) ) )))
              } # should be n.grid x nrow(X.tr)

              #Stop the cluster
              if (parallel.package[1] == "snow") {stopCluster(cl)}
            }
          }
        }
        ## KEY: update V.est properly
        if (stage>1) {V.est = as.vector(S.est)}
        ## clean some memory:
        suppressWarnings(remove("S.est", "dat.train", "dat.test"))
        ## Store the trained learners
        S.learners = c(S.learners, list(S.fit))
        remove("S.fit")
        gc(verbose = FALSE)
      }
      names(S.learners) <- paste("S", seq(n.stage, by = -1), sep = ".")
      if (verbose) {
        print(paste0(Sys.time(), ": S-learner training (", baseLearner[1], ") done!!"))
      }
    }
  }


  ######==================== random forests as baselearner ====================
  if (baseLearner[1] == "RF") {
    ######============  S-learner  ============
    if ("S" %in% metaLearners) {
      V.est = 0; S.learners = list()
      for (stage in seq(n.stage, by = -1)) {
        if (verbose) {
          print(paste0(Sys.time(), " @Stage(RF/S-): ", stage))
        }

        n.train = nrow(X[[stage]])
        ## construct training dataset
        if (stage == 1) {
          X.tr = data.frame(X = X[[stage]])
        } else {
          ## note the sequence of X, A, Y
          if (include.X == 1) {
            X.tr = data.frame(X = matrix(unlist(X[1:stage]), nrow = n.train, byrow = FALSE))
          } else {
            X.tr = data.frame(X = X[[stage]])
          }
          if (include.A == 1) { # nothing need to do when include.A == 0
            A.tmp = data.frame(A = A[[stage-1]])
            X.tr = cbind(X.tr, A.tmp)
          } else if (include.A == 2) {
            A.tmp = do.call(cbind.data.frame, A[1:(stage-1)])
            colnames(A.tmp) <- paste("A", 1:(stage-1), sep = ".")
            X.tr = cbind(X.tr, A.tmp)
          }
          if (include.Y == 1) { # nothing need to do when include.Y == 0
            Y.tmp = data.frame(Y = Y[[stage-1]]); colnames(Y.tmp) <- "y"
            X.tr = cbind(X.tr, Y.tmp)
          } else if (include.Y == 2) {
            Y.tmp = data.frame(Y = matrix(unlist(Y[1:(stage-1)]), ncol = stage-1, byrow = FALSE)); colnames(Y.tmp) <- paste("y", seq(ncol(Y.tmp)),sep = "_")
            X.tr = cbind(X.tr, Y.tmp)
          }
        }

        A.tr = A[[stage]]
        Y.tr = Y[[stage]]
        A.feasible = seq(min(A.list[[stage]]), max(A.list[[stage]]), length.out = n.grid)
        V.est = V.est + Y.tr*weights[stage]
        ## generate data set for S-learner:
        dat.tmp = stats::model.matrix(~trt*.-1, data.frame(trt = A.tr, X.tr))
        store.names = colnames(dat.tmp)

        ## train by ranger:
        S.fit = ranger(VVV~., data = data.frame(VVV = V.est, dat.tmp), ...)
        if (verbose) {
          print(paste0(Sys.time(), ": Fitting part done!! Next is doing some estimation jobs for next stage."))
        }

        if (stage>1) {
          if (parallel == FALSE) {
            S.est = my.sapply(1:nrow(X.tr), function(i) {
              # find proper A.range if there is any constraint on A given X/Y
              if (is.function(A.cnstr.func)) {
                if (is.null(x.select)) {
                  A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i,]) )])
                } else {
                  A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i, x.select]) )])
                }
              } else {
                A.range = A.feasible
              }

              dat.tmp = data.frame(trt = A.range, outer(rep(1, length(A.range)), as.numeric(X.tr[i,]))); colnames(dat.tmp) <- c("trt", colnames(X.tr))
              dat.tmp = stats::model.matrix(~trt*.-1, dat.tmp)
              return( max(predict(S.fit, data = data.frame(dat.tmp))$predictions))
            }) # should be n.grid x nrow(X.tr)
          } else {
            ## register cores
            if (is.numeric(parallel)) {
              # Start a cluster
              if (parallel.package[1] == "doMC"){
                doMC::registerDoMC(parallel)
              }
              if (parallel.package[1] == "doParallel") {
                doParallel::registerDoParallel(parallel)
              }
              if (parallel.package[1] == "snow") {
                cl <- makeCluster(parallel, type='SOCK')
                registerDoParallel(cl)
              }
            } else {
              # Start a cluster
              if (parallel.package[1] == "doMC"){
                doMC::registerDoMC(detectCores())
              }
              if (parallel.package[1] == "doParallel") {
                doParallel::registerDoParallel(detectCores())
              }
              if (parallel.package[1] == "snow") {
                cl <- makeCluster(detectCores(), type='SOCK')
                registerDoParallel(cl)
              }
            }

            S.est = foreach(i=1:nrow(X.tr), .packages=c("ranger"), .combine = f(nrow(X.tr))) %dopar% {
              # find proper A.range if there is any constraint on A given X/Y
              if (is.function(A.cnstr.func)) {
                if (is.null(x.select)) {
                  A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i,]) )])
                } else {
                  A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i, x.select]) )])
                }
              } else {
                A.range = A.feasible
              }
              dat.tmp = data.frame(trt = A.range, outer(rep(1, length(A.range)), as.numeric(X.tr[i,]))); colnames(dat.tmp) <- c("trt", colnames(X.tr))
              dat.tmp = stats::model.matrix(~trt*.-1, dat.tmp)
              return( max(predict(S.fit, data = data.frame(dat.tmp))$predictions))
            } # should be n.grid x nrow(X.tr)

            #Stop the cluster
            if (parallel.package[1] == "snow") {stopCluster(cl)}
          }
        }

        ## KEY: update V.est properly
        if (stage>1) {V.est = as.vector(S.est)}
        ## clean some memory:
        suppressWarnings(remove("S.est", "dat.train", "dat.test"))
        ## Store the trained learners
        S.learners = c(S.learners, list(S.fit))
        remove("S.fit")
        gc(verbose = FALSE)
      }
      names(S.learners) <- paste("S", seq(n.stage, by = -1), sep = ".")
      if (verbose) {
        print(paste0(Sys.time(), ": S-learner training (RF) done!!"))
      }
    }
  }


  ######==================== GAM as baselearner ====================
  if (baseLearner[1] == "GAM") {

    ######============  S-learner  ============
    if ("S" %in% metaLearners) {
      V.est = 0; S.learners = list()
      for (stage in seq(n.stage, by = -1)) {
        if (verbose) {
          print(paste0(Sys.time(), " @Stage(GAM/S-): ", stage))
        }

        n.train = nrow(X[[stage]])
        ## construct training dataset
        if (stage == 1) {
          X.tr = data.frame(X = X[[stage]])
        } else {
          ## note the sequence of X, A, Y
          if (include.X == 1) {
            X.tr = data.frame(X = matrix(unlist(X[1:stage]), nrow = n.train, byrow = FALSE))
          } else {
            X.tr = data.frame(X = X[[stage]])
          }
          if (include.A == 1) { # nothing need to do when include.A == 0
            A.tmp = data.frame(A = A[[stage-1]])
            X.tr = cbind(X.tr, A.tmp)
          } else if (include.A == 2) {
            A.tmp = do.call(cbind.data.frame, A[1:(stage-1)])
            colnames(A.tmp) <- paste("A", 1:(stage-1), sep = ".")
            X.tr = cbind(X.tr, A.tmp)
          }
          if (include.Y == 1) { # nothing need to do when include.Y == 0
            Y.tmp = data.frame(Y = Y[[stage-1]]); colnames(Y.tmp) <- "y"
            X.tr = cbind(X.tr, Y.tmp)
          } else if (include.Y == 2) {
            Y.tmp = data.frame(Y = matrix(unlist(Y[1:(stage-1)]), ncol = stage-1, byrow = FALSE)); colnames(Y.tmp) <- paste("y", seq(ncol(Y.tmp)),sep = "_")
            X.tr = cbind(X.tr, Y.tmp)
          }
        }

        A.tr = A[[stage]]
        Y.tr = Y[[stage]]
        A.feasible = seq(min(A.list[[stage]]), max(A.list[[stage]]), length.out = n.grid)
        V.est = V.est + Y.tr*weights[stage]
        ## generate data set for S-learner:
        dat.tmp = data.frame(trt = A.tr, X.tr)
        store.names = colnames(dat.tmp)

        ## train by glmnet with LASSO penalty:
        if (is.numeric(parallel)) {
          doParallel::registerDoParallel(parallel)
        } else {
          doParallel::registerDoParallel(detectCores())
        }
        S.fit = cv.glmnet(stats::model.matrix(~trt*.-1, dat.tmp), V.est, family = "gaussian", parallel = TRUE, ...)
        if (verbose) {
          print(paste0(Sys.time(), ": Fitting part done!! Next is doing some estimation jobs for next stage."))
        }

        if (stage>1) {
          if (parallel == FALSE) {
            S.est = my.sapply(1:nrow(X.tr), function(i) {
              # find proper A.range if there is any constraint on A given X/Y
              if (is.function(A.cnstr.func)) {
                if (is.null(x.select)) {
                  A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i,]) )])
                } else {
                  A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i, x.select]) )])
                }
              } else {
                A.range = A.feasible
              }

              dat.tmp = data.frame(trt = A.range, outer(rep(1, length(A.range)), as.numeric(X.tr[i,]))); colnames(dat.tmp) <- store.names
              return( max(predict(S.fit, newx = stats::model.matrix(~trt*.-1, dat.tmp), type = "response", s = "lambda.min")))
            }) # should be n.grid x nrow(X.tr)
          } else {
            ## register cores
            if (is.numeric(parallel)) {
              # Start a cluster
              if (parallel.package[1] == "doMC"){
                doMC::registerDoMC(parallel)
              }
              if (parallel.package[1] == "doParallel") {
                doParallel::registerDoParallel(parallel)
              }
              if (parallel.package[1] == "snow") {
                cl <- makeCluster(parallel, type='SOCK')
                registerDoParallel(cl)
              }
            } else {
              # Start a cluster
              if (parallel.package[1] == "doMC"){
                doMC::registerDoMC(detectCores())
              }
              if (parallel.package[1] == "doParallel") {
                doParallel::registerDoParallel(detectCores())
              }
              if (parallel.package[1] == "snow") {
                cl <- makeCluster(detectCores(), type='SOCK')
                registerDoParallel(cl)
              }
            }

            S.est = foreach(i=1:nrow(X.tr), .packages=c("glmnet"), .combine = f(nrow(X.tr))) %dopar% {
              # find proper A.range if there is any constraint on A given X/Y
              if (is.function(A.cnstr.func)) {
                if (is.null(x.select)) {
                  A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i,]) )])
                } else {
                  A.range = suppressWarnings(A.feasible[A.cnstr.func(A.feasible, as.matrix(X[[stage]][i, x.select]) )])
                }
              } else {
                A.range = A.feasible
              }
              dat.tmp = data.frame(trt = A.range, outer(rep(1, length(A.range)), as.numeric(X.tr[i,]))); colnames(dat.tmp) <- store.names
              return( max(predict(S.fit, newx = stats::model.matrix(~trt*.-1, dat.tmp), type = "response", s = "lambda.min")))
            } # should be n.grid x nrow(X.tr)

            #Stop the cluster
            if (parallel.package[1] == "snow") {stopCluster(cl)}
          }
        }

        ## KEY: update V.est properly
        if (stage>1) {V.est = as.vector(S.est)}
        ## Store the trained learners
        S.learners = c(S.learners, list(S.fit))
      }
      names(S.learners) <- paste("S", seq(n.stage, by = -1), sep = ".")
      if (verbose) {
        print(paste0(Sys.time(), ": S-learner training (GAM) done!!"))
      }
    }
  }


  ######========== Prepare outputs
  DTRres <- list(S.learners = S.learners,
                 controls = list(
                   n.stage = n.stage,
                   A.list  = A.list,
                   baseLearner   = baseLearner[1],
                   metaLearners  = metaLearners,
                   include.X = include.X,
                   include.Y = include.Y,
                   include.A = include.A,
                   A.cnstr.func = A.cnstr.func,
                   x.select = x.select,
                   n.grid = n.grid,
                   verbose = verbose,
                   parallel.package = parallel.package
                 )
  )

  class(DTRres) <- "metaDTR"
  return(DTRres)

}


