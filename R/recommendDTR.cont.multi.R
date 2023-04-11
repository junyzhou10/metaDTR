#' @title Recommend the optimal treatment at each stage (Continuous Treatment & Multiple Outcomes)
#' @author Junyi Zhou \email{junyzhou@iu.edu}
#' @description This function make recommendations of optimal treatment for a given subject at given stage
#' @param DTRs Output from [learnDTR.cont.multi()] that belong to class \code{metaDTR}.
#' @param currentDTRs Object from [recommendDTR.cont.multi()]. Records the results from previous stages so that users can
#'                    flexibly make predictions stage by stage. See details and example.
#' @param X.new A list of covariates at each stage. In practice, since recommendations are made
#'              stage by stage, here one stage of \code{X.new} will provide one stage's recommendation of action.
#' @param A.new A list of observed or predicted actions (from the algorithm). Users can provide real actions observed
#'              rather than algorithm recommended to make this function more flexible.
#'              It allows to handle cases that the realized action/observation may not be consistent
#'              with algorithm results. Only needed when \code{all.inclusive = TRUE}.
#'              Default is \code{NULL}.
#' @param Y.new A list. Required if \code{include.Y > 0} in [learnDTR.cont.multi()]. Default is \code{NULL}
#' @param A.feasible Optional list, default is \code{NULL}. This allow user to specify the subject level's feasible action/treatment
#'                   space. The length of list should be equal to the number of stages, and each element should be
#'                   an N x 2 of matrix, where N represents the number of subjects and each row is the range of
#'                   feasible action/treatment, i.e., (min, max). If \code{A.cnstr.func} is specified in [learnDTR.cont.multi()],
#'                   then feasible set will also be made to satisfy that.
#' @param A.cnstr.func Same as in [learnDTR.cont.multi()]. Provide a chance for it to be override, not very common though.
#'                     If not specified, it will be inherited directly from the returns of [learnDTR.cont.multi()].
#' @param n.grid Same as in [learnDTR.cont.multi()]. If not specified, it will be inherited from [learnDTR.cont.multi()].
#' @param parallel A boolean, for whether parallel computing is adopted. Also, if a numeric value, it implies the
#'                 number of cores to use. Otherwise, directly use the number from [learnDTR.cont.multi()]
#' @param verbose Console output allowed? Default is \code{NULL}, which will inherit the argument input of
#' @details This function make recommendations based on the trained learners from [learnDTR.cont.multi()] for new dataset.
#'          Since in real application, later stage covariates/outcomes are unobservable until treatments are
#'          assigned. So in most cases, [recommendDTR.cont.multi()] needs to be applied stage by stage, which is allowed in
#'          this package. User can provide available information stage by stage, and obtain the optimal recommendations
#'          at each stage, iteratively.
#'
#' @return It returns the optimal action/treatment recommendations at each stage and results are stored in a list for
#'         each meta-learner method.
#' @examples
#' ## Similar example as recommendDTR()
#' ## First: Modify dataset to 2 actions and 2 outcomes:
#' tmp = ThreeStg_Dat
#' tmp$X = lapply(tmp$X, function(x) as.data.frame(apply(x, 2, scale)))
#' tmp$X.test = lapply(tmp$X.test, function(x) as.data.frame(apply(x, 2, scale)))
#' tmp$A = lapply(tmp$A, function(x) cbind(rnorm(400,0,1), rnorm(400,0,2)))
#' tmp$Y = lapply(tmp$Y, function(x) cbind(x, rnorm(400,0,0.2)+x*runif(400,0,1)))
#'
#' ## Apply the main function to learn the DTRs
#' DTRs = learnDTR.cont.multi(X = tmp$X,
#'                           A = tmp$A,
#'                           Y = tmp$Y,
#'                           weights = rep(1, 3),
#'                           weights.Y = c(0.3,0.7),
#'                           baseLearner  = c("XGBoost"),
#'                           metaLearners = c("S"),
#'                           include.X = 1,
#'                           include.A = 2,
#'                           include.Y = 0,
#'                           A.box.cnstr = cbind(c(-2,2), c(-2,2)),
#'                           A.cnstr.func = function(a, x) {
#'                             abs(a[,1]+x[,1]) + abs(a[,2]+x[,2]) <= 3
#'                           },
#'                           x.select = c("V1", "V2"),
#'                           n.grid = 50,
#'                           parallel = F)
#' ## Second: Find DTR on test set
#' optDTR <- recommendDTR.cont.multi(DTRs, currentDTRs = NULL, n.grid = 50,
#'                         X.new = tmp$X.test, parallel = F, verbose = TRUE)
#' @import dbarts glmnet ranger xgboost pbapply doParallel snow utils foreach
#' @export
#' @seealso \code{\link{learnDTR.cont.multi}}

recommendDTR.cont.multi <- function(DTRs, currentDTRs = NULL,
                              X.new, A.new = NULL, Y.new = NULL,
                              A.feasible = NULL,
                              A.cnstr.func = NULL,
                              n.grid = 100, parallel = FALSE, verbose = NULL) {
  n.stage <- DTRs$controls$n.stage
  baseLearner <- DTRs$controls$baseLearner
  metaLearners <- DTRs$controls$metaLearners
  A.list <- DTRs$controls$A.list
  include.X <- DTRs$controls$include.X
  include.Y <- DTRs$controls$include.Y
  include.A <- DTRs$controls$include.A
  store.names <- DTRs$controls$store.names
  if (is.null(A.cnstr.func)) {
    A.cnstr.func <- DTRs$controls$A.cnstr.func
  }

  x.select <- DTRs$controls$x.select
  if (is.null(verbose)) {
    verbose <- DTRs$controls$verbose
  }
  if (is.null(n.grid)) {
    n.grid <- DTRs$controls$n.grid
  }

  # a simply defined function to reflect verbose requirement
  if (verbose) {
    my.sapply <- function(...){pbsapply(...)}
  } else {
    my.sapply <- function(...){sapply(...)}
  }

  ## Register cores if parallel
  if (parallel == FALSE) {

  } else {
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

    if (is.numeric(parallel)) {
      # Start a cluster
      cl <- makeCluster(parallel, type='SOCK')
      registerDoParallel(cl)
    } else {
      # Start a cluster
      cl <- makeCluster(detectCores(), type='SOCK')
      registerDoParallel(cl)
    }
  }

  ######==================== check/arrange inputs ====================
  if (is.null(currentDTRs)) {
    if (class(X.new)[1] != "list") {
      X.new = list(X.new)
    }
    if (is.null(A.feasible)) {
      rep(list(NULL), length(X.new))
    } else {
      if (class(A.feasible)[1] != "list") {
        A.feasible = list(A.feasible)
      }
    }
    A.obs  <- list(ifelse(is.null(A.new), NA, A.new))
    A.ind  <- !is.na(A.obs)[-1]
    A.opt.S <- A.opt.T <- A.opt.deC <- list() # main output
    start = 1 # start from first stage
  } else {
    A.obs  <- c(DTRs$controls$A.obs, list(ifelse(is.null(A.new), NA, A.new)))
    A.ind  <- !is.na(A.obs)[-1]
    if (class(X.new)[1] == "list") {
      X.new = c(currentDTRs$controls$X.new, X.new)
    } else {
      X.new = c(currentDTRs$controls$X.new, list(X.new))
    }
    if (class(Y.new)[1] == "list") {
      Y.new = c(currentDTRs$controls$Y.new, Y.new)
    } else {
      Y.new = c(currentDTRs$controls$Y.new, list(Y.new))
    }
    if (class(A.feasible)[1] == "list") {
      A.feasible = c(currentDTRs$controls$A.feasible, A.feasible)
    } else {
      A.feasible = c(currentDTRs$controls$A.feasible, list(A.feasible))
    }
    A.opt.S <- currentDTRs$A.opt.S
    A.opt.T <- currentDTRs$A.opt.T
    A.opt.deC <- currentDTRs$A.opt.deC

    start = length(X.new) # start from middle stages
  }

  n.step <- length(X.new)
  if (start > n.step) {
    stop("start stage exceeds the information provided. Please check!\n")
  }



  ######==================== bart/rf/gam as base learner ====================
  if (baseLearner %in% c("BART", "GAM", "RF", "XGBoost")) {
    ######============  S-learner  ============
    if ("S" %in% metaLearners) {
      for (stage in seq(start, n.step, by = 1)) { # forward now
        n.test = nrow(X.new[[stage]])

        ## retrieve training dataset structures
        if (stage == 1) {
          X.te = data.frame(X = X.new[[stage]])
        } else {
          ## follow the sequence of X, A, Y
          if (include.X == 1) {
            X.te = data.frame(X = matrix(unlist(X.new[1:stage]), nrow = n.test, byrow = FALSE))
          } else {
            X.te = data.frame(X = X.new[[stage]])
          }
          ## override is triggered
          A.new <- A.opt.S
          A.new[A.ind] <- A.obs[A.ind]
          if (include.A == 1) { # nothing need to do when include.A == 0
            A.tmp = data.frame(A = A.new[[stage-1]])
            # for (ii in seq(ncol(A.tmp))) {
            #   A.tmp[,ii] <- factor(A.tmp[,ii], levels = A.list[[stage-1]])
            # }
            X.te = cbind(X.te, A.tmp)
          } else if (include.A == 2) {
            A.tmp = do.call(cbind.data.frame, A.new[1:(stage-1)])
            colnames(A.tmp) <- paste("A", 1:(stage-1), sep = ".")
            # A.tmp = data.frame(A = matrix(unlist(A.new[1:(stage-1)]), nrow = n.test, byrow = FALSE))
            X.te = cbind(X.te, A.tmp)
          }
          if (include.Y == 1) { # nothing need to do when include.Y == 0
            Y.tmp = data.frame(Y = Y.new[[stage-1]]); colnames(Y.tmp) <- "y"
            X.te = cbind(X.te, Y.tmp)
          } else if (include.Y == 2) {
            Y.tmp = data.frame(Y = matrix(unlist(Y.new[1:(stage-1)]), nrow = n.test, byrow = FALSE)); colnames(Y.tmp) <- paste("y", seq(ncol(Y.tmp)),sep = "_")
            X.te = cbind(X.te, Y.tmp)
          }
        }

        ## predict outcome based on trained learners:
        A.range = do.call(expand.grid,
                             apply(A.list[[stage]], 2, function(x) seq(min(x), max(x), length.out = n.grid), simplify = FALSE)
        )
        A.range = as.matrix(A.range)

        if (baseLearner == "BART") {
          if (parallel == FALSE) {
            A.pred = my.sapply(1:nrow(X.te), function(i) {
              if (!is.null(A.feasible[[stage]])) {
                A.ff = A.range[A.range <= max(A.feasible[[stage]][i,2]) & A.range >= min(A.feasible[[stage]][i,1]), ]
              } else {
                A.ff = A.range
              }

              if (is.function(A.cnstr.func)) {
                if (is.null(x.select)) {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,])), ])
                } else {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,x.select])), ])
                }
              }

              dat.tmp = NULL
              for (p in 1:ncol(A.range)) {
                dat.tmp = cbind(dat.tmp,
                                stats::model.matrix(~trt*.-1, data.frame(trt = A.ff[,p], outer(rep(1,nrow(A.ff)), as.numeric(X.te[i,])) )))
              }
              colnames(dat.tmp) = store.names[[stage]]
              Y.pred = colMeans(predict(DTRs$S.learners[[n.stage - stage + 1]], newdata = dat.tmp))
              return( A.ff[which.max(Y.pred),] )
            })
          } else {
            A.pred = foreach(i=1:nrow(X.te), .packages=c("dbarts"), .combine = f(nrow(X.te))) %dopar% {
              if (!is.null(A.feasible[[stage]])) {
                A.ff = A.range[A.range <= max(A.feasible[[stage]][i,2]) & A.range >= min(A.feasible[[stage]][i,1]), ]
              } else {
                A.ff = A.range
              }
              if (is.function(A.cnstr.func)) {
                if (is.null(x.select)) {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,])), ])
                } else {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,x.select])), ])
                }
              }

              dat.tmp = NULL
              for (p in 1:ncol(A.range)) {
                dat.tmp = cbind(dat.tmp,
                                stats::model.matrix(~trt*.-1, data.frame(trt = A.ff[,p], outer(rep(1,nrow(A.ff)), as.numeric(X.te[i,])) )))
              }
              colnames(dat.tmp) = store.names[[stage]]
              Y.pred = colMeans(predict(DTRs$S.learners[[n.stage - stage + 1]], newdata = dat.tmp))
              return( A.ff[which.max(Y.pred),] )
            }
          }
        }

        if (baseLearner == "XGBoost") {
          if (parallel == FALSE) {
            A.pred = my.sapply(1:nrow(X.te), function(i) {
              if (!is.null(A.feasible[[stage]])) {
                A.ff = A.range[A.range <= max(A.feasible[[stage]][i,2]) & A.range >= min(A.feasible[[stage]][i,1]), ]
              } else {
                A.ff = A.range
              }
              if (is.function(A.cnstr.func)) {
                if (is.null(x.select)) {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,])), ])
                } else {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,x.select])), ])
                }
              }
              dat.tmp = NULL
              for (p in 1:ncol(A.range)) {
                dat.tmp = cbind(dat.tmp,
                                stats::model.matrix(~trt*.-1, data.frame(trt = A.ff[,p], outer(rep(1,nrow(A.ff)), as.numeric(X.te[i,])) )))
              }
              colnames(dat.tmp) = store.names[[stage]]
              Y.pred = predict(DTRs$S.learners[[n.stage - stage + 1]], newdata = xgb.DMatrix(data = dat.tmp))
              return( A.ff[which.max(Y.pred),] )
            })
          } else {
            A.pred = foreach(i=1:nrow(X.te), .packages=c("xgboost"), .combine = f(nrow(X.te))) %dopar% {
              if (!is.null(A.feasible[[stage]])) {
                A.ff = A.range[A.range <= max(A.feasible[[stage]][i,2]) & A.range >= min(A.feasible[[stage]][i,1]), ]
              } else {
                A.ff = A.range
              }
              if (is.function(A.cnstr.func)) {
                if (is.null(x.select)) {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,])), ])
                } else {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,x.select])), ])
                }
              }
              dat.tmp = NULL
              for (p in 1:ncol(A.range)) {
                dat.tmp = cbind(dat.tmp,
                                stats::model.matrix(~trt*.-1, data.frame(trt = A.ff[,p], outer(rep(1,nrow(A.ff)), as.numeric(X.te[i,])) )))
              }
              colnames(dat.tmp) = store.names[[stage]]
              Y.pred = predict(DTRs$S.learners[[n.stage - stage + 1]], newdata = xgb.DMatrix(data = dat.tmp))
              return( A.ff[which.max(Y.pred),] )
            }
          }
        }


        if (baseLearner == "RF") {
          if (parallel == FALSE) {
            A.pred = my.sapply(1:nrow(X.te), function(i) {
              if (!is.null(A.feasible[[stage]])) {
                A.ff = A.range[A.range <= max(A.feasible[[stage]][i,2]) & A.range >= min(A.feasible[[stage]][i,1]), ]
              } else {
                A.ff = A.range
              }
              if (is.function(A.cnstr.func)) {
                if (is.null(x.select)) {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,])), ])
                } else {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,x.select])), ])
                }
              }

              dat.tmp = NULL
              for (p in 1:ncol(A.range)) {
                dat.tmp = cbind(dat.tmp,
                                stats::model.matrix(~trt*.-1, data.frame(trt = A.ff[,p], outer(rep(1,nrow(A.ff)), as.numeric(X.te[i,])) )))
              }
              colnames(dat.tmp) = store.names[[stage]]
              Y.pred = predict(DTRs$S.learners[[n.stage - stage + 1]], data = data.frame(dat.tmp) )$predictions
              return( A.ff[which.max(Y.pred),] )
            })
          } else {
            A.pred = foreach(i=1:nrow(X.te), .packages=c("ranger"), .combine = f(nrow(X.te))) %dopar% {
              if (!is.null(A.feasible[[stage]])) {
                A.ff = A.range[A.range <= max(A.feasible[[stage]][i,2]) & A.range >= min(A.feasible[[stage]][i,1]), ]
              } else {
                A.ff = A.range
              }
              if (is.function(A.cnstr.func)) {
                if (is.null(x.select)) {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,])), ])
                } else {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,x.select])), ])
                }
              }

              dat.tmp = NULL
              for (p in 1:ncol(A.range)) {
                dat.tmp = cbind(dat.tmp,
                                stats::model.matrix(~trt*.-1, data.frame(trt = A.ff[,p], outer(rep(1,nrow(A.ff)), as.numeric(X.te[i,])) )))
              }
              colnames(dat.tmp) = store.names[[stage]]
              Y.pred = predict(DTRs$S.learners[[n.stage - stage + 1]], data = data.frame(dat.tmp) )$predictions
              return( A.ff[which.max(Y.pred),] )
            }
          }
        }


        if (baseLearner == "GAM") {
          if (parallel == FALSE) {
            A.pred = my.sapply(1:nrow(X.te), function(i) {
              if (!is.null(A.feasible[[stage]])) {
                A.ff = A.range[A.range <= max(A.feasible[[stage]][i,2]) & A.range >= min(A.feasible[[stage]][i,1]), ]
              } else {
                A.ff = A.range
              }
              if (is.function(A.cnstr.func)) {
                if (is.null(x.select)) {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,])), ])
                } else {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,x.select])), ])
                }
              }

              dat.tmp = NULL
              for (p in 1:ncol(A.range)) {
                dat.tmp = cbind(dat.tmp,
                                stats::model.matrix(~trt*.-1, data.frame(trt = A.ff[,p], outer(rep(1,nrow(A.ff)), as.numeric(X.te[i,])) )))
              }
              colnames(dat.tmp) = store.names[[stage]]
              Y.pred = predict(DTRs$S.learners[[n.stage - stage + 1]], dat.tmp,
                               type = "response", s = "lambda.min")
              return( A.ff[which.max(Y.pred),] )
            })
          } else {
            A.pred = foreach(i=1:nrow(X.te), .packages=c("glmnet"), .combine = f(nrow(X.te))) %dopar% {
              if (!is.null(A.feasible[[stage]])) {
                A.ff = A.range[A.range <= max(A.feasible[[stage]][i,2]) & A.range >= min(A.feasible[[stage]][i,1]), ]
              } else {
                A.ff = A.range
              }
              if (is.function(A.cnstr.func)) {
                if (is.null(x.select)) {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,])), ])
                } else {
                  A.ff = suppressWarnings(A.ff[A.cnstr.func(A.ff, as.matrix(X.new[[stage]][i,x.select])), ])
                }
              }

              dat.tmp = NULL
              for (p in 1:ncol(A.range)) {
                dat.tmp = cbind(dat.tmp,
                                stats::model.matrix(~trt*.-1, data.frame(trt = A.ff[,p], outer(rep(1,nrow(A.ff)), as.numeric(X.te[i,])) )))
              }
              colnames(dat.tmp) = store.names[[stage]]
              Y.pred = predict(DTRs$S.learners[[n.stage - stage + 1]], newx = dat.tmp,
                               type = "response", s = "lambda.min")
              return( A.ff[which.max(Y.pred),] )
            }
          }
        }

        A.opt.S = c(A.opt.S, list(t(A.pred)))
        A.new <- c(A.new, list(t(A.pred)))

        if (verbose) {
          print(paste0(Sys.time(), ": Stage ", stage, " done!!"))
        }
      }
      names(A.opt.S) <- c(names(currentDTRs$A.opt.S), paste("Stage", seq(start, n.step, by = 1), sep = "."))
    }



  } else {
    warning("Please choose a proper base-learner!\n")
  }

  if (!parallel == FALSE) {
    #Stop the cluster
    stopCluster(cl)
  }
  ######========== Prepare outputs
  OptDTR <- list(A.opt.S = A.opt.S,
                 controls = list(
                   X.new = X.new,
                   A.obs = A.obs,
                   A.feasible = A.feasible,
                   Y.new = Y.new,
                   n.stage = n.stage,
                   A.list  = A.list,
                   baseLearner   = baseLearner,
                   metaLearners  = metaLearners
                 )
  )
  class(OptDTR) <- "metaDTR"
  return(OptDTR)
}




