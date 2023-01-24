#' @title Learning DTR from Sequential Interventions
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
#'                    package \code{dbarts}, \code{RF} (random forests) by \code{ranger}, and \code{GAM}
#'                    through package \code{glmnet}, which can provide variable selection/sparsity by
#'                    various type of regularizations. So more in details.
#' @param metaLearners \code{c("S", "T", "deC")}. Meta-learner algorithms to learn the optimal DTR. To support more than two actions
#'                     at each stage, S-, A-, and deC-learner are available. But deC-learner only works when
#'                     \code{baseLearner = "GAM"} so far.
#' @param include.X 0 for no past X included in analysis; 1 for all past X included
#' @param include.A 0 for no past treatment assignment included in analysis; 1 for only last A included; 2 for all past
#'                  A included
#' @param include.Y 0 for no past reward/outcome Y included in analysis; 1 for only last Y included; 2 for all past
#'                  Y included
#' @param est.sigma Initial estimation of sigma. Only for T-learner with BART. If sample size is not enough to estimate
#'                  surface separately, or algorithm experience some trouble in getting sigma, use this argument to
#'                  provide an initial estimate.
#' @param verbose Console print allowed?
#' @param ... Additional arguments that can be passed to \code{dbarts::bart}, \code{ranger::ranger}, or \code{glmnet::cv.glmnet}
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
#'         sequential recommendations might require intermediate observations, [learnDTR] will not automatically
#'         provide prediction. But by another function [recommendDTR],
#'         predictions can flexibly been made stage by stage.
#' @examples
#' ## this is the sample adopted in:
#' ## https://jzhou.org/posts/optdtr/#case-1-random-assignment-with-two-treatment-options
#' DTRs = learnDTR(X = ThreeStg_Dat$X,
#'                 A = ThreeStg_Dat$A,
#'                 Y = ThreeStg_Dat$Y,
#'                 weights = rep(1, 3),
#'                 baseLearner  = c("BART"),
#'                 metaLearners = c("S", "T"),
#'                 include.X = 1,
#'                 include.A = 2,
#'                 include.Y = 0)
#' @import stats utils dbarts glmnet ranger
#' @references
#' Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu.
#' "Metalearners for estimating heterogeneous treatment effects using machine learning."
#' *Proceedings of the national academy of sciences* 116, no. 10 (2019): 4156-4165.
#' \cr\cr
#' Zhou, Junyi, Ying Zhang, and Wanzhu Tu. "A reference-free R-learner for treatment recommendation."
#' *Statistical Methods in Medical Research* (2022)
#' @export
#' @seealso \code{\link{recommendDTR}}

learnDTR <- function(X, A, Y, weights = rep(1, length(X)),
                     baseLearner  = c("RF", "BART", "GAM"),
                     metaLearners = c("S", "T", "deC"),
                     include.X = 0, include.A = 0, include.Y = 0,
                     est.sigma = NULL,
                     verbose = TRUE, ...
) {
  n.stage = length(X)
  ######==================== check inputs ====================
  if (length(unique(length(X), length(A), length(Y))) != 1){
    stop("Inconsistent input of X, A, and Y. Please check!\n")
  }

  A <- lapply(A, as.factor)
  A.list <- lapply(A, function(x) sort(unique(x)))
  S.learners <- T.learners <- deC.learners <- NULL

  ######==================== bart as baselearner ====================
  if (baseLearner[1] == "BART") {
    ######============  S-learner  ============
    if ("S" %in% metaLearners) {
      V.est = 0; S.learners = list()
      for (stage in seq(n.stage, by = -1)) {
        if (verbose) {
          print(paste0(Sys.time(), " @Stage(BART/S-): ", stage))
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
        K.grp = sort(unique(A.tr))
        V.est = V.est + Y.tr*weights[stage]
        ## generate data set for S-learner:
        dat.tmp = data.frame(trt = A.tr, X.tr)
        for (ii in K.grp) {
          tmp = data.frame(trt = ii, X.tr)
          dat.tmp = rbind(dat.tmp, tmp)
        }
        dat.tmp$trt = as.factor(dat.tmp$trt)
        dat.S = stats::model.matrix(~trt*.-1, dat.tmp)
        dat.train = dat.S[1:n.train,]; dat.test = dat.S[-(1:n.train),]
        ## train by bart:
        S.fit = bart(x.train = dat.train, y.train = V.est, keeptrainfits = F,
                     keeptrees = TRUE, verbose = FALSE, ...)
        S.est = matrix(colMeans(predict(S.fit, newdata = dat.test)), ncol = length(K.grp), byrow = F)
        ## KEY: update V.est properly
        V.est = apply(S.est, 1, max)
        ## clean some memory:
        remove("S.est", "dat.train", "dat.test")
        ## Store the trained learners
        invisible(S.fit$fit$state)
        S.learners = c(S.learners, list(S.fit))
        remove("S.fit")
        gc(verbose = FALSE)
      }
      names(S.learners) <- paste("S", seq(n.stage, by = -1), sep = ".")
      if (verbose) {
        print(paste0(Sys.time(), ": S-learner training (BART) done!!"))
      }
    }

    ######============  T-learner  ============
    if ("T" %in% metaLearners) {
      V.est = 0; T.learners = list()
      for (stage in seq(n.stage, by = -1)) {
        if (verbose) {
          print(paste0(Sys.time(), " @Stage(BART/T-): ", stage))
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
        K.grp = sort(unique(A.tr))
        n.train = nrow(X.tr)
        V.est = V.est + Y.tr*weights[stage]
        ## directly train T-learner with bart:
        T.est = NULL; T.stage = list();

        for (ii in K.grp) {
          if (is.null(est.sigma)) {
            if (as.numeric(table(A.tr)[paste(ii)]) < 1.05*ncol(X.tr)) {# in case n<p
              est.sigma = 1
            } else {
              est.sigma = NA
            }
          }
          ## if use BART package rather than dbarts
          # T.fit = mc.wbart(x.train = X.tr[A.tr==ii,], y.train = V.est[A.tr==ii],
          #                  ntree = 200L, keeptrainfits = FALSE, ...)
          ## if fit by dbarts
          T.fit = bart(x.train = X.tr[A.tr==ii,], y.train = V.est[A.tr==ii], keeptrainfits = F,
                       keeptrees = TRUE, verbose = FALSE,
                       sigest = est.sigma,
                       ...)
          T.est = cbind(T.est, colMeans(predict(T.fit, newdata = X.tr ) ) )
          invisible(T.fit$fit$state)
          T.stage = c(T.stage, list(T.fit))
          gc(verbose = FALSE)
        }
        ## KEY: update V.est properly
        V.est = apply(T.est, 1, max)
        ## clean some memory:
        remove("T.fit", "T.est", "X.tr")
        ## Store the trained learners
        names(T.stage) <- paste("A", K.grp, sep = ".")
        T.learners = c(T.learners, list(T.stage))
      }
      names(T.learners) <- paste("T", seq(n.stage, by = -1), sep = ".")
      if (verbose) {
        print(paste0(Sys.time(), ": T-learner training (BART) done!!"))
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
        K.grp = sort(unique(A.tr))
        V.est = V.est + Y.tr*weights[stage]
        ## generate data set for S-learner:
        dat.tmp = data.frame(trt = A.tr, X.tr)
        for (ii in K.grp) {
          tmp = data.frame(trt = ii, X.tr)
          dat.tmp = rbind(dat.tmp, tmp)
        }
        dat.tmp$trt = as.factor(dat.tmp$trt)
        dat.S = stats::model.matrix(~trt*.-1, dat.tmp)
        dat.train = dat.S[1:n.train,]; dat.test = dat.S[-(1:n.train),]
        ## train by bart:
        S.fit = ranger(V~., data = data.frame(V = V.est, dat.train), ...)
        S.est = matrix(predict(S.fit, data = data.frame(dat.test))$predictions, ncol = length(K.grp), byrow = F)
        ## KEY: update V.est properly
        V.est = apply(S.est, 1, max)
        ## clean some memory:
        remove("S.est", "dat.train", "dat.test")
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

    ######============  T-learner  ============
    if ("T" %in% metaLearners) {
      V.est = 0; T.learners = list()
      for (stage in seq(n.stage, by = -1)) {
        if (verbose) {
          print(paste0(Sys.time(), " @Stage(RF/T-): ", stage))
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
        K.grp = sort(unique(A.tr))
        n.train = nrow(X.tr)
        V.est = V.est + Y.tr*weights[stage]
        ## directly train T-learner with bart:
        T.est = NULL; T.stage = list();

        for (ii in K.grp) {
          T.fit = ranger(V~., data = data.frame(V = V.est, X.tr)[A.tr==ii,], ...)
          T.est = cbind(T.est, predict(T.fit, data = data.frame(X.tr))$predictions)
          T.stage = c(T.stage, list(T.fit))
          gc(verbose = FALSE)
        }
        ## KEY: update V.est properly
        V.est = apply(T.est, 1, max)
        ## clean some memory:
        remove("T.fit", "T.est", "X.tr")
        ## Store the trained learners
        names(T.stage) <- paste("A", K.grp, sep = ".")
        T.learners = c(T.learners, list(T.stage))
      }
      names(T.learners) <- paste("T", seq(n.stage, by = -1), sep = ".")
      if (verbose) {
        print(paste0(Sys.time(), ": T-learner training (RF) done!!"))
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
        K.grp = sort(unique(A.tr))
        V.est = V.est + Y.tr*weights[stage]
        ## generate data set for S-learner:
        dat.train = data.frame(trt = A.tr, X.tr)
        dat.tmp = dat.train
        for (ii in K.grp) {
          tmp = data.frame(trt = ii, X.tr)
          dat.tmp = rbind(dat.tmp, tmp)
        }
        dat.tmp$trt = as.factor(dat.tmp$trt)
        dat.S = stats::model.matrix(~trt*.-1, dat.tmp)
        dat.train = dat.S[1:n.train,]; dat.test = dat.S[-(1:n.train),]
        ## train by glmnet with LASSO penalty:
        S.fit = cv.glmnet(dat.train, V.est, family = "gaussian", parallel = TRUE, ...)
        S.est = matrix(predict(S.fit, newx = dat.test, type = "response", s = "lambda.min"), ncol = length(K.grp), byrow = F)

        ## KEY: update V.est properly
        V.est = apply(S.est, 1, max)
        ## Store the trained learners
        S.learners = c(S.learners, list(S.fit))
      }
      names(S.learners) <- paste("S", seq(n.stage, by = -1), sep = ".")
      if (verbose) {
        print(paste0(Sys.time(), ": S-learner training (GAM) done!!"))
      }
    }


    ######============  T-learner  ============
    if ("T" %in% metaLearners) {
      V.est = 0; T.learners = list()
      for (stage in seq(n.stage, by = -1)) {
        if (verbose) {
          print(paste0(Sys.time(), " @Stage(GAM/T-): ", stage))
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
        K.grp = sort(unique(A.tr))
        n.train = nrow(X.tr)
        V.est = V.est + Y.tr*weights[stage]
        ## directly train T-learner with glmnet:
        T.est = NULL; T.stage = list();
        for (ii in K.grp) {
          T.fit = cv.glmnet(stats::model.matrix(~.-1, X.tr)[A.tr==ii,], V.est[A.tr==ii], family = "gaussian", parallel = TRUE, ...)
          T.est = cbind(T.est, predict(T.fit, newx = stats::model.matrix(~.-1, X.tr), type = "response", s = "lambda.min"))
          T.stage = c(T.stage, list(T.fit))
        }

        ## KEY: update V.est properly
        V.est = apply(T.est, 1, max)
        ## Store the trained learners
        names(T.stage) <- paste("A", K.grp, sep = ".")
        T.learners = c(T.learners, list(T.stage))
      }
      names(T.learners) <- paste("T", seq(n.stage, by = -1), sep = ".")
      if (verbose) {
        print(paste0(Sys.time(), ": T-learner training (GAM) done!!"))
      }
    }



    ######============  deC-learner  ============
    if ("deC" %in% metaLearners) {
      V.est = 0; deC.learners = list()
      for (stage in seq(n.stage, by = -1)) {
        if (verbose) {
          print(paste0(Sys.time(), " @Stage(GAM/deC-): ", stage))
        }

        # generate simplex coordinates
        k = length(A.list[[stage]])
        z = rep(1, k-1)
        e = diag(x = 1, k-1)
        W = cbind((k-1)^(-0.5) * z,  (k/(k-1))^(0.5)*e - z*(1+sqrt(k))/(k-1)^1.5)

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
        K.grp = sort(unique(A.tr))
        n.train = nrow(X.tr)
        V.est = V.est + Y.tr*weights[stage]

        ## train S-learner with glmnet:
        ## generate data set for S-learner:
        dat.train = data.frame(trt = A.tr, X.tr)
        dat.tmp = dat.train
        for (ii in K.grp) {
          tmp = data.frame(trt = ii, X.tr)
          dat.tmp = rbind(dat.tmp, tmp)
        }
        dat.tmp$trt = as.factor(dat.tmp$trt)
        dat.S = stats::model.matrix(~trt*.-1, dat.tmp)
        dat.train = dat.S[1:n.train,]; dat.test = dat.S[-(1:n.train),]
        if (is.null(S.learners)) {
          ## train by glmnet with LASSO penalty:
          S.fit = cv.glmnet(dat.train, V.est, family = "gaussian", parallel = TRUE, ...)
        } else {
          S.fit = S.learners[[n.stage - stage + 1]]
        }
        S.est = matrix(predict(S.fit, newx = dat.test, type = "response", s = "lambda.min"), ncol = length(K.grp), byrow = F)

        ## train T-learner with glmnet:
        T.est = NULL; T.stage = list();
        for (ii in K.grp) {
          if (is.null(T.learners)) {
            T.fit = cv.glmnet(stats::model.matrix(~.-1, X.tr)[A.tr==ii,], V.est[A.tr==ii], family = "gaussian", parallel = TRUE, ...)
          } else {
            T.fit <- T.learners[[n.stage - stage + 1]][[which(K.grp == ii)]]
          }
          T.est = cbind(T.est, predict(T.fit, newx = stats::model.matrix(~.-1, X.tr), type = "response", s = "lambda.min"))
        }
        ## obtain nuisance parameter h():
        h.hat = rowMeans(cbind(S.est, T.est))

        ## estimate optimal treatment via deC-learner:
        # transform data X into required shape:
        x.whole = stats::model.matrix(~., X.tr); trt = as.numeric(as.factor(A.tr))
        x.new = sapply(seq(n.train), function(i){
          as.vector(outer(W[,trt[i]], x.whole[i,]))
        })
        x.new = t(x.new)
        penalty_f = c(rep(0,k-1), rep(1, (ncol(x.whole)-1)*(k-1)))
        fit.tau  = cv.glmnet(x.new, V.est-h.hat, family = "gaussian", parallel = TRUE, maxit = 100000, penalty.factor = penalty_f, intercept=FALSE)
        ## estimated treatment effect:
        best.beta = stats::coef(fit.tau,s="lambda.min")
        best.beta = matrix(best.beta[-1], nrow = ncol(x.whole), byrow = T)
        deC.est   = x.whole %*% best.beta %*% W

        ## KEY: update V.est properly
        V.est = apply(deC.est, 1, max) + h.hat # i add h.hat back here

        ## Store the trained learners
        deC.learners = c(deC.learners, list(best.beta))
      }
      names(deC.learners) <- paste("deC", seq(n.stage, by = -1), sep = ".")
      if (verbose) {
        print(paste0(Sys.time(), ": deC-learner training (GAM) done!!"))
      }
    }
  }

  ######========== Prepare outputs
  DTRres <- list(S.learners = S.learners,
                 T.learners = T.learners,
                 deC.learners = deC.learners,
                 controls = list(
                   n.stage = n.stage,
                   A.list  = A.list,
                   baseLearner   = baseLearner[1],
                   metaLearners  = metaLearners,
                   include.X = include.X,
                   include.Y = include.Y,
                   include.A = include.A
                 )
  )
  class(DTRres) <- "metaDTR"
  return(DTRres)

}


