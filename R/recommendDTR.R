#' @title Recommend the optimal treatment at each stage
#' @author Junyi Zhou \email{junyzhou@iu.edu}
#' @description This function make recommendations of optimal treatment for a given subject at given stage
#' @param DTRs Output from [learnDTR()] that belong to class \code{metaDTR}.
#' @param currentDTRs Object from [recommendDTR()]. Records the results from previous stages so that users can
#'                    flexibly make predictions stage by stage. See details and example.
#' @param X.new A list of covariates at each stage. In practice, since recommendations are made
#'              stage by stage, here one stage of \code{X.new} will provide one stage's recommendation of action.
#' @param A.new A list of observed or predicted actions (from the algorithm). Users can provide real actions observed
#'              rather than algorithm recommended to make this function more flexible.
#'              It allows to handle cases that the realized action/observation may not be consistent
#'              with algorithm results. Only needed when \code{all.inclusive = TRUE}.
#'              Default is \code{NULL}.
#' @param Y.new A list. Required if \code{include.Y > 0} in [learnDTR()]. Default is \code{NULL}
#' @details This function make recommendations based on the trained learners from [learnDTR()] for new dataset.
#'          Since in real application, later stage covariates/outcomes are unobservable until treatments are
#'          assigned. So in most cases, [recommendDTR()] needs to be applied stage by stage, which is allowed in
#'          this package. User can provide available information at any stage and obtain the optimal recommendations
#'          at that stage.
#'
#' @return It returns the optimal action/treatment recommendations at each stage and results are stored in a list for
#'         each meta-learner method.
#' @examples
#' ## To see details of applying this function to obtain optimal DTR on new dataset, please
#' ##   refer to author's post: https://jzhou.org/posts/optdtr/
#' @export
#' @seealso \code{\link{learnDTR}}

recommendDTR <- function(DTRs, currentDTRs = NULL,
                         X.new, A.new = NULL, Y.new = NULL) {
  n.stage <- DTRs$controls$n.stage
  baseLearner <- DTRs$controls$baseLearner
  metaLearners <- DTRs$controls$metaLearners
  A.list <- DTRs$controls$A.list
  include.X <- DTRs$controls$include.X
  include.Y <- DTRs$controls$include.Y
  include.A <- DTRs$controls$include.A

  ######==================== check/arrange inputs ====================
  if (is.null(currentDTRs)) {
    if (class(X.new)[1] != "list") {
      X.new = list(X.new)
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
    A.opt.S <- currentDTRs$A.opt.S
    A.opt.T <- currentDTRs$A.opt.T
    A.opt.deC <- currentDTRs$A.opt.deC

    start = length(X.new) # start from middle stages
  }

  n.step <- length(X.new)
  if (start > n.step) {
    stop("start stage exceeds the information provided. Please check!\n")
  }



  ######==================== bart/gam as base learner ====================
  if (baseLearner %in% c("BART", "GAM")) {
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
            #   A.tmp[,ii] <- A.tmp[,ii]
            # }
            X.te = cbind(X.te, A.tmp)
          } else if (include.A == 2) {
            A.tmp = data.frame(A = matrix(unlist(A.new[1:(stage-1)]), nrow = n.test, byrow = FALSE))
            # for (ii in seq(ncol(A.tmp))) {
            #   A.tmp[,ii] <- A.tmp[,ii]
            # }
            X.te = cbind(X.te, A.tmp)
          }
          if (include.Y == 1) { # nothing need to do when include.Y == 0
            Y.tmp = data.frame(Y = Y.new[[stage-1]])
            X.te = cbind(X.te, Y.tmp)
          } else if (include.Y == 2) {
            Y.tmp = data.frame(Y = matrix(unlist(Y.new[1:(stage-1)]), nrow = n.test, byrow = FALSE))
            X.te = cbind(X.te, Y.tmp)
          }
        }

        ## predict outcome based on trained learners:
        Y.pred <- dat.tmp <- NULL
        for (ii in A.list[[stage]]) {
          dat.tmp = rbind(dat.tmp, data.frame(trt = ii, X.te))
        }
        dat.tmp$trt = as.factor(dat.tmp$trt)
        if (baseLearner == "BART") {
          Y.pred = colMeans(predict(DTRs$S.learners[[n.stage - stage + 1]], newdata = stats::model.matrix(~trt*.-1, dat.tmp)))
          Y.pred = matrix(Y.pred, ncol = length(A.list[[stage]]), byrow = F)
        }
        if (baseLearner == "GAM") {
          Y.pred = matrix(predict(DTRs$S.learners[[n.stage - stage + 1]], newx = stats::model.matrix(~trt*.-1, dat.tmp),
                                 type = "response", s = "lambda.min"), ncol = length(A.list[[stage]]), byrow = F)
        }
        A.pred = A.list[[stage]][apply(Y.pred, 1, which.max)]
        A.opt.S = c(A.opt.S, list(A.pred))
        A.new <- c(A.new, list(A.pred))
      }
      names(A.opt.S) <- c(names(currentDTRs$A.opt.S), paste("Stage", seq(start, n.step, by = 1), sep = "."))
    }

    ######============  T-learner  ============
    if ("T" %in% metaLearners) {
      for (stage in seq(start, n.step, by = 1)) {
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
          A.new <- A.opt.T
          A.new[A.ind] <- A.obs[A.ind]
          if (include.A == 1) { # nothing need to do when include.A == 0
            A.tmp = data.frame(A = A.new[[stage-1]])
            X.te = cbind(X.te, A.tmp)
          } else if (include.A == 2) {
            A.tmp = data.frame(A = matrix(unlist(A.new[1:(stage-1)]), nrow = n.test, byrow = FALSE))
            X.te = cbind(X.te, A.tmp)
          }
          if (include.Y == 1) { # nothing need to do when include.Y == 0
            Y.tmp = data.frame(Y = Y.new[[stage-1]])
            X.te = cbind(X.te, Y.tmp)
          } else if (include.Y == 2) {
            Y.tmp = data.frame(Y = matrix(unlist(Y.new[1:(stage-1)]), nrow = n.test, byrow = FALSE))
            X.te = cbind(X.te, Y.tmp)
          }
        }

        ## predict outcome based on trained learners:
        Y.pred <- NULL
        for (ii in seq(length(A.list[[stage]]))) {
          if (baseLearner == "GAM") {
            Y.pred = cbind(Y.pred, predict(DTRs$T.learners[[n.stage - stage + 1]][[ii]], newx = stats::model.matrix(~.-1, X.te),
                                           type = "response", s = "lambda.min"))
          }
          if (baseLearner == "BART") {
            Y.pred = cbind(Y.pred, colMeans(predict(DTRs$T.learners[[n.stage - stage + 1]][[ii]], newdata = X.te)))
          }
        }
        A.pred = A.list[[stage]][apply(Y.pred, 1, which.max)]
        A.opt.T = c(A.opt.T, list(A.pred))
        A.new <- c(A.new, list(A.pred))
      }
      names(A.opt.T) <- c(names(currentDTRs$A.opt.T), paste("Stage", seq(start, n.step, by = 1), sep = "."))
    }



    ######============  deC-learner  ============
    if ("deC" %in% metaLearners) {
      for (stage in seq(start, n.step, by = 1)) {
        n.test = nrow(X.new[[stage]])
        # generate simplex coordinates
        k = length(A.list[[stage]])
        z = rep(1, k-1)
        e = diag(x = 1, k-1)
        W = cbind((k-1)^(-0.5) * z,  (k/(k-1))^(0.5)*e - z*(1+sqrt(k))/(k-1)^1.5)

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
          A.new <- A.opt.deC
          A.new[A.ind] <- A.obs[A.ind]
          if (include.A == 1) { # nothing need to do when include.A == 0
            A.tmp = data.frame(A = A.new[[stage-1]])
            X.te = cbind(X.te, A.tmp)
          } else if (include.A == 2) {
            A.tmp = data.frame(A = matrix(unlist(A.new[1:(stage-1)]), nrow = n.test, byrow = FALSE))
            X.te = cbind(X.te, A.tmp)
          }
          if (include.Y == 1) { # nothing need to do when include.Y == 0
            Y.tmp = data.frame(Y = Y.new[[stage-1]])
            X.te = cbind(X.te, Y.tmp)
          } else if (include.Y == 2) {
            Y.tmp = data.frame(Y = matrix(unlist(Y.new[1:(stage-1)]), nrow = n.test, byrow = FALSE))
            X.te = cbind(X.te, Y.tmp)
          }
        }

        ## predict outcome based on trained betas:
        Y.pred = stats::model.matrix(~., X.te) %*% DTRs$deC.learners[[n.stage - stage + 1]] %*% W
        A.pred = A.list[[stage]][apply(Y.pred, 1, which.max)]
        A.opt.deC = c(A.opt.deC, list(A.pred))
        A.new <- c(A.new, list(A.pred))
      }
      names(A.opt.deC) <- c(names(currentDTRs$A.opt.deC), paste("Stage", seq(start, n.step, by = 1), sep = "."))
    }



  }


  ######========== Prepare outputs
  OptDTR <- list(A.opt.S = A.opt.S,
                 A.opt.T = A.opt.T,
                 A.opt.deC = A.opt.deC,
                 controls = list(
                   X.new = X.new,
                   A.obs = A.obs,
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




