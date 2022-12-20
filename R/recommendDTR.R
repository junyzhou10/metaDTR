#' @title Recommend the optimal treatment at each stage
#' @author Junyi Zhou \email{junyzhou@iu.edu}
#' @description This function make recommendations of optimal treatment for a given subject at given stage
#' @param DTRs Output from [learnDTR()] that belong to class \code{metaDTR}.
#' @param X.new A list of covariates at each stage. In practice, since recommendations are made
#'              stage by stage, here one stage of \code{X.new} will provide one stage's recommendation of action.
#' @param A.new A list of observed or predicted actions (from the algorithm). Users can provide real actions observed
#'              rather than algorithm recommended to make this function more flexible.
#'              It allows to handle cases that the realized action/observation may not be consistent
#'              with algorithm results. Only needed when \code{all.inclusive = TRUE}.
#'              Default is \code{NULL}.
#' @param Y.new A list. Only required if \code{all.inclusive = TRUE} in [learnDTR()]. Default is \code{NULL}
#' @param start It allows predictions to start from middle stages (to save time),
#'              in which case \code{A.new} and \code{Y.new}
#'              should be properly provided if \code{all.inclusive = TRUE}.
#' @details This function make recommendations based on the trained learners from [learnDTR()] for new dataset.
#'
#' @return It returns the optimal action/treatment recommendations at each stage and results are stored in a list for
#'         each meta-learner method.
#' @export
#' @seealso \code{\link{learnDTR}}

recommendDTR <- function(DTRs, X.new,
                         A.new = NULL, Y.new = NULL,
                         start = 1) {
  n.stage <- DTRs$controls$n.stage
  all.inclusive <- DTRs$controls$all.inclusive
  baseLearner <- DTRs$controls$baseLearner
  metaLearners <- DTRs$controls$metaLearners
  A.list <- DTRs$controls$A.list
  n.step <- length(X.new)
  A.ind  <- ifelse(is.null(A.new), TRUE, FALSE)
  A.opt.S <- A.opt.T <- A.opt.deC <- list() # main output

  ######==================== check inputs ====================
  if (start > n.step) {
    stop("start stage exceeds the information provided. Please check!\n")
  }

  ######==================== bart as baselearner ====================
  if (baseLearner[1] == "BART") {
    ######============  S-learner  ============
    if ("S" %in% metaLearners) {
      for (stage in seq(start, n.step, by = 1)) { # forward now
        n.test = nrow(X.new[[stage]])
        if (all.inclusive) {
          if (stage > 1) {
            if (is.null(Y.new)) {
              stop("Missing Y.new given all.inclusive == TRUE! \n")
            }
            if (is.null(A.new)) {
              stop("Missing A.new given all.inclusive == TRUE! \n")
            }
            X.te = data.frame(A = matrix(unlist(A.new[1:(stage-1)]), ncol = stage-1, byrow = FALSE), # A
                              X = matrix(unlist(X.new[1:stage]), nrow = n.test, byrow = FALSE),     # X
                              Y = matrix(unlist(Y.new[1:(stage-1)]), ncol = stage-1, byrow = FALSE)  # Y
            )
            for (ii in seq(stage-1)) {
              X.te[,ii] <- as.factor(X.te[,ii])
            }
          } else {
            X.te = X.new[[stage]]
          }

        } else {
          X.te = X.new[[stage]]
        }
        ## predict outcome based on trained learners:
        Y.pred <- dat.tmp <- NULL
        for (ii in A.list[[stage]]) {
          dat.tmp = rbind(dat.tmp, data.frame(trt = ii, X.te))
        }
        dat.tmp$trt = as.factor(dat.tmp$trt)
        Y.pred = colMeans(predict(DTRs$S.learners[[n.stage - stage + 1]], newdata = stats::model.matrix(~trt*.-1, dat.tmp)))
        Y.pred = matrix(Y.pred, ncol = length(A.list[[stage]]), byrow = F)
        A.pred = A.list[[stage]][apply(Y.pred, 1, which.max)]
        A.opt.S = c(A.opt.S, list(A.pred))
        if (A.ind) {
          A.new <- A.opt.S
        }
      }
      names(A.opt.S) <- paste("Stage", seq(n.step, by = 1), sep = ".")
    }

    ######============  T-learner  ============
    if ("T" %in% metaLearners) {
      for (stage in seq(start, n.step, by = 1)) {
        n.test = nrow(X.new[[stage]])
        if (all.inclusive) {
          if (stage > 1) {
            if (is.null(Y.new)) {
              stop("Missing Y.new given all.inclusive == TRUE! \n")
            }
            if (is.null(A.new)) {
              stop("Missing A.new given all.inclusive == TRUE! \n")
            }
            X.te = data.frame(A = matrix(unlist(A.new[1:(stage-1)]), ncol = stage-1, byrow = FALSE), # A
                              X = matrix(unlist(X.new[1:stage]), nrow = n.test, byrow = FALSE),     # X
                              Y = matrix(unlist(Y.new[1:(stage-1)]), ncol = stage-1, byrow = FALSE)  # Y
            )
            for (ii in seq(stage-1)) {
              X.te[,ii] <- as.factor(X.te[,ii])
            }
          } else {
            X.te = X.new[[stage]]
          }

        } else {
          X.te = X.new[[stage]]
        }
        ## predict outcome based on trained learners:
        Y.pred <- NULL
        for (ii in seq(length(A.list[[stage]]))) {
          Y.pred = cbind(Y.pred, colMeans(predict(DTRs$T.learners[[n.stage - stage + 1]][[ii]], newdata = X.te)))
        }
        A.pred = A.list[[stage]][apply(Y.pred, 1, which.max)]
        A.opt.T = c(A.opt.T, list(A.pred))
        if (A.ind) {
          A.new <- A.opt.T
        }
      }
      names(A.opt.T) <- paste("Stage", seq(n.step, by = 1), sep = ".")
    }
  }



  ######==================== GAM as baselearner ====================
  if (baseLearner[1] == "GAM") {

  }

  ######========== Prepare outputs
  OptDTR <- list(A.opt.S = A.opt.S,
                 A.opt.T = A.opt.T,
                 A.opt.deC = A.opt.deC,
                 controls = list(
                   n.stage = n.stage,
                   A.list  = A.list,
                   baseLearner   = baseLearner,
                   metaLearners  = metaLearners,
                   all.inclusive = all.inclusive
                 )
  )
  class(OptDTR) <- "metaDTR"
  return(OptDTR)
}




