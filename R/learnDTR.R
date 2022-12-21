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
#' @param baseLearner Choose one baselearner for meta-learn er algorithms. So far supports \code{BART} and
#'                    \code{GAM} with splines and LASSO penalization.
#' @param metaLearners Meta-learner algorithms to learn the optimal DTR. To support more than two actions
#'                     at each stage, S-, A-, and deC-learner are available. But deC-learner only works when
#'                     \code{baseLearner = "GAM"} so far.
#' @param all.inclusive If \code{TRUE}, covariates adopted in training at stage \code{T} includes
#'                      \code{X[[t]], t<=T}, \code{A[[t]], t<T}, and \code{Y[[t]], t<T}. In other words,
#'                      \code{X} actually should only store additional covariates at each stage. Note that if
#'                      there are subjects dropping out during the study, it will cause error.
#' @details This function supports to find the optimal dynamic treatment regime (DTR) for either randomized experiments
#'          or observational studies. Also, thanks to meta-learner structure, S-, T-, and deC-learner can naturally
#'          support multiple action options at any stage.
#'
#' @return It includes learning results, basically, the trained functions that can be used for predictions. Since
#'         sequential recommendations might require intermediate observations, [learnDTR()] will not automatically
#'         provide prediction. But by another function [recommendDTR],
#'         predictions can flexibly been made stage by stage.
#' @import stats utils dbarts glmnet
#' @export
#' @seealso \code{\link{recommendDTR}}

learnDTR <- function(X, A, Y, weights = rep(1, length(X)),
                     baseLearner  = c("BART", "GAM"),
                     metaLearners = c("S", "T", "deC"),
                     all.inclusive = FALSE
) {
  n.stage = length(X)
  ######==================== check inputs ====================
  if (length(unique(length(X), length(A), length(Y))) != 1){
    stop("Inconsistent input of X, A, and Y. Please check!\n")
  }
  A.list = list() # stores actions available at each stage
  for (stage in seq(n.stage)) {
    A.list = c(A.list, list( sort(unique(A[[stage]])) ))
  }

  S.learners <- T.learners <- deC.learners <- NULL

  ######==================== bart as baselearner ====================
  if (baseLearner[1] == "BART") {
    ######============  S-learner  ============
    if ("S" %in% metaLearners) {
      V.est = 0; S.learners = list()
      for (stage in seq(n.stage, by = -1)) {
        n.train = nrow(X[[stage]])
        if (all.inclusive) {
          if (stage > 1) {
            X.tr = data.frame(A = matrix(unlist(A[1:(stage-1)]), ncol = stage-1, byrow = FALSE), # A
                              X = matrix(unlist(X[1:stage]), nrow = n.train, byrow = FALSE),     # X
                              Y = matrix(unlist(Y[1:(stage-1)]), ncol = stage-1, byrow = FALSE)  # Y
            )
            for (ii in seq(stage-1)) {
              X.tr[,ii] <- as.factor(X.tr[,ii])
            }
          } else {
            X.tr = X[[stage]]
          }

        } else {
          X.tr = X[[stage]]
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
        ## train by bart:
        S.fit = bart(x.train = dat.train, y.train = V.est, x.test = dat.test,
                     ntree = 200, keeptrees = TRUE, verbose = FALSE)
        S.est = matrix(colMeans(S.fit$yhat.test), ncol = length(K.grp), byrow = F)
        ## KEY: update V.est properly
        V.est = apply(S.est, 1, max)
        ## Store the trained learners
        S.learners = c(S.learners, list(S.fit))
      }
      names(S.learners) <- paste("S", seq(n.stage, by = -1), sep = ".")
    }

    ######============  T-learner  ============
    if ("T" %in% metaLearners) {
      V.est = 0; T.learners = list()
      for (stage in seq(n.stage, by = -1)) {
        if (all.inclusive) {
          if (stage > 1) {
            X.tr = data.frame(A = matrix(unlist(A[1:(stage-1)]), ncol = stage-1, byrow = FALSE), # A
                              X = matrix(unlist(X[1:stage]), nrow = n.train, byrow = FALSE),     # X
                              Y = matrix(unlist(Y[1:(stage-1)]), ncol = stage-1, byrow = FALSE)  # Y
            )
            for (ii in seq(stage-1)) {
              X.tr[,ii] <- as.factor(X.tr[,ii])
            }
          } else {
            X.tr = X[[stage]]
          }

        } else {
          X.tr = X[[stage]]
        }
        A.tr = A[[stage]]
        Y.tr = Y[[stage]]
        K.grp = sort(unique(A.tr))
        n.train = nrow(X.tr)
        V.est = V.est + Y.tr*weights[stage]
        ## directly train T-learner with bart:
        T.est = NULL; T.stage = list();
        for (ii in K.grp) {
          T.fit = bart(x.train = X.tr[A.tr==ii,], y.train = V.est[A.tr==ii], x.test = X.tr,
                       ntree = 200, keeptrees = TRUE, verbose = FALSE,
                       sigest = ifelse(as.numeric(table(A.tr)[paste(ii)]) < 1.05*ncol(X.tr), 1, NA))  # in case n<p
          T.est = cbind(T.est, colMeans(T.fit$yhat.test))
          T.stage = c(T.stage, list(T.fit))
        }
        ## KEY: update V.est properly
        V.est = apply(T.est, 1, max)
        ## Store the trained learners
        names(T.stage) <- paste("A", K.grp, sep = ".")
        T.learners = c(T.learners, list(T.stage))
      }
      names(T.learners) <- paste("T", seq(n.stage, by = -1), sep = ".")
    }
  }



  ######==================== GAM as baselearner ====================
  if (baseLearner[1] == "GAM") {

  }

  ######========== Prepare outputs
  DTRres <- list(S.learners = S.learners,
                 T.learners = T.learners,
                 deC.learners = deC.learners,
                 controls = list(
                   n.stage = n.stage,
                   A.list  = A.list,
                   baseLearner   = baseLearner,
                   metaLearners  = metaLearners,
                   all.inclusive = all.inclusive
                 )
  )
  class(DTRres) <- "metaDTR"
  return(DTRres)

}

