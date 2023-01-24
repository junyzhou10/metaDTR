#' One simulated data of a three-stage SMART study
#'
#' A total of 400 subjects for training and 1000 subjects for testing are simulated following idea in paper:
#' Zhao, Y.-Q., Zeng, D., Laber, E. B., and Kosorok, M. R. (2015), “New statistical learning methods for estimating optimal dynamic treatment regimes,” Journal of the American Statistical Association, Taylor & Francis, 110, 583–598.
#'
#' There are two actions at each stage. No intermediate reward is considered.
#'
#' \itemize{
#'   \item X List of covariates at each stage
#'   \item Y List of outcomes/rewards at each stage
#'   \item A List of observed actions at each stage
#'   \item X.test Test dataset
#' }
#' @docType data
#' @name ThreeStg_Dat
#' @format A list of X, Y, and A. Each element further contains a list of three element for three stages.
#'
"ThreeStg_Dat"

# ## ThreeStg_Dat generation code:
# set.seed(54321)
# n.sbj = 400; n.test = 1000; sigma = 1
# n.stage = 3
# nX1 = 10; nX2 <- nX3 <- 5
# ## Treatment Assignments A at each stage:
# A.tr = matrix(rbinom(n.sbj*n.stage,1,0.5), ncol = n.stage);
# ## Observed covariates X before each stage:
# X1.tr = matrix(rnorm(n.sbj*nX1, 45, 15), ncol = nX1)
# X2.tr = matrix(rnorm(n.sbj*nX2, X1.tr[,1:nX2], 10), ncol = nX2)
# X3.tr = matrix(rnorm(n.sbj*nX3, X2.tr[,1:nX3], 10), ncol = nX3)
# ## Outcome/Reward generator:
# y.fun <- function(X1, X2, X3, A) {
#   20 - abs(0.6*X1[,1])*abs(A[,1] - (X1[,1]>30)) -
#     abs(0.8*X2[,1] - 60)*abs(A[,2] - (X2[,1]>40)) -
#     abs(1.4*X3[,1] - 40)*abs(A[,3] - (X3[,1]>40))
# }
# Y.tr = y.fun(X1.tr, X2.tr, X3.tr, A.tr) + rnorm(n.sbj,0,sigma)
# ## test data:
# X1.te = matrix(rnorm(n.test*nX1, 45, 15), ncol = nX1)
# X2.te = matrix(rnorm(n.test*nX2, X1.tr[,1:nX2],10), ncol = nX2)
# X3.te = matrix(rnorm(n.test*nX3, X2.tr[,1:nX3],10), ncol = nX3)
# X = list(X1.tr, X2.tr, X3.tr)
# X.test = list(X1.te, X2.te, X3.te)
# A = list(as.factor(A.tr[,1]), as.factor(A.tr[,2]), as.factor(A.tr[,3]))
# Y = list(rep(0,n.sbj), rep(0,n.sbj), Y.tr)
# ThreeStg_Dat = list(X = X, X.test = X.test, Y = Y, A = A)
