#' One simulated data of a three-stage SMART study
#'
#' A total of 400 subjects is simulated following idea in paper:
#' Zhao, Y.-Q., Zeng, D., Laber, E. B., and Kosorok, M. R. (2015), “New statistical learning methods for estimating optimal dynamic treatment regimes,” Journal of the American Statistical Association, Taylor & Francis, 110, 583–598.
#'
#' There are two actions at each stage. No intermediate reward is considered.
#'
#' \itemize{
#'   \item X List of covariates at each stage
#'   \item Y List of outcomes/rewards at each stage
#'   \item A List of observed actions at each stage
#' }
#' @docType data
#' @name TwoStg_Dat
#' @format A list of X, Y, and A. Each element further contains a list of three element for three stages.
#'
"TwoStg_Dat"
