# metaDTR
Learning Dynamic Treatment Regime (DTR) via meta-learners

# Method
The package supports learning from multi-stage and multi-armed randomized experiments or observed studies, and then make personalized treatment sequence recommendations.
Please find method details on the author's [post](https://jzhou.org/posts/optdtr/). 

# Usage
The package currently supports S- and T-learner. So far, [BART (Chipman 2010)](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full) is available as baselearner. 
There is an example of code in the appendix of [author's post](https://jzhou.org/posts/optdtr/).

# Limitations
So far, base learner only supports BART. GAM is under development. For meta-learners, now supports S- and T-learner. X-learner will not be included because it is not staightforward in multi-armed cases, and not suitable for outcome types other than continuous. de-centralized learner (deC-learner) is under development now.

Also, only continuous outcome type is allowed. 
