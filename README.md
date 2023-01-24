# metaDTR
Learning Dynamic Treatment Regime (DTR) via meta-learners. It can learn from studies with multi-stage interventions as well as more than two treatment arms at each intervention. 

# Method
The package supports learning from multi-stage and multi-armed randomized experiments or observed studies, and then recommend personalized sequence of treatments/interventions. Meta-learners are adopted in training stages: there are S- and T-learner from Q-learning framework and deC-learner (author proposed, not published yet) from A-learning camp.

Please find method details on the author's [post](https://jzhou.org/posts/optdtr/). 

# Usage
To install the package:
```
devtools::install_github("junyzhou10/metaDTR")
```

The package currently supports S-, T-, and deC-learner. So far, [BART (Chipman 2010)](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full) and [GAM (Hastie 2017)](https://www.taylorfrancis.com/chapters/edit/10.1201/9780203738535-7/generalized-additive-models-trevor-hastie) is available as baselearner. 
There is an example of code in the appendix of author's [post](https://jzhou.org/posts/optdtr/).

# Limitations
So far, base learner only supports BART, random forest (RF), and GAM. For meta-learners, now supports S-, T-, and deC-learner. X-learner will not be included because it is not staightforward in multi-armed cases, and not suitable for outcome types other than continuous. Details of de-centralized learner (deC-learner) will be available after publication.

Also, only continuous outcome type is allowed at this point. Incorporating binary endpoints with log odds ratio as causal estimand can the next step of work. 

# Lastest Update: 1/23/2023
- Add random forest as baselearner. Note that RF is suggested to use only when sample size is larger enough, or persuing numerical efficiency. Otherwise, BART is more desirable as baselearner.
