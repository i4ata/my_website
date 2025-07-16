This is a basic visualization I developed for my Master's thesis. It was originally intended to be a debugging tool for my code but it turned out to be pretty useful. There will soon be a paper about it (hopefully)! Very basically, it is an extension of regular decision trees that incorporates evidence for confounding and causal relationships between the features. It is applied for regression and survival data.

The available models are trained on the following setups:

### Synthetic Datasets

Each synthetic dataset has a survival and regression variant and the first 6 setups have a variant with only binary variables and only continuous variables. The 7-th one is applied to only continuous variables. Also, except for it, the relationships between the variables and the outcome as well as the relationships between the variables themselves are only linear.

1. **Weak Independent & Strong Confounded Causes**: Here we have a variable that is a weak but independent cause of the outcome, a variable that is a strong but confounded cause of the outcome, and 2 variables that have no effect on the outcome but are significantly correlated with the second (confounded) variable.

2. **Confounded Unobserved Cause**: This setup is the same as the previous one, only the strong confounded cause is not observed.

3. **Nested Confounder**: Here we have 2 independent causes of the outcome, and a third variable, which has no effect and is correlated with the product of the two causes.

4. **Contextual Causal Relationship**: Here we have 2 variables, one has an independent cause of the outcome, and the effect of the other one is only present given that the first variable is positive (e.g., a disease is lethal for men but harmless for women).

5. **Double Contextual Relationship**: Here we have 3 variables, the first two are independent causes, while the effect of the third one is only present if both of the other variables are positive

6. **Random Variables**: In this trivial sanity-check setup, none of the available variables have an effect of the outcome.

7. **Non-linear relationship**: Here we have a variable that has an effect on the outcome only if it is less than -1 or greater than 1. Additionally, we have 2 variables that have no effect but are correlated with the first one.

### Public Datasets

The method has been tested on 6 clinical datasets, 4 for survival and 2 for regression data:

1. Survival: [NHANES](https://shap.readthedocs.io/en/latest/generated/shap.datasets.nhanesi.html#shap.datasets.nhanesi), [NACD](https://github.com/haiderstats/ISDEvaluation/blob/master/README.md), [METABRIC](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric), [SUPPORT2](https://hbiostat.org/data/)
2. Regression: [Obesity](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition), [Liver Disorders](https://archive.ics.uci.edu/dataset/60/liver+disorders)

## Method

The model is an extension of decision trees and is designed to incorporate evidence for confounding and causal relationships between the features. Basically, the best feature at each node is chosen by first stratifying the data based on all of the other features. That way if the variable is confounded by another one, its effect on the outcome will disappear after stratification For example, if we model mortality and a deadly disease is more prominent in women than in men, we would falsely conclude that gender has an effect on mortality, even though it's just that more women catch the disease. Stratifying by having the disease or not, we would find that the effect of gender is insignificant within the groups. Overall this provides more insight into the data, makes the models more explainable and trustworthy.

In survival modeling, the stratified logrank test is used to quantify the effect of a variable within multiple strata, while for regression, a mixed linear model with a single binary predictor is applied. There are 2 main issues, which are that splitting a variable by $n$ binary conditions results in $2^n$ groups, so we can quickly run out of samples when stratifying. To account for this, I do 2 things:

1. Stratify iteratively starting from the most independently associated features (e.g. if having an IPhone is not associated with mortality, we won't bother stratifying by that). That is done until some predefined minimum stratum size is reached.

2. Binarize continuous variables at each node using their strongest independent threshold. When stratifying, we would like to have a stratum for each unique value of a continuous variable (e.g. people aged 34, 35, 36, etc.). Instead, I use only the threshold at which the difference between the dependent variable in the two groups is the strongest (e.g. 'age' becomes 'age > 50').

Feel free to interact with the models. I will expand a bit further this page with technical details as there are some pretty cool optimization techniques.
