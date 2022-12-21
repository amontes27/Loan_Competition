# Loan_Competition
Python code and process used to generate submissions to 2 Kaggle competitions using bank loan repayment data

Links:
[Regression](https://www.kaggle.com/competitions/data-sci-3-reg-2022-bank-loans),
[Classification](https://www.kaggle.com/competitions/data-sci-3-classification-2022-loan-repayment)

---
# Overview
- Performed feature engineering and selection using summary statistics, visualizations, and model feature importances
- Tuned MARS, decision tree, and boosted models to generate regression and classification ensembles
- Placed 12th and 34th of 178 participants in the regression and classification components, respectively

---
# Detailed Process
## Regression Problem (see [code](Regression_Competition_Code.ipynb))
### Data Analysis and Cleaning
I started my model-building process by concatenating the train and test data and creating histograms, boxplots, and catplots to visualize the distribution of each predictor in the dataset. For some categorical predictors with many levels such as addr_state and emp_title, I created an “other” category for levels with relatively few occurrences. I then visualized the relationship between each predictor and money_made_inv using scatterplots and boxplots.

### Variable Selection
After building base models (MARS and decision tree) containing all the predictors in the data to serve as a performance baseline, I used the visualizations to drop predictors with low variance and response correlations. I then built and tuned decision tree, bagging, random forest, and boosted models and found the feature importances for each one. This allowed me to drop unimportant predictors present in each model. I repeated this process with the new data until I was left with a small subset of the most important predictors that significantly improved the base models’ performance. 

### Final Model Tuning
For a single model approach, I found that using K-fold cross-validation to tune a MARS model with 500 max_terms and a max_degree of 4 produced the best results in my predictions. After this, I explored additional model types and ensemble methods to improve it. Overall, stacking the MARS model with a pruned decision tree provided the best submission results. This was likely because a pruned tree provided a simple model that balanced out the more complex interactions between predictors in the MARS model, creating more accurate predictions.


## Classification Problem (see [code](Classification_Competition_Code.ipynb))
### Data Analysis and Cleaning
I used the same methods as in the regression problem to visualize the predictors’ distributions and consolidate categorical variable levels. I also visualized each predictor’s distribution for both categories of hi_int_prncp_pd with scatterplots and boxplots. 

### Variable Selection
Much like the regression problem, I used the visualizations to drop predictors with low variance within their distributions and across each category of hi_int_prncp_pd. I then created various types of models and calculated the feature importance of each one to drop the least important predictors over several iterations. I also found that the accuracy of a decision tree base model increased as I removed predictors.

### Final Model Tuning
Similar to the regression problem, I tuned a variety of models and tried different ensemble methods to combine them. Overall, boosting approaches generated the best single-model performance. My final model combined XGBoost and random forest models in a soft voting ensemble to produce the most accurate predictions. The random forest model likely helped reduce any overfitting that occurred in the XGBoost model by creating many de-correlated trees.
