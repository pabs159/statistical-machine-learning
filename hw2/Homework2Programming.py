#!/usr/bin/env python
# coding: utf-8

# # <p style="text-align: center;">EE 381V: Statistical Machine Learning</p>
# # <p style="text-align: center;">Homework 2: Programming Assignments</p>
# ## <p style="text-align: center;">Total points: 65  </p>
# ## <p style="text-align: center;">Due: March 8 by 11:59 pm (submission via Gradescope)</p>
# 
# Ideally, your solution to the assignments should be written in and submitted as a **Jupyter notebook**. Please make sure your code runs and the graphics (along with anything else that you want to be considered) are displayed in your notebook before submitting.
# 
# For the theoretical parts of the questions below (e.g., computation of gradients), the most convenient approach is to type the solution in the provided spaces ("Markdown" cells) using LaTeX (if not familiar with LaTeX, please check the markdown cells below stating questions for examples of writing equations in LaTeX). Alternatively, you could write down the solution on paper and submit a pdf file of the scan/photo.

# # <font color='blue'> Question 1: Stochastic Gradient Descent (20 pts) </font>
#     
# In class, we introduced gradient descent and its stochastic variant. In this problem, we will explore the latter. Assume that we want to predict values of two-dimensional feature vectors, i.e., we are given a dataset $\{({\bf x}_1,y_1),\dots,({\bf x}_N,y_N)\}$ of features ${\bf x}_i = [x_{i1} \;\; x_{i2}]^T$ and their corresponding values $y_i$, and would like to predict value $y$ of a never before seen ${\bf x} = [x_{1} \;\; x_{2}]^T$.
# 
# Good refrence [HERE](https://towardsdatascience.com/step-by-step-tutorial-on-linear-regression-with-stochastic-gradient-descent-1d35b088a843)
# 
#  1. (5 pts) Assuming that we want to learn the coefficients via stochastic gradient descent, derive the update rules for all 4 coefficients (i.e., $w_0$, $w_1$, $w_2$ and $w_3$) of the model 
# $$y = w_0 + w_1x_1 + w_2x_1x_2 + w_3e^{-x_1}.$$
# 
#  2. (15 pts) Write a Python code that implements the SGD rules found in part (a) to train the non-linear model $$ y = w_0 + w_1x_1 + w_2x_1x_2 + w_3e^{-x_1}.$$ Attempt to format similarly to scikit-learn's models. The template of the class is given. The init function of the class takes as input the learning rate, regularization constant and number of epochs. The fit method must take as inputs X and y. The _predict_ method takes an X value (optionally, an array of values). Use your new gradient descent regression to train on data given in "SGD_samples.csv" for 15 epochs, using learning rates [0, .0001, .001, .01, 0.1, 1, 10, 100] and regularization (ridge regression) constants [0,10,100]. Plot the mean-square error (MSE) and $w$ parameters as functions of epoch (for 15 epochs) for the best 2 combinations of the learning_rate and regularization for SGD. Report the MSE at the end of 15 epochs for the two best combinations.
# 

# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Regression:
    
    def __init__(self, learning_rate, regularization, n_epoch):
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.regularization = regularization
        self.coef = None
        
    def sgd(self, gradient):
        self.coef -= self.learning_rate * gradient
        #print(self.coef)
    
    def fit(self, X, y, update_rule='sgd', plot=False):
        mse = []
        coefs = np.zeros((self.n_epoch, X.shape[0], 4))
        #coefs = np.zeros((15, 4))
        #coefs = []
        X = self.get_features(X)
        self.coef = np.random.rand(X.shape[1])  # Initialize coefficients
        for epoch in range(self.n_epoch):
            for i in range(X.shape[0]):
                # Compute error
                y_pred = self.linearPredict(X[i])
                #print(y_pred.shape)
                error = y_pred - y[i]
                # Compute gradients
                #gradient = 2 * np.dot(X[i], error)
                gradient = 2 * np.dot(X[i], error) * (self.regularization / X.shape[0])
                #print(gradient)
                # Update weights
                #print(self.learning_rate)
                self.sgd(gradient)
                coefs[epoch,i,:] = self.coef

                
            #coefs.append(self.coef)
            #print(coefs)
            residuals = y - self.linearPredict(X)
            mse.append(np.mean(residuals**2))
        self.lowest_mse = mse[-1]
        if plot == True:
            plt.figure()
            plt.plot(range(self.n_epoch), mse)
            plt.xlabel('epoch')
            plt.ylabel('MSE')
            plt.figure()
            #print(coefs)
            #coefs = np.array(coefs)
            coefs = coefs.mean(axis=1)
            #print(f"after: {coefs}")
            plt.plot(range(self.n_epoch), coefs[:,0], label='w0')
            plt.plot(range(self.n_epoch), coefs[:,1], label='w1')
            plt.plot(range(self.n_epoch), coefs[:,2], label='w2')
            plt.plot(range(self.n_epoch), coefs[:,3], label='w3')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('parameter value')
            plt.show()

    def get_features(self, X):
        '''
        this output of this function can be used to compute the gradient in `fit`
        '''
        x = np.zeros((X.shape[0], 4))
        x[:,0] = 1
        x[:,1] = X[:,0]
        x[:,2] = X[:,0] * X[:,1]
        x[:,3] = np.exp(-X[:,0])
        return x
        
    def linearPredict(self, X):  
        return np.dot(X, self.coef)


# In[34]:


data = pd.read_csv('SGD_samples.csv')
X = np.array([data['x1'].values, data['x2'].values]).T
y = data['y'].values
n_epochs = 1
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
#learning_rates = [1e-5]
regularization_constants = [0, 10, 100]
#regularization_constants = [0]


# ## Answers

# ### 1. Derive the coefficent updates for all 4 coefficients of the model (use Markdown to type) (5 pts)

# ![alt text](3-1.png "Title")

# ### 2. Write the code here (10pts)

# In[35]:


# Train with different parameters
results = {}
for lr in learning_rates:
    for reg in regularization_constants:
        model = Regression(lr, reg, 15)
        model.fit(X, y)
        results[(lr, reg)] = model.lowest_mse
        print(model.lowest_mse)



# ### 3. Explore running the regression with different learning rates and regularization (use the class 'Regression' you completed in the previous section). Plot the required results. (5 pts)
# 

# In[36]:


# Find the two best combinations
best_combinations = sorted(results.items(), key=lambda x: x[1])[:5]
best_combination_mse = {k: v for k, v in best_combinations}

# Plot the MSE and parameters for the two best combinations
for combination, mse in best_combination_mse.items():
    lr, reg = combination
    print(f"Learning rate: {lr}, Regularization constant: {reg}, Final MSE: {mse}")

    # Plot MSE and parameters
    model = Regression(lr, reg, 15)
    model.fit(X, y, plot=True)


# # <font color='blue'> Question 2: Regularized regression (25 pts)</font>
# In this question, we will explore the applications of ridge (i.e., $l_2$-norm) and lasso ($l_1$-norm) regularized regression using sklearn package in Python. To this end, we will use the dataset Fish.csv (posted to Canvas). The dataset is taken from https://www.kaggle.com/aungpyaeap/fish-market and includes 7 columns with 6 features (variables) of the fish and 1 target (label):
# 
# x1: Species: Species name of fish (nonnumeric)\
# x2: Length1: Vertical length in cm\
# x3: Length2: Diagonal length in cm\
# x4: Length3: Cross length in cm\
# x5: Height: Height in cm\
# x6: Width: Diagonal width in cm
# 
# y: Weight of the fish in grams
# 
# We want to learn the function $y = w_1x_1+w_2x_2+w_3x_3+w_4x_4+w_5x_5+w_6x_6$ to predict the weight of a fish from the features.
# 
# * Use the below code to load the dataset. Create a train_test split of 75:25 with random state = 50
# 
# * Scale the data so that each of the independent variables has zero mean and unit variance. You can use  [sklearn.preprocessing.scale](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html) function for this.

# In[19]:


from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


fish_data = pd.read_csv("Fish.csv", index_col=0)
#df = df[~df.isin([0, np.nan, np.inf, -np.inf]).any(axis=1)]
X = fish_data.drop(columns=['Weight'], axis=1)
y = fish_data['Weight']
#X = df.drop(['Weight'], axis=1)
#y = df['Weight']


# a) Use sklearn.linear_model.Lasso and sklearn.linear_model.Ridge classes to run [5-fold cross validation](http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#example-exercises-plot-cv-diabetes-py) using sklearn's [KFold](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html). For the sweep of the regularization parameter, we will look at a grid of values ranging from $\lambda = 10^{10}$ to $\lambda = 10^{-2}$. In Python, you can create this range of values as follows:
# 
#       import numpy as np
#       alphas =  10**np.linspace(10,-2,100)*0.5
# 
# Report the best $\lambda$ based on cross validation. The cross validation should happen on your training data using average MSE as the scoring metric. (5pts)
# 
# b) Run ridge and lasso regression for all alphas specified above (on training data), and plot the coefficients learned for each of them - there should be one figure each for lasso and ridge, so a total of two figures; the results for different features should be plotted in the same figure. What do you qualitatively observe as the value of the regularization parameter is changed? (5pts)
# 
# c) Run the (unconstrained) linear regression and its regularized variants (both ridge and lasso) on the training data. For ridge and lasso, use only the best regularization parameter. Report the prediction error (MAE) and the mean-squared error (MSE) on the test data for each method. (3pts)
# 
# d) Run lasso again with cross validation using [sklearn.linear_model.LassoCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html). Set the cross-validation parameters as follows:
# 
#     LassoCV(alphas=None, cv=10, max_iter=10000)
# 
# Report the best $\lambda$ based on cross-validation. Run lasso on the training data using the best $\lambda$ and report the coefficients for all variables. (3pts)
# 
# e) Why did we have to scale the data before regularization? (3pts)
# 
# f) Lasso and ridge regularization techniques are often used to combat overfitting during linear regression. Which of the two yields sparser models (i.e., fewer parameters) when the tuning parameter $\lambda$ is large (but not infinite)? (3pts)
# 
# g) [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) is a type of a regression model that uses combined $l_1$ and $l_2$ priors as regularizers. Run ElasticNet with the same values of alphas on the training data, and find the best value for alpha based on the MSE. Report the MSE on test data and plot a graph showing the predicted and actual values/labels. Comment/compare the results of the three regression models. (3pts)

# ## Answers

# ### a. (5 pts)

# In[20]:


import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import scale
from sklearn.linear_model import Lasso, Ridge, LassoCV
from sklearn.metrics import mean_squared_error

alphas =  10**np.linspace(10,-2,100)*0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)

# Scale the data
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

# Define KFold
kf = KFold(n_splits=5)

# Lasso
lasso_cv_mses = []
lasso_best_alpha = None
lasso_best_mse = float('inf')

# Ridge
ridge_cv_mses = []
ridge_best_alpha = None
ridge_best_mse = float('inf')
warnings.simplefilter('ignore', category=ConvergenceWarning)
for alpha in alphas:
    mse_sum_lasso = 0
    mse_sum_ridge = 0
    for train_index, val_index in kf.split(X_train_scaled):
        X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # Lasso
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train_fold, y_train_fold)
        y_val_pred = lasso.predict(X_val_fold)
        
        mse_lasso = mean_squared_error(y_val_fold, y_val_pred)
        mse_sum_lasso += mse_lasso
        avg_mse_lasso = mse_sum_lasso / kf.get_n_splits()
        lasso_cv_mses.append(avg_mse_lasso)

        # Ridged 
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_fold, y_train_fold)
        y_val_pred = ridge.predict(X_val_fold)
        
        mse_ridge = mean_squared_error(y_val_fold, y_val_pred)
        mse_sum_ridge += mse_ridge
        avg_mse_ridge = mse_sum_ridge / kf.get_n_splits()
        ridge_cv_mses.append(avg_mse_ridge)

    if avg_mse_lasso < lasso_best_mse:
        lasso_best_mse = avg_mse_lasso
        lasso_best_alpha = alpha
    if avg_mse_ridge < lasso_best_mse:
        ridge_best_mse = avg_mse_ridge
        ridge_best_alpha = alpha

print("Best alpha for Lasso:", lasso_best_alpha)
print("Best alpha for Ridge:", ridge_best_alpha)


# ### b. (5 pts)

# In[21]:


import matplotlib.pyplot as plt

# Ridge regression
ridge_coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_coefs.append(ridge.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_coefs)
plt.title('Ridge Coefficients vs. Alpha')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.xscale('log')
plt.legend(X.columns)
plt.show()

# Lasso regression
lasso_coefs = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)
    lasso_coefs.append(lasso.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alphas, lasso_coefs)
plt.title('Lasso Coefficients vs. Alpha')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.xscale('log')
plt.legend(X.columns)
plt.show()


# ### c. (3 pts)

# In[22]:


# Linear Regression
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)
linear_reg_pred = linear_reg.predict(X_test_scaled)

linear_reg_mae = mean_absolute_error(y_test, linear_reg_pred)
linear_reg_mse = mean_squared_error(y_test, linear_reg_pred)

print("Linear Regression MAE:", linear_reg_mae)
print("Linear Regression MSE:", linear_reg_mse)

# Ridge Regression with best alpha
ridge_best = Ridge(alpha=ridge_best_alpha)
ridge_best.fit(X_train_scaled, y_train)
ridge_pred = ridge_best.predict(X_test_scaled)

ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)

print("Ridge Regression MAE:", ridge_mae)
print("Ridge Regression MSE:", ridge_mse)

# Lasso Regression with best alpha
lasso_best = Lasso(alpha=lasso_best_alpha)
lasso_best.fit(X_train_scaled, y_train)
lasso_pred = lasso_best.predict(X_test_scaled)

lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)

print("Lasso Regression MAE:", lasso_mae)
print("Lasso Regression MSE:", lasso_mse)


# ### d. (3 pts)

# In[23]:


lasso_cv = LassoCV(alphas=None, cv=10, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print("Best alpha for LassoCV:", lasso_cv.alpha_)
print("Coefficients for all variables:", lasso_cv.coef_)


# ### e. (3 pts)

# Scaling the data before regularization is important because regularization penalizes the size of coefficients. If the features are not on the same scale, then the regularization term might penalize larger coefficients more, leading to bias in the model. Scaling the data ensures that all features contribute equally to the regularization term.

# ### f. (3 pts)

# Lasso regularization yields sparser models when the tuning parameter $\lambda$ is large (but not infinite). This is because Lasso regularization tends to force the coefficients of less important features to zero, effectively removing them from the model. Ridge regularization, on the other hand, only shrinks the coefficients towards zero but doesn't typically set them exactly to zero, so it doesn't result in sparsity as Lasso does.

# ### g. (3 pts)

# In[24]:


from sklearn.linear_model import ElasticNet

elasticnet_mses = []
best_elasticnet_alpha = None
best_elasticnet_mse = float('inf')

for alpha in alphas:
    elasticnet = ElasticNet(alpha=alpha)
    elasticnet.fit(X_train_scaled, y_train)
    elasticnet_pred = elasticnet.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, elasticnet_pred)
    elasticnet_mses.append(mse)
    
    if mse < best_elasticnet_mse:
        best_elasticnet_mse = mse
        best_elasticnet_alpha = alpha

print("Best alpha for ElasticNet:", best_elasticnet_alpha)
print("MSE on test data for ElasticNet:", best_elasticnet_mse)

# Plot predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, elasticnet_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('ElasticNet: Predicted vs Actual')
plt.show()


# Comparing the results of the three regression models, we observe that ElasticNet combines the penalties of Lasso and Ridge, providing a balance between variable selection (sparsity) and coefficient shrinkage. It often performs well when there are correlated predictors, as it can select groups of correlated variables together. In this specific case, it may perform similarly or slightly better than Ridge or Lasso alone, depending on the dataset and the strength of correlations between features.

# # <font color='blue'> Question 3: Feature Selection (20 pts)</font>
# 
# In this question, we will explore the importance of feature selection. You may use the code below to load the dataset "zoo.csv" which contains a set of features for different animals and the output variable ("class_type"). 
# 
# 1. Show the correlation matrix for this dataset; its dimension should be n_features x n_features. List the top 5 most positively correlated features with class_type. (**3pts**)
# 
# 2. List top-5 most negatively correlated features with class_type. (**3pts**)
# 
# 3. Use https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE to perform feature selection with estimator = LogisticRegression(max_iter=1000). List the top 5 features selected by RFE. (**2pts**)
# 
# To proceed, create a train_test split of 80:20 (training:test) with random state = 50. As part of the preprocessing, we should use [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): Instead of fitting a model to the original data, we use StandardScaler to first center each feature ([Example](http://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html#sphx-glr-auto-examples-applications-plot-prediction-latency-py)). Remember that when dealing with training and testing data, we fit preprocessing parameters using training data and apply them to testing data. You should scale only the features (independent variables), not the target variable y.
# 
# 4. Train a [Linear Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) using training data and report the ${R^2}$ score on the test set for each of the following:
# 
#    a) Model trained with top-5 most positvely correlated features from part 1. (**2pts**)
#    
#    b) Model trained with top-5 most negatively correlated features from part 2. (**2pts**)
#    
#    c) Model trained with five features selected by RFE in part 3. (**2pts**)
#    
#    d) Finally, train a linear regressor on the entire training data using all the original features. (**2pts**)
# 
# 
# 5. Comment on your results.  (**4pts**) 

# In[26]:


import pandas as pd
from sklearn.model_selection import (train_test_split,KFold)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("zoo.csv")
df = df.drop(['animal_name'], axis = 1)
meta_val = (df['class_type'] <= 4) + 0
df['class_type']=meta_val

y = df['class_type'].values
X = df.drop(['class_type'], axis = 1)

from sklearn import preprocessing
#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state=50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)


df_corr = pd.DataFrame(X_train)
df_corr['class_type'] = y_train


# In[27]:


df.head()


# In[29]:


# Display correlation matrix
correlation_matrix = df_corr.corr()
#print("Correlation Matrix:")
#print(correlation_matrix)

# Step 2: Find the top 5 most positively correlated and negatively correlated features with class_type
corr_with_class = correlation_matrix['class_type'].drop('class_type')
top_positively_correlated = corr_with_class.nlargest(5)
top_negatively_correlated = corr_with_class.nsmallest(5)

print("\nTop 5 most positively correlated features with class_type:")
print(top_positively_correlated)
print("\nTop 5 most negatively correlated features with class_type:")
print(top_negatively_correlated)

# Step 3: Feature selection using RFE with Logistic Regression
estimator = LogisticRegression(max_iter=1000)
selector = RFE(estimator, n_features_to_select=5)
selector = selector.fit(X, y)
selected_features_rfe = X.columns[selector.support_]

print("\nTop 5 features selected by RFE:")
print(selected_features_rfe)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Linear Regression models using different feature sets
# Step 6: Evaluate the models on the test set

# Model trained with top-5 most positively correlated features
lr_pos = LinearRegression()
lr_pos.fit(X_train[top_positively_correlated.index], y_train)
score_pos = lr_pos.score(X_test[top_positively_correlated.index], y_test)
print("\nScore on test set with top-5 most positively correlated features:", score_pos)

# Model trained with top-5 most negatively correlated features
lr_neg = LinearRegression()
lr_neg.fit(X_train[top_negatively_correlated.index], y_train)
score_neg = lr_neg.score(X_test[top_negatively_correlated.index], y_test)
print("Score on test set with top-5 most negatively correlated features:", score_neg)

# Model trained with features selected by RFE
lr_rfe = LinearRegression()
lr_rfe.fit(X_train[selected_features_rfe], y_train)
score_rfe = lr_rfe.score(X_test[selected_features_rfe], y_test)
print("Score on test set with features selected by RFE:", score_rfe)

# Model trained with all original features
lr_all = LinearRegression()
lr_all.fit(X_train_scaled, y_train)
score_all = lr_all.score(X_test_scaled, y_test)
print("Score on test set with all original features:", score_all)


# Comments on the results
The model trained with the top-5 most positively correlated features performs better than others, indicating the importance of these features. Interestingly, the model trained with the top-5 most negatively correlated features also performs relatively well, suggesting a non-linear relationship between these features and the target.
The model trained with features selected by RFE performs competitively, indicating the effectiveness of RFE in selecting relevant features.
The model trained with all original features performs similarly to the model trained with RFE-selected features, indicating that RFE successfully identified important features.
# ##### ANSWER 
