# Project 3

## Multivariate Regression Analysis

For single variable regression, we have used the following equation to demonstrate the goal of our model:

<img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}(y):=F(x)">

In words, we are estimating *x* with some function of *y*. However, as we can see in the word "multivariable", we are going to have multiple variables which our function applies to in this regression:

<img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}(y):=F(x_1,x_2,x_3,...x_p)">

Where, *x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, ... x<sub>p</sub>* are each of the variables which we believe to be relevant for estimating *y*.

The most straightforward approach, and our baseline for this assignment on multivariate data, will be to use an Ordinary Least Squares (OLS) model. This can be understood in single variable data as the line which can be drawn through a scatter plot with the least residuals (sum of distance from points to line). In multivariate data, an OLS model will still take the best fit, this time through *n+1* dimensional space, where *n* is the number of variables. For example, for two variables, we have the best fit 2-D plane through 3-D space.

One caveat of this method is that we cannot have more variables than observations. Since OLS is essentially solving a linear algebra matrix, if we take for example a 3 variable dataset with 3 observations, you will always be able to find the best fit perfectly with no residuals. Thus, if we add a fourth variable it will be completely redundant.

Thus we must be careful with using large numbers of variables, even if there are more observations, we are at risk of overfitting. This is why we limit ourselves to the variables we expect to be relevant.

## Gradient Boosting

As stated, any proper OLS model will have some residuals between *y* and our predicted value, *&ycirc;*. We can represent this difference with the term *&Delta;y*. One way in which we can try to improve upon OLS is to model this residuals themselves (find *E(y)*) and add this model to our original one. In other words, if we call our model of residuals *h*, and our OLS model is *F*, then our new model will be based on *F + h*. Gradient boosting algorithms find *h* using decision trees.

In this assignment we are going to use XGBoost, a type of gradient boosting algorithm, to model multivariate data.

## Loading Data and Building Models

We are going to evaluate the cross-validated error of each of our models against two other regression methods previously covered: LOESS and boosted LOESS.

The data sets used will each be taken from previous classes: a wine quality data set and a student academic performance data set.

```Python
import numpy as np
import pandas as pd
winedata = pd.read_csv('/tmp/winequality-red.csv')
studdata = pd.read_csv('/tmp/student_prediction.csv')
```
In the wine data, we will use *quality* as our dependent variable, and *volatile acidity*, *pH*, and *alcohol* as predictive features. For student data, the *GRADE* variable indicates how the student performed in the class, and will be our dependent variable. Our predictive features will be *FATHER_EDU*, *STUDY_HRS*, and *LISTENS*.

```Python
Xwine = winedata[['volatile acidity', 'pH', 'alcohol']].values
ywine = winedata['quality'].values

Xstud = studdata[['FATHER_EDU', 'STUDY_HRS', 'LISTENS']].values
ystud = studdata['GRADE'].values
```
```Python
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
from matplotlib import pyplot
```

Now that we have all of the libraries we'll need, we can define the functions which will create our models. First, the kernel which will be used for our locally weighted regression methods: tricubic. Then we will define the standard LWG method.

```Python
# Tricubic Kernel, this will be used for LOESS and boosted LOESS
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

  #Defining the kernel local regression model

def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        #A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        #theta = linalg.solve(A, b) # A*theta = b
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
```

The following is an addition to our previous LWR model which accounts for boosting. Using our previous example of *F + h*, our *F* is the LWR algorithm, and *h* is the random forest algorithm which is trained on the residuals from the original regression.

```Python
def boosted_lwr(X, y, xnew, kern, tau, intercept):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  # Now train the Decision Tree on y_i - F(x_i)
  new_y = y - Fx
  #model = DecisionTreeRegressor(max_depth=2, random_state=123)
  model = RandomForestRegressor(n_estimators=100,max_depth=2)
  #model = model_xgb
  model.fit(X,new_y)
  output = model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output
```

Lastly, we will import a boosted regression method called XGBoost, or Extreme Gradient Boosting.

```Python
import xgboost as xgb
```

## Results

We are going to compare the quality of our models using their mean absolute error, calculated from a 10-fold cross validation.

```Python
def model_mae(X, y):
  scale = StandardScaler()
  mse_lwr = []
  mse_blwr = []
  mse_ols = []
  mse_xgb = []
  kf = KFold(n_splits=10,shuffle=True,random_state=3)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    lin_reg = LinearRegression().fit(xtrain, ytrain)
    yhat_ols = lin_reg.predict(xtest)
    yhat_lwr = lw_reg(xtrain,ytrain, xtest,Tricubic,tau=0.9,intercept=True)
    yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Tricubic,tau=0.9,intercept=True)
    model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
    model_rf.fit(xtrain,ytrain)
    yhat_rf = model_rf.predict(xtest)
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    mse_lwr.append(mse(ytest,yhat_lwr))
    mse_blwr.append(mse(ytest,yhat_blwr))
    mse_ols.append(mse(ytest,yhat_ols))
    mse_xgb.append(mse(ytest,yhat_xgb))
  print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
  print('The Cross-validated Mean Squared Error for BLWR is : '+str(np.mean(mse_blwr)))
  print('The Cross-validated Mean Squared Error for OLS is : '+str(np.mean(mse_ols)))
  print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
  return None
```

```Python
model_mae(Xstud, ystud)
```

Output: 
- The Cross-validated Mean Squared Error for LWR is : 6.332361979495421
- The Cross-validated Mean Squared Error for BLWR is : 6.3990660475919565
- The Cross-validated Mean Squared Error for OLS is : 5.011407996783873
- The Cross-validated Mean Squared Error for XGB is : 4.873095834118606

```Python
model_mae(Xwine, ywine)
```

Output:
- The Cross-validated Mean Squared Error for LWR is : 0.44046726462433317
- The Cross-validated Mean Squared Error for BLWR is : 0.43971188199540956
- The Cross-validated Mean Squared Error for OLS is : 0.4441520738913711
- The Cross-validated Mean Squared Error for XGB is : 0.4569170434605724

While we would expect our boosted algorithms to have better results, in actuality it is more inconclusive. With the students dataset, the OLS and XGB algorithms are signficantly better than weighted regression. However, for the wine dataset, all are approximately the same, with boosted linear weighted model being the most successful.

These differences could be due to the nature of our datasets being very different. The student dataset is relatively small (*N = 145*), and the variable we tested would likely have simple relationships with the data (more studying = better grade, etc.). On the other hand, something as subjective as wine taste will likely have more complex variables. This dataset is also significantly larger (*N = 1599*).

This could explain why a more sophisticated fit like LWR would perform better for the wine. On the other hand, OLS would have an advantage in the simpler student dataset.






