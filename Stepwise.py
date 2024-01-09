'''
Available methods are the followings:
[1] StepwiseRegression

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 31-01-2024

'''
import pandas as pd, numpy as np
from scipy import stats
from sklearn.metrics import (r2_score, 
                             mean_squared_error, 
                             roc_auc_score)

__all__ = ["StepwiseRegression"]

class ValidateParams:
    
    '''Validate parameters'''
    
    def Interval(self, Param, Value, dtype=int, 
                 left=None, right=None, closed="both"):

        '''
        Validate numerical input.

        Parameters
        ----------
        Param : str
            Parameter's name

        Value : float or int
            Parameter's value

        dtype : {int, float}, default=int
            The type of input.

        left : float or int or None, default=None
            The left bound of the interval. None means left bound is -∞.

        right : float, int or None, default=None
            The right bound of the interval. None means right bound is +∞.

        closed : {"left", "right", "both", "neither"}
            Whether the interval is open or closed. Possible choices are:
            - "left": the interval is closed on the left and open on the 
              right. It is equivalent to the interval [ left, right ).
            - "right": the interval is closed on the right and open on the 
              left. It is equivalent to the interval ( left, right ].
            - "both": the interval is closed.
              It is equivalent to the interval [ left, right ].
            - "neither": the interval is open.
              It is equivalent to the interval ( left, right ).

        Returns
        -------
        Value : float or int
            Parameter's value

        '''
        Options = {"left"    : (np.greater_equal, np.less), # a<=x<b
                   "right"   : (np.greater, np.less_equal), # a<x<=b
                   "both"    : (np.greater_equal, np.less_equal), # a<=x<=b
                   "neither" : (np.greater, np.less)} # a<x<b

        f0, f1 = Options[closed]
        c0 = "[" if f0.__name__.find("eq")>-1 else "(" 
        c1 = "]" if f1.__name__.find("eq")>-1 else ")"
        v0 = "-∞" if left is None else str(dtype(left))
        v1 = "+∞" if right is None else str(dtype(right))
        if left  is None: left  = -np.inf
        if right is None: right = +np.inf
        interval = ", ".join([c0+v0, v1+c1])
        tuples = (Param, dtype.__name__, interval, Value)
        err_msg = "%s must be %s or in %s, got %s " % tuples    

        if isinstance(Value, dtype):
            if not (f0(Value, left) & f1(Value, right)):
                raise ValueError(err_msg)
        else: raise ValueError(err_msg)
        return Value

    def StrOptions(self, Param, Value, options, dtype=str):

        '''
        Validate string or boolean inputs.

        Parameters
        ----------
        Param : str
            Parameter's name
            
        Value : float or int
            Parameter's value

        options : set of str
            The set of valid strings.

        dtype : {str, bool}, default=str
            The type of input.
        
        Returns
        -------
        Value : float or int
            Parameter's value

        '''
        if Value not in options:
            err_msg = f'{Param} ({dtype.__name__}) must be either '
            for n,s in enumerate(options):
                if n<len(options)-1: err_msg += f'"{s}", '
                else: err_msg += f' or "{s}" , got %s'
            raise ValueError(err_msg % Value)
        return Value
    
    def check_range(self, param0, param1):
        
        '''
        Validate number range.
        
        Parameters
        ----------
        param0 : tuple(str, float)
            A lower bound parameter e.g. ("name", -100.)
            
        param1 : tuple(str, float)
            An upper bound parameter e.g. ("name", 100.)
        '''
        if param0[1] >= param1[1]:
            raise ValueError(f"`{param0[0]}` ({param0[1]}) must be less"
                             f" than `{param1[0]}` ({param1[1]}).")

class ANOVA(ValidateParams):
    
    '''
    Analysis of Variance (ANOVA)
    
    Parameters
    ----------
    estimator : Estimator instance
        An estimator whether regressor or classifier with `fit` 
        method that accepts 3 positional arguments, which are `X`, 
        `y`, and `sample_weight`. It must also provide information 
        about feature importance i.e. `coef_` after fitting. In 
        addition, regressor and classifier must provide methods, 
        which are `predict(X)`, and `predict_proba(X)`, respectively.

    '''
    def __init__(self, estimator):
        
        # Validate estimator
        has_fit = hasattr(estimator, "fit")
        has_predict = (hasattr(estimator, "predict") | 
                       hasattr(estimator, "predict_proba"))
        if (has_fit & has_predict)==False:
            raise ValueError('An estimator must provide 2 methods, which '
                             'are `fit`, and `predict` or `predict_proba`.')
        else: self.estimator = estimator
        
    def __anova__(self, X, y, sample_weight=None, alpha=0.05):
        
        '''Fit estimator and perform ANOVA'''
        y = np.array(y).flatten().copy()
        try: model = self.estimator.fit(X, y, sample_weight)
        except: model = self.estimator.fit(X, y)
        coefs = np.r_[model.intercept_, model.coef_.flatten()]

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:,1]
            return self.__logistic__(X, y, y_proba, coefs, alpha)
        elif hasattr(model, "predict"):
            y_pred = model.predict(X).flatten()
            return self.__linear__(X, y, y_pred, coefs, alpha)
        else: pass
    
    def __linear__(self, X, y_true, y_pred, coefs, alpha=0.05):
        
        '''
        Peform analysis of variance on linear regression
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y_true : array-like of shape (n_samples,)
            Ground truth (correct) target values.

        y_pred : array-like of shape (n_samples,)
            Estimated target values.
            
        coefs : array of shape (intercept_ + n_features,)
            Estimated coefficients for the linear regression problem.
        
        alpha : float, default=0.05
            Passing criterion or p-value of two-tailed distribution.
        
        Returns
        -------
        {"feature": list of features
         "coef"   : estimated coefficients
         "stderr" : standard errors
         "t"      : t-statistics
         "pvalue" : P-values (two-tailed)
         "lower"  : lower bounds (alpha/2)
         "upper"  : upper bounds (1 - alpha/2)
         "alpha"  : alpha
         "r2"     : R-Squared
         "adj_r2" : adjusted R-Squared
         "mse"    : Mean-Squared-Error}
         
        '''
        # Validate alpha [0.,1.]
        alpha = self.Interval("alpha", alpha, float, 0, 1, "both")
        
        # Initialize parameters
        features = ["intercept"] + list(X)
        X = np.array(X).copy()
        X = np.hstack((np.ones((len(X),1)), X))

        # Degree of freedom and MSE
        N,K = X.shape
        df  = N - K - 1
        mse = (sum((y_true-y_pred)**2)) / df
        
        # Standard error, t-stat, and p-values
        stderrs  = np.sqrt(mse*(np.linalg.inv(X.T.dot(X)).diagonal()))
        t_stats  = coefs / stderrs
        p_values = np.array([2*(1-stats.t.cdf(abs(t), df)) for t in t_stats])
        t = stats.t.ppf(1-alpha/2, df) * np.abs(stderrs)
        
        # Regression statistics
        r2 = r2_score(y_true, y_pred)
        factor = ((1-r2) * ((N-1)/(N-K-1)))
        adj_r2 = r2 - (1-r2) * factor
        mse = (sum((y_true-y_pred)**2)) / (N-K-1)

        return {"feature": features,
                "coef"   : coefs,
                "stderr" : stderrs, 
                "t"      : t_stats, 
                "pvalue" : p_values, 
                "lower"  : coefs - t, 
                "upper"  : coefs + t, 
                "alpha"  : alpha,
                "r2"     : r2, 
                "adj_r2" : adj_r2, 
                "mse"    : mse}
    
    def __logistic__(self, X, y_true, y_proba, coefs, alpha=0.05):
        
        '''
        Peform analysis of variance on logistic regression
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y_true : array-like of shape (n_samples,)
            True labels or binary label indicators. 

        y_proba : array-like of shape (n_samples,)
            Probability estimates. i.e. estimator.classes_[1] and thus 
            estimator.predict_proba(X, y)[:, 1].
            
        coefs : array of shape (intercept_ + n_features,)
            Estimated coefficients for the logistic regression problem.
            
        alpha : float, default=0.05
            Passing criterion or p-value of two-tailed distribution.
        
        Returns
        -------
        {"feature": list of features
         "coef"   : estimated coefficients
         "stderr" : standard errors
         "t"      : t-statistics
         "pvalue" : P-values (two-tailed)
         "lower"  : lower bounds (alpha/2)
         "upper"  : upper bounds (1 - alpha/2)
         "alpha"  : alpha
         "gini"   : GINI index
         "ks"     : adjusted R-Squared}
         
        '''
        # Initialize parameters
        features = ["intercept"] + list(X)
        X = np.array(X).copy()
        X = np.hstack((np.ones((len(X),1)), X))
        
        # Degree of freedom and MSE
        N,K = X.shape
        df  = N - K - 1
        
        # Standard error, t-stat, and p-values
        V = (y_proba * (1-y_proba)).reshape(-1,1)
        W = np.identity(len(y_proba)) * V
        stderrs  = np.sqrt(np.linalg.inv(X.T.dot(W).dot(X)).diagonal())
        t_stats  = coefs / stderrs
        p_values = np.array([2*(1-stats.t.cdf(abs(t), df)) for t in t_stats])
        t = stats.t.ppf(1-alpha/2, df) * np.abs(stderrs)
        
        # Logistic regression statistics
        auc = roc_auc_score(y_true, y_proba)
        ks  = stats.ks_2samp(y_proba[y_true==0], 
                             y_proba[y_true==1])
      
        return {"feature": features,
                "coef"   : coefs,
                "stderr" : stderrs, 
                "t"      : t_stats, 
                "pvalue" : p_values, 
                "lower"  : coefs - t, 
                "upper"  : coefs + t, 
                "alpha"  : alpha,
                "gini"   : 2*auc - 1, 
                "ks"     : ks.statistic}

class StepwiseRegression(ANOVA, ValidateParams):
    
    '''
    Stepwise Regression

    Parameters
    ----------
    estimator : Estimator instance
        An estimator whether regressor or classifier with `fit` 
        method that accepts 3 positional arguments, which are `X`, 
        `y`, and `sample_weight`. It must also provide information 
        about feature importance i.e. `coef_` after fitting. In 
        addition, regressor and classifier must provide methods, 
        which are `predict(X)`, and `predict_proba(X)`, respectively.

    method : {"backward", "forward", "stepwise"}, default="forward"
        - 'forward'  : forward selection
        - 'backward' : backward elimination
        - 'stepwise' : a combination of forward selection and backward
                       elimination
    
    add_features : {int, None}, default=1
        Number of additional features to be retained during forward
        selection. If None, it keeps adding until criterion is no 
        longer satisfied. This is relevant when method is `stepwise`.

    fwd_alpha : float, default=0.05
        Selecting criterion (two-tailed p-value) of `foward` selection.
        This specifies significance level for feature to enter. This 
        couble be greater than the usual 0.05 level so that it is not 
        too difficult to enter features into the estimator. 

    bwd_alpha : float, default=0.05
        Removing criterion (two-tailed p-value) of `backward` 
        elimination. This specifies significance level for feature to 
        be removed. This could be greater than the usual 0.05 level so 
        that it is not too easy to remove features from the estimator.
    
    Attributes
    ----------
    estimator_ : Estimator instance
        Fitted estimator with selected features.
    
    features : list of str
        List of selected features.

    results_ : dict
        It contains results from each step of selection/elimination.
        In each key, it contains information as follows:
        
            "feature": list of features
            "coef"   : estimated coefficients
            "stderr" : standard errors
            "t"      : t-statistics
            "pvalue" : P-values (two-tailed)
            "lower"  : lower bounds (alpha/2)
            "upper"  : upper bounds (1 - alpha/2)
            "alpha"  : alpha
            "r2"     : R-Squared (1)
            "adj_r2" : adjusted R-Squared (1)
            "mse"    : Mean-Squared-Error (1)
            "gini"   : GINI index (2)
            "ks"     : adjusted R-Squared (2)
            
        Note: (1) when estimator is regressor
              (2) when estimator is classifer

    '''
    def __init__(self, estimator, method='forward', add_features=1, 
                 fwd_alpha=0.05, bwd_alpha=0.05):
        
        # Validate method i.e. {"forward", "backward", "stepwise"}
        args = (["forward", "backward", "stepwise"], str)
        self.method = self.StrOptions('method', method, *args)
        
        # Validate alpha [0.,1.]
        args = (float, 0, 1, "both")
        self.fwd_alpha = self.Interval("fwd_alpha", fwd_alpha, *args)
        self.bwd_alpha = self.Interval("bwd_alpha", bwd_alpha, *args)
        
        # Validate add_features
        if add_features is not None:
            args = ("add_features", add_features, int, 1, None, "left")
            self.add_features = self.Interval(*args)
        else: self.add_features = add_features
            
        super().__init__(estimator)
    
    def fit(self, X, y, sample_weight=None, n_features=None, use_features=None):
        
        '''
        Fit linear model.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.
            
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples. If 
            not provided, then each sample is given unit weight.
        
        n_features : int, default=None
            The number of features to be retained. If None, it defaults 
            to X.shape[1] when `method` is "forward", and 1 when `method` 
            is "backward".

        use_features : list of str, default=None
            Subset of features in `X` to start with. All elements must be 
            strings that correspond to column names. If None, no features 
            is selected. This is only relevant when `method` is "forward".
        
        Returns
        -------
        self : object
            Returns the instance itself.
            
        '''
        # Validate parameters
        if isinstance(X, pd.DataFrame)==False: 
            raise ValueError(f"`X` must be pandas.DataFrame. " 
                             f"Got {type(X)} instead.")
        
        # Initialize parameters
        self.features = list(X)
        self.results_ = dict()
        
        if self.method == "backward":
            args = (X, y, sample_weight, n_features)
            self.features = self.__backward__(*args)
        elif self.method == "forward":
            args = (X, y, sample_weight, n_features, use_features)
            self.features = self.__forward__(*args)
        elif self.method == "stepwise":
            args = (X, y, sample_weight)
            self.features = self.__stepwise__(*args)
        else: pass

        # Fit estimator with final set of features
        args = (X[self.features], y, sample_weight)
        try: self.estimator_ = self.estimator.fit(*args)
        except: self.estimator_ = self.estimator.fit(*args[:2])
        self.n_iters = len(self.results_)
        
        return self
    
    def __backward__(self, X, y, sample_weight=None, n_features=None):
        
        '''
        A backward-elimination removes features that is the least 
        statistically significant from the full estimator one by one 
        until all features remaining in the equation is statistically 
        significant i.e. greater than defined p-value. Features never 
        return once removed.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples. If 
            not provided, then each sample is given unit weight.
            
        n_features : int, default=None
            The number of features to be retained in estimator. If None, 
            it defaults to 1.
            
        Returns
        -------
        use_features : list of str
            List of remaining features.
            
        '''
        # Initialize parameters
        use_features = list(X)
        if n_features is None: n_features = 1
        params = (y, sample_weight, self.bwd_alpha)
        
        # Validate parameters
        args = (int, 1, X.shape[1], "both")
        self.Interval("n_features", n_features, *args)
        
        while len(use_features) > n_features:
            
            # Fit estimator and perform ANOVA
            anova = self.__anova__(X[use_features].copy(), *params)
            anova.update({"method" : "backward"})
            self.results_.update({len(self.results_) : anova})
            
            # Determine variable with maximum p-value
            # index >= 1 (ignore intercept)
            k = np.argmax(anova["pvalue"][1:]) 
            if anova["pvalue"][1:][k] >= self.bwd_alpha: 
                use_features.remove(use_features[k])
                anova = self.__anova__(X[use_features].copy(), *params)
                anova.update({"method" : "backward"})
                self.results_.update({len(self.results_)-1 : anova})
            else: break
        
        return use_features
    
    def __forward__(self, X, y, sample_weight=None, n_features=None, use_features=None):
        
        '''
        A forward-selection adds features one by one, based on which 
        feature is the most statistically significant, to an empty 
        estimator unitl there are no remaining statistically significant
        features i.e. less than or equal to defined p-value. Features 
        never leave once added.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values. 

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples. If 
            not provided, then each sample is given unit weight.
            
        n_features : int, default=None
            The number of features to be retained. If None, it defaults 
            to maximum number of features i.e. X.shape[1].
            
        use_features : list of str, default=None
            Subset of features in `X` to start with. All elements must be 
            strings that correspond to column names. If None, no features 
            is selected.

        Returns
        -------
        use_features : list of str
            List of remaining features.
            
        '''
        # Initialize parameters
        if use_features is None: use_features =list()
        features = list(set(list(X)).difference(use_features))
        if n_features is None: n_features = X.shape[1]
        
        # Validate parameters 
        args = (int, 1, X.shape[1], "both")
        self.Interval("n_features", n_features, *args)
        
        while len(use_features) < n_features:
            
            min_pvalue = 1.0
            for new_feature in features:
                
                # Fit estimator and perform ANOVA
                X0 = X[use_features + [new_feature]].copy()
                anova  = self.__anova__(X0, y, sample_weight, self.fwd_alpha)
                pvalue = anova["pvalue"][-1]
                
                # Find next best feature
                if pvalue < min_pvalue:
                    min_pvalue   = pvalue
                    best_feature = new_feature
                    best_stderrs = anova
                
            # Determine variable with minimum p-value
            # Enter the best feature in when there is no feature
            if (min_pvalue <= self.fwd_alpha) | (len(use_features)==0):
                use_features.append(best_feature)
                features.remove(best_feature)
                best_stderrs.update({"method" : "forward"})
                self.results_.update({len(self.results_) : best_stderrs})
            else: break

        return use_features
    
    def __stepwise__(self, X, y, sample_weight=None):
        
        '''
        Stepwise regression is a combination of forward selection and
        backward elimination. At each step, a new feature that satisfies
        criterion i.e. p-value is added. Then a model gets evaluated. If 
        one or more features are no longer passing p-value, they are 
        pruned. Then the process repeats until set of features does not 
        change.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples. If 
            not provided, then each sample is given unit weight.
        
        Returns
        -------
        use_features : list of str
            List of remaining features.
            
        '''
        use_features, base_features = list(), [None]
        while set(base_features)!=set(use_features):
            
            if self.add_features is not None:
                add_features = len(use_features) + self.add_features
                add_features = int(np.fmin(add_features, X.shape[1]))
            else: add_features = None
            
            base_features = use_features.copy()
            args = (X, y, sample_weight, add_features, use_features)
            use_features = self.__forward__(*args)
            use_features = self.__backward__(X[use_features], y, sample_weight)

        return use_features