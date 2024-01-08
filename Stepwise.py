'''
Available methods are the followings:
[1] StepwiseRegression

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 31-01-2024

'''
import pandas as pd, numpy as np
from sklearn.metrics import (r2_score, 
                             mean_squared_error, 
                             roc_auc_score)
from scipy import stats

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
        method that provides information about feature importance i.e. 
        `coef_`. In addition to method, regressor and classifier must 
        also provide `predict`, and `predict_proba`, respectively.

    alpha : float, default=0.05
        Passing criterion or p-value of two-tailed distribution.

    '''
    
    def __init__(self, estimator, alpha=0.05):
        
        # Validate estimator
        has_fit = hasattr(estimator, "fit")
        has_predict = (hasattr(estimator, "predict") | 
                       hasattr(estimator, "predict_proba"))
        if (has_fit & has_predict)==False:
            raise ValueError('An estimator must provide 2 methods, which '
                             'are `fit`, and `predict` or `predict_proba`.')
        else: self.estimator = estimator
        
        # Validate alpha [0.,1.]
        self.alpha = self.Interval("alpha", alpha, float, 0, 1, "both")
    
    def __anova__(self, X, y, sample_weight=None):
        
        '''Fit estimator and perform ANOVA'''
        y = np.array(y).flatten().copy()
        model = self.estimator.fit(X, y, sample_weight)
        coefs = np.r_[model.intercept_, model.coef_.flatten()]
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:,1]
            return self.__logistic__(X, y, y_proba, coefs)
        elif hasattr(model, "predict"):
            y_pred = model.predict(X).flatten()
            return self.__linear__(X, y, y_pred, coefs)
        else: pass
    
    def __linear__(self, X, y_true, y_pred, coefs):
        
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
        
        Returns
        -------
        {"feature": list of features
         "coef"   : estimated coefficients (intercept_ + n_features,)
         "stderr" : standard errors
         "t"      : t-statistics
         "pvalue" : P-values (two-tailed distribution)
         "lower"  : lower bounds (alpha/2)
         "upper"  : upper bounds (1 - alpha/2)
         "r2"     : R-Squared
         "adj_r2" : adjusted R-Squared
         "mse"    : Mean-Squared-Error}
         
        '''
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
        t = stats.t.ppf(1-self.alpha/2, df) * np.abs(stderrs)
        
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
                "r2"     : r2, 
                "adj_r2" : adj_r2, 
                "mse"    : mse}
    
    def __logistic__(self, X, y_true, y_proba, coefs):
        
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
        
        Returns
        -------
        {"feature": list of features
         "coef"   : estimated coefficients
         "stderr" : standard errors
         "t"      : t-statistics
         "pvalue" : P-values (two-tailed distribution)
         "lower"  : lower bounds (alpha/2)
         "upper"  : upper bounds (1 - alpha/2)
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
        t = stats.t.ppf(1-self.alpha/2, df) * np.abs(stderrs)
        
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
                "gini"   : 2*auc - 1, 
                "ks"     : ks.statistic}

class StepwiseRegression(ANOVA, ValidateParams):
    
    '''
    Stepwise Regression

    Parameters
    ----------
    estimator : Estimator instance
        An estimator whether regressor or classifier with `fit` 
        method that provides information about feature importance i.e. 
        `coef_`. In addition to method, regressor and classifier must 
        also provide `predict`, and `predict_proba`, respectively.

    method : {"backward", "forward", "stepwise"}, default="forward"
        - 'forward'  : forward selection
        - 'backward' : backward elimination
        - 'stepwise' : a combination of forward selection and backward
                       elimination

    alpha : float, default=0.05
        Passing criterion or p-value of two-tailed distribution.
    
    Attributes
    ----------
    estimator_ : Estimator instance
        Fitted estimator with selected features.
    
    features : list of str
        List of selected features.
    
    results_ : dict of numpy (masked) ndarrays
        It contains results from each step of selection/elimination.
        
        {"feature": list of features
         "coef"   : estimated coefficients (intercept_ + n_features,)
         "stderr" : standard errors
         "t"      : t-statistics
         "pvalue" : P-values (two-tailed distribution)
         "lower"  : lower bounds (alpha/2)
         "upper"  : upper bounds (1 - alpha/2)
         "r2"     : R-Squared
         "adj_r2" : adjusted R-Squared
         "mse"    : Mean-Squared-Error}
         "gini"   : GINI index
         "ks"     : adjusted R-Squared}

    '''
    def __init__(self, estimator, method='forward', alpha=0.05):
        
        args = ('method', method, ["forward", "backward", "stepwise"], str)
        self.method = self.StrOptions(*args)
        super().__init__(estimator, alpha)
    
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
        self.estimator_ = self.estimator.fit(*args)
        self.n_iters = len(self.results_)
        
        return self
    
    def __backward__(self, X, y, sample_weight=None, n_features=None):
        
        '''
        Backward elimination removes features from the full estimator one
        by one until no features overcome the threshold value i.e. greater
        than defined p-value. Features never return once removed.
        
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
        
        # Validate parameters
        args = (int, 1, X.shape[1], "both")
        self.Interval("n_features", n_features, *args)
        
        while len(use_features) >= n_features:
            
            # Fit estimator and perform ANOVA
            anova = self.__anova__(X[use_features].copy(), y, sample_weight)
            anova.update({"method" : "backward"})
            self.results_.update({len(self.results_) : anova})
            
            # Determine variable with maximum p-value
            # index >= 1 (ignore intercept)
            k = np.argmax(anova["pvalue"][1:]) 
            if anova["pvalue"][1:][k] < self.alpha: break
            else: use_features.remove(use_features[k])

        return use_features
    
    def __forward__(self, X, y, sample_weight=None, n_features=None, use_features=None):
        
        '''
        Forward selection adds features one by one to an empty estimator
        until no features overcome the threshold value i.e. less than or
        equal to defined p-value. Features never leave once added.
        
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
                anova  = self.__anova__(X0, y, sample_weight)
                pvalue = anova["pvalue"][-1]
                
                # Find next best feature
                if pvalue < min_pvalue:
                    min_pvalue   = pvalue
                    best_feature = new_feature
                    best_stderrs = anova
                
            # Determine variable with minimum p-value
            if min_pvalue <= self.alpha:
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
            base_features = use_features.copy()
            args = (X, y, sample_weight, len(use_features)+1, use_features)
            use_features = self.__forward__(*args)
            use_features = self.__backward__(X[use_features], y, sample_weight)

        return use_features