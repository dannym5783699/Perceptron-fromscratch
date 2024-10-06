from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class ClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class=None, binary : bool = True, **model_params):
        """
        model_class: The model class to instantiate
        model_params: Parameters for the model class (passed by GridSearchCV)
        """
        self.model_class = model_class  # The model class to instantiate
        self.model_params = model_params  # Parameters to instantiate the model
        self.model = None  # Model instance, initialized in fit()
        self.binary = binary    # Whether the model is binary or not

    def set_params(self, **params):
        """
        Override set_params to handle model-specific parameters correctly.
        """
        if 'model_class' in params:
            self.model_class = params.pop('model_class')
        if 'binary' in params:
            self.binary = params.pop('binary')
        
        self.model_params.update(params)  # Update the model_params with new parameters
        return self

    def get_params(self, deep=True):
        """
        Override get_params to return model parameters, so GridSearchCV can tune them.
        """
        return {'model_class': self.model_class, 'binary' : self.binary, **self.model_params}

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.model_class is None:
            raise ValueError("No model class specified. Please provide a valid model_class.")
        
        # Initialize the model with the provided parameters
        self.model = self.model_class(**self.model_params)
        
        # Fit the model with the data
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray):
        if self.model is None:
            raise ValueError("This ClassifierWrapper instance is not fitted yet. Call 'fit' first.")
        
        y_pred = self.model.forward(X)

        if self.binary:
            # Convert continuous scores to binary predictions
            # Should not modify already binary predictions
            y_pred = np.where(y_pred > 0, 1, 0)

        return y_pred
    

