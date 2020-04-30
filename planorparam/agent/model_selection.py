import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import RandomizedSearchCV

class ModelSelector():
    def __init__(self):
        self.manual_models = []
        self.learned_models = []
        self.manual_model_classifiers = [] #expected uncertainty/error measurement for manual models. learned models should have their own

    def add(self, model, model_type="manual"):
        if model_type == "manual":
            self.manual_models.append(model)
            self.manual_model_classifiers = ManualModelClassifier()
        else:
            self.learned_models.append(model)
    """
    Only relevant for manual models now
    """
    def add_history(self, states, errors, input_model):
        for model, classifier in zip(self.manual_models, self.manual_model_classifiers):
            if input_model == model:
                classifier.train(states, errors)
                return
        raise ValueError("Could not find input_model", input_model)


    def select_model(self, state, action, tol):
        for model, classifier in zip(self.manual_models, self.manual_model_classifiers):
            if classifier.predict(state, action) < tol: #no action for now
                return model
        for learned_model in self.learned_models:
            if learned_model.predict(state)[1] < tol: #no action for now
                return learned_model
        return learned_model[0] #best learned model we have; our only one.

class ManualModelClassifier():
    def __init__(self, params_file=None):
        if params_file is None:
            n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]  # Create the random grid
            rf = RFR()
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
            self.rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=70, cv=3, verbose=1,
                                           random_state=42, n_jobs=12)  # Fit the random search model

    def train(self, states, errors, params_file = None):
        self.rf_random.fit(states, errors)
    """
    :param predicts the amount of error
    """
    def predict(self, states):
        self.rf_random.predict(states)

