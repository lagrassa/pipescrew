import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import RandomizedSearchCV

class ModelSelector():
    def __init__(self, use_history=False):
        self.manual_models = []
        self.learned_models = []
        if use_history:
            try:
                self.old_states = np.load("data/old_states.npy")
                self.old_errors = np.load("data/old_errors.npy")
            except FileNotFoundError:
                self.old_states = None
                self.old_errors = None
        else:
            self.old_states = None
            self.old_errors = None

        self.manual_model_classifiers = [] #expected uncertainty/error measurement for manual models. learned models should have their own

    def add(self, model, model_type="manual"):
        if model_type == "manual":
            self.manual_models.append(model)
            classifier = ManualModelClassifier(init_states = self.old_states, init_errors = self.old_errors)
            self.manual_model_classifiers.append(classifier)
        else:
            self.learned_models.append(model)
    """
    Only relevant for manual models now
    """
    def add_history(self, states, errors, input_model):
        #add to states in data
        if self.old_states is not None:
            self.old_states = np.vstack([self.old_states, states])
            self.old_errors = np.vstack([self.old_errors, errors])
        else:
            self.old_states = states
            self.old_errors = errors
        np.save("data/old_states.npy", self.old_states)
        np.save("data/old_errors.npy", self.old_errors)
        for model, classifier in zip(self.manual_models, self.manual_model_classifiers):
            if input_model == model:
                classifier.train(states, errors)
                return
        raise ValueError("Could not find input_model", input_model)


    def select_model(self, state, action, tol, check_learned_model = False):
        for model, classifier in zip(self.manual_models, self.manual_model_classifiers):
            try:
                if classifier.predict(state.reshape(1,-1)) < tol: #no action for now
                    return model
            except:
                print("not fitted yet")
                return model

        if not check_learned_model:
            return self.learned_models[0]
        for learned_model in self.learned_models:
            if learned_model.predict(state, action)[1] < tol: #no action for now
                return learned_model
        return self.learned_models[0] #best learned model we have; our only one.

class ManualModelClassifier():
    def __init__(self, init_states = None, init_errors = None, params_file=None):
        if params_file is None:
            n_estimators = [int(x) for x in np.linspace(start=20, stop=1000, num=10)]
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
            self.rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=30, cv=3, verbose=1,
                                           random_state=42, n_jobs=12)  # Fit the random search model
            if init_states is not None:
                self.train(init_states, init_errors)

    def train(self, states, errors, params_file = None):
        training_states = states
        training_errors = errors
        try:
            old_states = np.load("data/old_states.npy")
            old_errors = np.load("data/errors.npy")
            training_states = np.vstack([old_states, states])
            training_errors = np.vstack([old_errors, errors])
        except FileNotFoundError:
            print("COuld not find fikle")
        if len(training_errors.shape) == 1:
            training_errors = training_errors.reshape(1,-1)
        self.rf_random.fit(training_states.T, training_errors.T)
        pred_states = self.rf_random.predict(states.T)
        print(np.mean(pred_states-errors), "mean errors on current data")
    """
    :param predicts the amount of error
    """
    def predict(self, states):
        pred = self.rf_random.predict(states)
        print("prediction", pred)
        return pred

