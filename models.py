import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from memory_profiler import profile
from lifelines import KaplanMeierFitter, ExponentialFitter, concordance_index
from scipy.optimize import minimize

class KaplanMeier:
    def __init__(self, dataset, time_column, event_column):
        self.dataset = dataset
        self.time_column = time_column
        self.event_column = event_column
        self.times = np.sort(dataset[time_column].unique())
        self.event_times = np.sort(dataset[time_column][dataset[event_column] == 1].unique())
        self.survival_probabilities = self.get_survival_probabilities()

    def get_survival_probabilities(self):
        survival_probabilities = []
        n = len(self.dataset)

         # Handle first time point separately
        t = self.times[0]
        #n_t = len(self.dataset[self.dataset[self.time_column] >= t])
        #d_t = len(self.dataset[(self.dataset[self.time_column] == t) & (self.dataset[self.event_column] == 1)])
        prob_of_death = 0 
        survival_probabilities.append(1 - prob_of_death)  # First probability

        for t in self.times[1:]:
            n_t = len(self.dataset[self.dataset[self.time_column] >= t])
            d_t = len(self.dataset[(self.dataset[self.time_column] == t) & (self.dataset[self.event_column] == 1)])
            prob_of_death = d_t/n_t if n > 0 else 0
            survival_probability_at_time_t = (1 - prob_of_death) * survival_probabilities[-1]
            survival_probabilities.append(survival_probability_at_time_t)
        return survival_probabilities

    def plot(self):
        plt.figure(1)
        plt.step(self.times, self.survival_probabilities, where='post')
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title('Kaplan-Meier Curve')
        plt.savefig('KaplanMeierByHand.png')

class ExponentialSurvivalCurve:
    def __init__(self, dataset, time_column, event_column):
        self.dataset = dataset
        self.time_column = time_column
        self.event_column = event_column
        self.times = np.sort(dataset[time_column].unique())
        self.event_times = np.sort(dataset[time_column][dataset[event_column] == 1].unique())
        self.time_values = np.linspace(self.dataset[self.time_column].min(), self.dataset[self.time_column].max(), 700)
        self.survival_probabilities = self.get_survival_probabilities()

    def log_likelihood_exponential(self,lambda_param, times, events):
        """
        Calculate log-likelihood for exponential distribution
        
        Parameters:
        - lambda_param: rate parameter
        - times: observed times
        - events: event indicators (1 for event, 0 for censored)
        """
        log_likelihood = np.sum(events * np.log(lambda_param) - lambda_param * times)

        return -log_likelihood  # Minimizing negative log-likelihood
    
    def find_lambda(self):
        """
        Find the lambda parameter that maximizes the likelihood of the data
        """
        # Initial guess for lambda
        result = minimize(self.log_likelihood_exponential, x0=[0.01], args=(self.dataset[self.time_column], self.dataset[self.event_column]), bounds=[(1e-10, None)])
        return result.x[0]

    def get_survival_probabilities(self):
        lambda_rate = self.find_lambda()
        #print(lambda_rate)
        survival_probabilities = np.exp( - self.time_values*lambda_rate)
        return survival_probabilities


    def plot(self):
        plt.figure()
        plt.step(self.time_values, self.survival_probabilities, where='post')
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title('Exponential Parametric Survival Curve')
        plt.savefig('ExponentialParametricSurvivalCurve.png')

class MahalanobisKNearestNeighbor:
    def __init__(self, dataset, time_column, event_column, k):
        self.dataset = dataset
        self.time_column = time_column
        self.event_column = event_column
        self.times = np.sort(dataset[time_column].unique())
        self.event_times = np.sort(dataset[time_column][dataset[event_column] == 1].unique())
        self.k = k
        self.feature_columns = None  # Will be set during fit
    
    
    def fit(self, X, y=None):
        """
        Prepare the model for making predictions.
        X: DataFrame of features
        y: Not needed for this model but included for sklearn compatibility
        """
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        # Split the data
        self.train = self.dataset.sample(frac=0.7, random_state=200)
        self.validation = self.dataset.drop(self.train.index)
        self.test = self.validation.sample(frac=0.5, random_state=200)
        self.validation = self.validation.drop(self.test.index)
        
        return self

    def predict(self, X):
        """
        Predict survival probabilities for each observation in X
        X: DataFrame of features for new observations
        Returns: Array of survival probability curves, shape (n_samples, n_times)
        """
        if self.feature_columns is None:
            raise ValueError("Model must be fit before making predictions")
            
        # Ensure X has the same columns as training data
        X = X[self.feature_columns]
        
        # Get survival curves for each observation
        survival_curves = []
        for idx in range(len(X)):
            survival_probs = self.get_survival_probabilities(idx)
            survival_curves.append(survival_probs)
            
        return np.array(survival_curves)
    
    def predict_survival_function(self, X):
        """
        Alias for predict() that returns survival functions
        Included for compatibility with other survival analysis packages
        """
        return self.predict(X)
        
    def mahalanobis_distance(self, x1, x2):
        feature_df = self.train.drop(columns=[self.time_column, self.event_column])
        for column in feature_df.columns:
            if feature_df[column].dtype == 'object':
                feature_df[column] = pd.Categorical(feature_df[column]).codes

        covariance_matrix = np.cov(feature_df, rowvar=False)
        inv_cov_matrix = np.linalg.inv(covariance_matrix)
        
        obs1 = feature_df.iloc[x1]
        obs2 = feature_df.iloc[x2]
        diff = obs1 - obs2
        distance = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff))
        return distance
    
    def get_k_nearest_neighbors(self, x):
        distances = []
        for i in range(len(self.train)):
            distances.append((i, self.mahalanobis_distance(x, i)))
        distances.sort(key=lambda x: x[1])
        return distances[:self.k]
    
    def get_survival_probabilities(self, x):
        neighbors = self.get_k_nearest_neighbors(x)
        
        def weight_function(distance):
            return np.exp(-distance)
        
        survival_probabilities = []
        t = self.times[0]
        
        n_weighted = sum(weight_function(dist) 
                        for idx, dist in neighbors 
                        if self.dataset.iloc[idx][self.time_column] >= t)
        
        d_weighted = sum(weight_function(dist) 
                        for idx, dist in neighbors 
                        if (self.dataset.iloc[idx][self.time_column] == t and 
                            self.dataset.iloc[idx][self.event_column] == 1))
        
        prob_of_death = d_weighted/n_weighted if n_weighted > 0 else 0
        survival_probabilities.append(1 - prob_of_death)
        
        for t in self.times[1:]:
            n_weighted = sum(weight_function(dist) 
                            for idx, dist in neighbors 
                            if self.dataset.iloc[idx][self.time_column] >= t)
            
            d_weighted = sum(weight_function(dist) 
                            for idx, dist in neighbors 
                            if (self.dataset.iloc[idx][self.time_column] == t and 
                                self.dataset.iloc[idx][self.event_column] == 1))
            
            prob_of_death = d_weighted/n_weighted if n_weighted > 0 else 0
            survival_probability_at_time_t = (1 - prob_of_death) * survival_probabilities[-1]
            survival_probabilities.append(survival_probability_at_time_t)
        
        return survival_probabilities
    def plot_survival_curves(self, X, labels=None, title="Survival Curves", figsize=(10, 6)):
        """
        Plot survival curves for given observations
        
        Parameters:
        X: DataFrame of features for observations to plot
        labels: List of labels for each observation (optional)
        title: Title for the plot
        figsize: Tuple of (width, height) for the plot
        """
        survival_curves = self.predict(X)
        
        plt.figure(figsize=figsize)
        
        # If no labels provided, use generic ones
        if labels is None:
            labels = [f"Patient {i+1}" for i in range(len(X))]
        
        # Plot each survival curve
        for i, curve in enumerate(survival_curves):
            plt.step(self.times, curve, where='post', label=labels[i])
        
        plt.xlabel(f"Time ({self.time_column})")
        plt.ylabel("Survival Probability")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        
        # Set y-axis limits
        plt.ylim(0, 1.05)
        plt.savefig('MahalanobisKNearestNeighbor.png')
        return plt
    def calculate_concordance_index(self, X_test, verbose=False):
        """
        Calculate the concordance index (C-index) for the model
        
        Parameters:
        X_test: DataFrame of test data including time and event columns
        verbose: If True, print additional information
        
        Returns:
        float: C-index score
        """
        # Get survival probabilities for test data
        survival_curves = self.predict(X_test.drop(columns=[self.time_column, self.event_column]))
        
        # Get the actual times and events from test data
        actual_times = X_test[self.time_column].values
        actual_events = X_test[self.event_column].values
        
        # Calculate risk scores as negative of the area under the survival curve
        # (higher risk = lower survival probability)
        risk_scores = [-np.trapz(curve, self.times) for curve in survival_curves]
        
        # Calculate C-index
        c_index = concordance_index(actual_times, 
                                  -np.array(risk_scores),  # Negative because higher risk = lower survival
                                  actual_events)
        
        if verbose:
            print(f"Concordance Index: {c_index:.3f}")
            print("Note: C-index of 0.5 indicates random predictions")
            print("      C-index of 1.0 indicates perfect ranking")
        
        return c_index

  
