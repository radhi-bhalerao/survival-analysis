import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from memory_profiler import profile
from lifelines import KaplanMeierFitter, ExponentialFitter
from lifelines.utils import concordance_index
from scipy.optimize import minimize
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class SurvivalDataset(Dataset):
    """Dataset class for survival data"""
    def __init__(self, X, durations, events):
        self.X = X
        self.durations = durations
        self.events = events
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.durations[idx], self.events[idx]
    
    def normalize(self):
        """Normalize input features"""
        copy = self.X.copy()
        for every_column in copy.columns:
            unique_vals = set(copy[every_column].unique())
            if unique_vals != {0, 1}:
                copy[every_column] = (copy[every_column] - copy[every_column].mean()) / copy[every_column].std()
        return copy
    
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

class CoxPHModel(nn.Module):
    def __init__(self, input_dim):
        super(CoxPHModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.scaler = StandardScaler()
        self.loss_history = {'train': [], 'val': []}
        
    def forward(self, x):
        return self.linear(x)
    
    def _negative_log_partial_likelihood(self, risk_scores, durations, events):
        ordered_idx = torch.argsort(durations, descending=True)
        risk_scores = risk_scores[ordered_idx]
        events = events[ordered_idx]
        
        # Calculate cumulative sum of risk scores
        cumsum_risk = torch.cumsum(risk_scores, dim=0)
        
        # Calculate log-sum-exp using the log-sum-exp trick
        max_risk = torch.max(risk_scores)
        log_risk = max_risk + torch.log(torch.cumsum(torch.exp(risk_scores - max_risk), dim=0))
        
        # Calculate individual likelihood contributions
        likelihood = risk_scores - log_risk
        
        # Zero out censored observations
        uncensored_likelihood = likelihood * events
        
        # Calculate negative log likelihood
        logL = -torch.sum(uncensored_likelihood)
        
        # Normalize by number of events
        num_events = torch.sum(events)
        return logL / (num_events + 1e-8)

    def prepare_data(self, df, feature_cols, duration_col, event_col, test_size=0.20, random_state=42):
        """Prepare and split data"""
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        X_train = self.scaler.fit_transform(train_df[feature_cols])
        X_test = self.scaler.transform(test_df[feature_cols])
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        durations_train = torch.FloatTensor(train_df[duration_col].values)
        durations_test = torch.FloatTensor(test_df[duration_col].values)
        events_train = torch.FloatTensor(train_df[event_col].values)
        events_test = torch.FloatTensor(test_df[event_col].values)
        
        train_dataset = SurvivalDataset(X_train, durations_train, events_train)
        test_dataset = SurvivalDataset(X_test, durations_test, events_test)
        
        return train_dataset, test_dataset
    
    def fit(self, df, feature_cols, duration_col, event_col, 
            batch_size=32, learning_rate=0.01, num_epochs=1000,
            test_size=0.2, random_state=42, verbose=True):
        """Fit the model using batch training"""
        self.loss_history = {'train': [], 'val': []}
        
        # Prepare data with batching
        train_dataset, test_dataset = self.prepare_data(
            df, feature_cols, duration_col, event_col, test_size, random_state
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.9)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=verbose, min_lr=1e-6
        )
        
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        patience = 100  # Early stopping patience
        
        for epoch in range(num_epochs):
            # Training
            self.train()
            train_losses = []
            for batch_X, batch_durations, batch_events in train_loader:
                optimizer.zero_grad()
                risk_scores = self(batch_X)
                loss = self._negative_log_partial_likelihood(
                    risk_scores, batch_durations, batch_events
                )
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_losses = []
                for batch_X, batch_durations, batch_events in test_loader:
                    risk_scores = self(batch_X)
                    val_loss = self._negative_log_partial_likelihood(
                        risk_scores, batch_durations, batch_events
                    )
                    val_losses.append(val_loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            self.loss_history['train'].append(avg_train_loss)
            self.loss_history['val'].append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = self.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                #print(f"Early stopping at epoch {epoch}")
                break
            
            if verbose and (epoch + 1) % 100 == 0:
                c_index = self.evaluate_batch(test_loader)
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(f'Train Loss: {avg_train_loss:.4f}')
                print(f'Val Loss: {avg_val_loss:.4f}')
                print(f'Val C-index: {c_index:.4f}')
                print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}\n')
        
        # Load best weights
        if best_weights is not None:
            self.load_state_dict(best_weights)
    
    def evaluate_batch(self, data_loader):
        """Evaluate model performance using concordance index"""
        self.eval()
        all_risk_scores = []
        all_durations = []
        all_events = []
        
        with torch.no_grad():
            for batch_X, batch_durations, batch_events in data_loader:
                risk_scores = self(batch_X)
                all_risk_scores.extend(risk_scores.numpy().flatten())
                all_durations.extend(batch_durations.numpy())
                all_events.extend(batch_events.numpy())
        
        return concordance_index(
            np.array(all_durations),
            -np.array(all_risk_scores),
            np.array(all_events)
        )
    
    def plot_loss_curve(self, figsize=(10, 6), title="Training and Validation Loss Over Time"):
        """
        Plot the loss curves from training
        
        Args:
            figsize (tuple): Figure size (width, height)
            title (str): Plot title
        """
        if not self.loss_history['train']:
            print("No loss history available. Train the model first using the fit method.")
            return
        
        plt.figure(figsize=figsize)
        plt.plot(self.loss_history['train'], label='Training Loss')
        plt.plot(self.loss_history['val'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log Partial Likelihood')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig('CoxPHModelLossCurve.png')
    
    def get_coefficients(self, feature_names=None):
        """
        Get model coefficients as a dictionary
        
        Args:
            feature_names (list): List of feature names
            
        Returns:
            dict: Dictionary of coefficients
        """
        coef_array = self.linear.weight.data.numpy().flatten()
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(coef_array))]
            
        return {name: float(coef) for name, coef in zip(feature_names, coef_array)}
    
    def report_results(self, feature_names=None):
        """
        Print model coefficients and their interpretation
        
        Args:
            feature_names (list): List of feature names
        """
        coefficients = self.get_coefficients(feature_names)
        
        print("\nCox PH Model Coefficients:")
        print("-" * 40)
        
        for feature, value in coefficients.items():
            hazard_ratio = np.exp(value)
            print(f"{feature}:")
            print(f"  Coefficient: {value:.4f}")
            print(f"  Hazard Ratio: {hazard_ratio:.4f}")
            print(f"  Interpretation: A one unit increase in {feature} is associated with")
            if hazard_ratio > 1:
                print(f"  a {((hazard_ratio - 1) * 100):.1f}% increase in hazard")
            else:
                print(f"  a {((1 - hazard_ratio) * 100):.1f}% decrease in hazard")
            print()
    
    def predict_risk(self, X):
        """
        Predict risk scores for new data
        
        Args:
            X (numpy.ndarray or pandas.DataFrame): Input features
            
        Returns:
            numpy.ndarray: Predicted risk scores
        """
        if isinstance(X, pd.DataFrame):
            X = self.scaler.transform(X)
        X = torch.FloatTensor(X)
        self.eval()
        with torch.no_grad():
            risk_scores = self(X)
        return risk_scores.numpy()