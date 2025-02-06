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
from torch.utils.data import Dataset, DataLoader, TensorDataset 
from sklearn.linear_model import LogisticRegression

class SurvivalDataset(Dataset):
    """Dataset class for survival data"""
    def __init__(self, X, durations, events):
        self.X = X
        self.durations = durations
        self.events = events
    
    def __len__(self):
        return len(self.X.columns)
    
    def __getitem__(self, idx):
        return self.X[idx], self.durations[idx], self.events[idx]
    
    def get_features(self):
        feature_dx = self.X.drop(columns=[self.durations, self.events])
        return feature_dx.columns.values
    
    @classmethod
    def split_dataset(cls, dataset, durations, events, test_size=0.2, val_size=0.2, random_state=None):
        """
        Split the dataset into train, validation, and test sets
        
        Args:
        - dataset (SurvivalDataset): Original dataset
        - test_size (float): Proportion of data to use for testing
        - val_size (float): Proportion of training data to use for validation
        - random_state (int): Random seed for reproducibility
        
        Returns:
        - Tuple of SurvivalDataset instances (train, val, test)
        """
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Total number of samples
        total_samples = len(dataset.X)
        print(total_samples)
        # Generate indices for splitting
        indices = np.random.permutation(total_samples)
        
        # Calculate split indices
        test_split = int(total_samples * test_size)
        train_val_split = total_samples - test_split
        val_split = int(train_val_split * val_size)
        train_split = train_val_split - val_split
        
        # Split indices
        train_indices = indices[:train_split]
        val_indices = indices[train_split:train_val_split]
        test_indices = indices[train_val_split:]
        
        print(len(train_indices), len(val_indices), len(test_indices))
        norm_copy = dataset.X.copy()
        features = norm_copy.drop(columns=[dataset.durations, dataset.events]).columns
        #print(copy.columns)
        for every_column in features:
            unique_vals = set(norm_copy[every_column].unique())
            if unique_vals != {0, 1}:
                norm_copy[every_column] = (norm_copy[every_column] - norm_copy[every_column].mean()) / norm_copy[every_column].std()
        norm_copy = norm_copy.drop(columns=[dataset.durations, dataset.events])
        print(norm_copy.columns)
       # Create new datasets
        train_dataset = cls(
            norm_copy.iloc[train_indices], 
            dataset.X[durations].iloc[train_indices], 
            dataset.X[durations].iloc[train_indices]
        )
        
        val_dataset = cls(
            norm_copy.iloc[val_indices], 
            dataset.X[durations].iloc[val_indices], 
            dataset.X[durations].iloc[val_indices]
        )
        
        test_dataset = cls(
            norm_copy.iloc[test_indices], 
            dataset.X[durations].iloc[test_indices], 
            dataset.X[events].iloc[test_indices]
        )
        
        return train_dataset, val_dataset, test_dataset
    
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
        #self.scaler = StandardScaler()
        self.loss_history = {'train': [], 'val': []}
    def forward(self, x):
        return self.linear(x)
    
    def _negative_log_partial_likelihood(self, risk_scores, durations, events):
        # Sort by durations in descending order
        ordered_idx = np.argsort(durations)
        
        # Reorder risk_scores and events according to sorted durations
       
        risk_scores = risk_scores[ordered_idx] #beta*features
        events = events[ordered_idx]
        durations = durations[ordered_idx]
        
        # Calculate cumulative sum of exp(risk_scores)
        exp_risk_scores = torch.exp(risk_scores)
        #print(exp_risk_scores)
        cumsum_exp_risk = torch.cumsum(exp_risk_scores.flip(0), dim=0).flip(0)
        #print(cumsum_exp_risk)
        
        # Calculate individual likelihood contributions
        log_risk = torch.log(cumsum_exp_risk + 1e-8)  # Log of cumulative sum of risk scores at each time
        likelihood = risk_scores - log_risk  # Individual likelihood at each time
        #print(likelihood)
        # Zero out censored observations
        uncensored_likelihood = likelihood * events
        #print(uncensored_likelihood)
        # Calculate negative log likelihood
        logL = -torch.sum(uncensored_likelihood)
        #print(logL)
        
        # Normalize by number of events (non-censored)
        num_events = torch.sum(events)
        #print(num_events)
        
        risk_scores = logL / (num_events)  + 1e-8 # Add a small epsilon to avoid division by zero

        if num_events == 0:
            print("Warning: num_events is zero!")
             # Skip this batch or handle it in some other way

        return risk_scores # Add a small epsilon to avoid division by zero

    def fit(self, df, feature_cols, duration_col, event_col, 
        batch_size=64, learning_rate=0.005, num_epochs=150,
        test_size=0.2, val_size=0.2, random_state=100, verbose=True):
        """Fit the model using batch training"""
        self.loss_history = {'train': [], 'val': []}
        self.loss_history_detailed = {'train': [], 'val': [], 'epoch': []}
        
        # Prepare data
        train_dataset, val_dataset, test_dataset = SurvivalDataset.split_dataset(df, duration_col, event_col, test_size, val_size, random_state=100)
        #print(f"Train dataset size: {train_dataset.events.sum()}")
        #print(f"Validation dataset size: {len(val_dataset.X.columns)}")
        #print(f"Test dataset size: {len(test_dataset.X.columns)}")
        #print(train_dataset.X.values)
        # Convert to TensorDataset 
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(train_dataset.X.values, dtype=torch.float32),
                torch.tensor(train_dataset.durations.values, dtype=torch.float32),
                torch.tensor(train_dataset.events.values, dtype=torch.float32)
            ), 
            batch_size, 
            shuffle=False, drop_last=True
        )

        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(val_dataset.X.values, dtype=torch.float32),
                torch.tensor(val_dataset.durations.values, dtype=torch.float32),
                torch.tensor(val_dataset.events.values, dtype=torch.float32)
            ), 
            batch_size, 
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(test_dataset.X.values, dtype=torch.float32),
                torch.tensor(test_dataset.durations.values, dtype=torch.float32),
                torch.tensor(test_dataset.events.values, dtype=torch.float32)
            ), 
            batch_size
        )
                
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        patience = 100  # Early stopping patience
        
        for epoch in range(num_epochs):
            # Training
            #self.train()
            train_losses = []
            for batch_X, batch_durations, batch_events in train_loader:
                optimizer.zero_grad()
                risk_scores = self(batch_X)
                loss = self._negative_log_partial_likelihood(
                    risk_scores, batch_durations, batch_events
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
                
            # Validation
            self.eval()
            with torch.no_grad():
                val_losses = []
                for batch_X, batch_durations, batch_events in val_loader:
                    risk_scores = self(batch_X)
                    val_loss = self._negative_log_partial_likelihood(
                        risk_scores, batch_durations, batch_events
                    )
                    val_losses.append(val_loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            self.loss_history['train'].append(avg_train_loss)
            self.loss_history['val'].append(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = self.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            #print(f"Risk scores: {risk_scores}")

            
            if verbose and (epoch + 1) % 5 == 0:
                train_c_index = self.evaluate_batch(train_loader)
                val_c_index = self.evaluate_batch(val_loader)
                test_c_index = self.evaluate_batch(test_loader)
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(f'Train Loss: {avg_train_loss:.4f}')
                print(f'Val Loss: {avg_val_loss:.4f}')
                print(f'Train C-index: {train_c_index:.4f}')
                print(f'Val C-index: {val_c_index:.4f}')
                print(f'Test C-index: {test_c_index:.4f}')
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
        X = torch.FloatTensor(X)
        self.eval()
        with torch.no_grad():
            risk_scores = self(X)
        return risk_scores.numpy()
    
class DeepSurv(nn.Module):
    def __init__(self, n_in, hidden_layers_sizes=None, activation='relu', dropout=None, lr=0.001):
        super(DeepSurv, self).__init__()
        
        if hidden_layers_sizes is None:
            hidden_layers_sizes = [64, 32]
        
        layers = []
        prev_size = n_in
        for size in hidden_layers_sizes:
            layers.append(nn.Linear(prev_size, size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'selu':
                layers.append(nn.SELU())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            prev_size = size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
    
    def forward(self, x):
        return self.network(x)
    
    def _negative_log_likelihood(self, risk, E, T):
        sorted_indices = torch.argsort(T)
        T_sorted = T[sorted_indices]
        E_sorted = E[sorted_indices]
        risk_sorted = risk[sorted_indices]

        hazard_ratio = torch.exp(risk_sorted)
        log_risk = torch.log(torch.cumsum(hazard_ratio,0) + 1e-8)
        uncensored_likelihood = risk_sorted - log_risk
        censored_likelihood = uncensored_likelihood * E_sorted
        num_observed_events = torch.sum(E_sorted)
        # Including time factor (T) to model the risk more effectively
        likelihood = censored_likelihood * T_sorted
        return -torch.sum(likelihood) / num_observed_events + 1e-8

    def concordance_index(self, T, C, risk):
        """
        Compute the C-index for the given true event times (T) and predicted risks (log-hazard scores).
        
        Parameters:
        - T: True event times (1D array)
        - C: Event indicator (1D array: 1 if event occurred, 0 if censored)
        - risk: Predicted risk scores (1D array: predicted log-hazard ratios)
        
        Returns:
        - C-index score
        """
        #print(f"True event times shape: {T.shape}")
        #print(f"Predicted risk shape: {risk.shape}")

        # Convert numpy arrays to PyTorch tensors
        if isinstance(T, np.ndarray):
            T = torch.tensor(T, dtype=torch.float32)  # Convert to tensor if it's a numpy array
        if isinstance(C, np.ndarray):
            C = torch.tensor(C, dtype=torch.float32)  # Convert to tensor if it's a numpy array
        if isinstance(risk, np.ndarray):
            risk = torch.tensor(risk, dtype=torch.float32)  # Convert to tensor if it's a numpy array
        
        # Detach tensors to get rid of the computation graph (if needed)
        T = T.detach().cpu().numpy()
        C = C.detach().cpu().numpy()
        risk = risk.squeeze().detach().cpu().numpy()  # Flatten and remove unnecessary dimension

        return concordance_index(T, risk, C)

    def predict_risk(self, X):
        X = torch.FloatTensor(X)
        with torch.no_grad():
            risk = self(X)
        return risk.numpy()
    def fit(self, X, T, C, X_val=None, T_val=None, C_val=None, epochs=80, batch_size=1800):
        X = torch.FloatTensor(X)
        T = torch.FloatTensor(T)
        C = torch.FloatTensor(C)
        
        dataset = torch.utils.data.TensorDataset(X, T, C)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        train_losses = []
        val_losses = []
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, verbose=True)
        train_cindex = []
        val_cindex = []
        train_allrisk = []
        train_full_cindex_score = []
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for batch_X, batch_T, batch_C in dataloader:
                #print(f"Epoch {epoch}, Size = {len(batch_X)}")
                self.optimizer.zero_grad()
                risk = self(batch_X)
                loss = self._negative_log_likelihood(risk, batch_C, batch_T)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            train_losses.append(epoch_loss)
            #train_allrisk.append(risk)
            # Compute C-index for the current batch
            train_cindex_score = self.concordance_index(batch_T, batch_C, risk)
            train_cindex.append(train_cindex_score)
            
            if X_val is not None and T_val is not None and C_val is not None:
                self.eval()
                with torch.no_grad():
                    val_risk = self(torch.FloatTensor(X_val))
                    val_loss = self._negative_log_likelihood(val_risk, torch.FloatTensor(C_val), torch.FloatTensor(T_val))
                    val_losses.append(val_loss.item())

                    # Compute C-index for validation set
                    val_cindex_score = self.concordance_index(T_val, C_val, val_risk)
                    val_cindex.append(val_cindex_score)
                
                scheduler.step(val_loss)  # Step the scheduler after each epoch
        
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
                            f"Val C-index: {val_cindex_score:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")
        
        self.plot_loss(train_losses, val_losses, train_cindex, val_cindex)


    def predict(self, X):
        X = torch.FloatTensor(X)
        with torch.no_grad():
            risk = self(X)
        
        time_points = np.arange(0, 121, 6)  # 0 to 120 months (10 years) with 6-month increments
        survival_probabilities = np.exp(-np.exp(risk.numpy()) * time_points[:, None].T)
        
        return survival_probabilities
    
    def plot_loss(self, train_losses, val_losses, train_c_index, val_c_index):
        # Create a figure with 1 row and 2 columns for two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Loss
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot C-index
        ax2.plot(train_c_index, label='Training C-index')
        ax2.plot(val_c_index, label='Validation C-index')
        ax2.set_title('Training and Validation C-index')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('C-index')
        ax2.legend()
        ax2.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        plt.savefig('DeepSurv_Loss_and_Cindex.png')

    def plot_average_survival_curve(self, X_train):
        survival_probabilities = self.predict(X_train)
       
        average_survival_curve = np.mean(survival_probabilities, axis=0)
        
        time_points = np.arange(0, 121, 6)  # 0 to 120 months (10 years) with 6-month increments
        
        #plt.figure(figsize=(10, 6))
        plt.plot(time_points, average_survival_curve, color='blue', label='DeepSurv')
        #plt.title('Average Survival Curve Predicted by DeepSurv Model')
        #plt.xlabel('Time (months)')
        #plt.ylabel('Survival Probability')
        #plt.grid(True)
        #plt.savefig('Average_Survival_Curve.png')
        #plt.close()
    
    def save_model(self, path):
        """
        Save the model state dictionary and optimizer state
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'network_architecture': {
                'n_in': self.network[0].in_features,
                'hidden_layers_sizes': [layer.out_features for layer in self.network if isinstance(layer, nn.Linear)][:-1],
                'activation': 'relu',  # You might want to make this dynamic based on your initialization
                'dropout': self.network[2].p if any(isinstance(layer, nn.Dropout) for layer in self.network) else None
            }
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_model(cls, path):
        """
        Load a saved model
        """
        checkpoint = torch.load(path)
        
        # Create a new model instance with the saved architecture
        arch = checkpoint['network_architecture']
        model = cls(
            n_in=arch['n_in'],
            hidden_layers_sizes=arch['hidden_layers_sizes'],
            activation=arch['activation'],
            dropout=arch['dropout']
        )
        
        # Load the saved state dictionaries
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return model

class DeepSurvIPCW(nn.Module):
    def __init__(self, n_in, hidden_layers_sizes=None, activation='relu', dropout=None, lr=0.001):
        super(DeepSurvIPCW, self).__init__()
        
        if hidden_layers_sizes is None:
            hidden_layers_sizes = [64, 32]
        
        layers = []
        prev_size = n_in
        for size in hidden_layers_sizes:
            layers.append(nn.Linear(prev_size, size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'selu':
                layers.append(nn.SELU())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            prev_size = size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
    
    def forward(self, x):
        return self.network(x)
    
    def compute_ipcw_weights(self, X, C):
        """
        Compute the IPCW weights using a logistic regression model
        that estimates the probability of censoring.
        """
        # Create a logistic regression model to predict the censoring probability
        model = LogisticRegression(max_iter=1000)
        
        # Fit the model to the features and censoring indicator
        model.fit(X, C.numpy())  # X is the feature matrix, C is the censoring indicator (0 or 1)
        
        # Predict the probability of censoring
        censor_probs = model.predict_proba(X)[:, 1]  # Probability of being censored (P(C=1))
        
        # Compute IPCW weights: Inverse of the censoring probabilities
        ipc_weights = 1.0 / censor_probs
        return ipc_weights
    
    def _negative_log_likelihood(self, risk, E, T, ipc_weights):
        sorted_indices = torch.argsort(T)
        T_sorted = T[sorted_indices]
        E_sorted = E[sorted_indices]
        risk_sorted = risk[sorted_indices]

        hazard_ratio = torch.exp(risk_sorted)
        log_risk = torch.log(torch.cumsum(hazard_ratio, 0) + 1e-8)
        uncensored_likelihood = risk_sorted - log_risk
        censored_likelihood = uncensored_likelihood * E_sorted
        num_observed_events = torch.sum(E_sorted)

        # Including time factor (T) to model the risk more effectively
        likelihood = censored_likelihood * T_sorted
        
        # Apply IPCW weights: Modify likelihood by multiplying with IPCW weights
        weighted_likelihood = likelihood * ipc_weights[sorted_indices]
        
        # Return the negative log likelihood with IPCW correction
        return -torch.sum(weighted_likelihood) / num_observed_events + 1e-8

    def fit(self, X, T, C, X_val=None, T_val=None, C_val=None, epochs=80, batch_size=1800):
        X = torch.FloatTensor(X)
        T = torch.FloatTensor(T)
        C = torch.FloatTensor(C)
        
        # Compute IPCW weights
        ipc_weights = self.compute_ipcw_weights(X, C)
        
        dataset = torch.utils.data.TensorDataset(X, T, C)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        train_losses = []
        val_losses = []
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, verbose=True)
        train_cindex = []
        val_cindex = []
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for batch_X, batch_T, batch_C in dataloader:
                self.optimizer.zero_grad()
                risk = self(batch_X)
                
                # Compute IPCW weights for the batch
                batch_ipc_weights = ipc_weights[batch_X.numpy().argmax(axis=1)]  # Select IPCW weights for the batch
                
                # Calculate the loss with IPCW
                loss = self._negative_log_likelihood(risk, batch_C, batch_T, batch_ipc_weights)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            train_losses.append(epoch_loss)

            # Compute C-index for the current batch
            train_cindex_score = self.concordance_index(batch_T, batch_C, risk)
            train_cindex.append(train_cindex_score)
            
            if X_val is not None and T_val is not None and C_val is not None:
                self.eval()
                with torch.no_grad():
                    val_risk = self(torch.FloatTensor(X_val))
                    val_ipc_weights = self.compute_ipcw_weights(X_val, C_val)
                    val_loss = self._negative_log_likelihood(val_risk, torch.FloatTensor(C_val), torch.FloatTensor(T_val), val_ipc_weights)
                    val_losses.append(val_loss.item())

                    # Compute C-index for validation set
                    val_cindex_score = self.concordance_index(T_val, C_val, val_risk)
                    val_cindex.append(val_cindex_score)
                
                scheduler.step(val_loss)  # Step the scheduler after each epoch
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val C-index: {val_cindex_score:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")
        
        self.plot_loss(train_losses, val_losses, train_cindex, val_cindex)

    def concordance_index(self, T, C, risk):
        """
        Compute the concordance index (C-index) for the model predictions.
        
        Args:
        - T: True event times (numpy array or torch tensor).
        - C: Censoring indicators (numpy array or torch tensor).
        - risk: The predicted risk scores (numpy array or torch tensor).

        Returns:
        - c_index: The concordance index (a float value between 0 and 1).
        """
        # Ensure T, C, risk are tensors
        T = torch.tensor(T).float()
        C = torch.tensor(C).float()
        risk = torch.tensor(risk).float()
        
        # Initialize variables to count concordant, discordant, and tied pairs
        concordant = 0
        discordant = 0
        tied = 0
        n = len(T)
        
        # Compare every pair of individuals
        for i in range(n):
            for j in range(i+1, n):
                if T[i] != T[j]:  # Only consider pairs where event times are different
                    if C[i] == 1 and C[j] == 1:  # Both must be uncensored
                        # Compare predicted risks and actual event times
                        if risk[i] < risk[j] and T[i] < T[j]:
                            concordant += 1
                        elif risk[i] > risk[j] and T[i] > T[j]:
                            concordant += 1
                        else:
                            discordant += 1
                    elif C[i] == 1 or C[j] == 1:  # Handle censoring
                        tied += 1
        
        # Compute the C-index: concordant / (concordant + discordant + tied)
        c_index = concordant / (concordant + discordant + tied) if (concordant + discordant + tied) > 0 else 0
        return c_index
