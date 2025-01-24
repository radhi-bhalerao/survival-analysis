import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from memory_profiler import profile
from lifelines import KaplanMeierFitter, ExponentialFitter
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
        print(lambda_rate)
        survival_probabilities = np.exp( - self.time_values*lambda_rate)
        return survival_probabilities


    def plot(self):
        plt.figure()
        plt.step(self.time_values, self.survival_probabilities, where='post')
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title('Exponential Parametric Survival Curve')
        plt.savefig('ExponentialParametricSurvivalCurve.png')

dataset = pd.read_csv('/Users/rbhalerao/Desktop/CPH200B/heart_failure_clinical_records_dataset.csv')

km = KaplanMeier(dataset=dataset, 
                 time_column='time', 
                 event_column='DEATH_EVENT')

# Generate the plot
km.plot()

plt.figure(2)
#Using lifelines package
kmf = KaplanMeierFitter()
kmf.fit(durations=dataset['time'], event_observed=dataset['DEATH_EVENT'])
kmf.survival_function_
kmf.cumulative_density_
kmf.plot_survival_function()
plt.savefig('KaplanMeierByLifelines.png')

plt.figure(3)
exf = ExponentialFitter().fit(durations=dataset['time'], event_observed=dataset['DEATH_EVENT'], label='ExponentialFitter')
exf.plot_survival_function()
print(exf.lambda_)
plt.savefig('ExponentialFitter.png')

plt.figure(4)
exp = ExponentialSurvivalCurve(dataset=dataset,time_column='time', 
                 event_column='DEATH_EVENT')
exp.plot()
