import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

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
        n_t = len(self.dataset[self.dataset[self.time_column] >= t])
        d_t = len(self.dataset[(self.dataset[self.time_column] == t) & (self.dataset[self.event_column] == 1)])
        prob_of_death = d_t/n_t if n_t > 0 else 0
        survival_probabilities.append(1 - prob_of_death)  # First probability

        for t in self.times[1:]:
            n_t = len(self.dataset[self.dataset[self.time_column] >= t])
            d_t = len(self.dataset[(self.dataset[self.time_column] == t) & (self.dataset[self.event_column] == 1)])
            prob_of_death = d_t/n_t if n > 0 else 0
            survival_probability_at_time_t = (1 - prob_of_death) * survival_probabilities[-1]
            survival_probabilities.append(survival_probability_at_time_t)
        return survival_probabilities

    def plot(self):
        plt.step(self.times, self.survival_probabilities, where='post')
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title('Kaplan-Meier Curve')
        plt.show()

dataset = pd.read_csv('/Users/rbhalerao/Desktop/CPH200B/heart_failure_clinical_records_dataset.csv')
km = KaplanMeier(dataset=dataset, 
                 time_column='time', 
                 event_column='DEATH_EVENT')

# Generate the plot
km.plot()
