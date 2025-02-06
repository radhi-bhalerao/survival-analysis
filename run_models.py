import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from memory_profiler import profile
from lifelines import KaplanMeierFitter, ExponentialFitter, CoxPHFitter
from scipy.optimize import minimize
from models import KaplanMeier, ExponentialSurvivalCurve, MahalanobisKNearestNeighbor, CoxPHModel, SurvivalDataset, DeepSurv, DeepSurvIPCW
from pycox import datasets as dfp
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

#PART 1.1
#Loading the dataset    
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
plt.savefig('ExponentialFitter.png')

plt.figure(4)
exp = ExponentialSurvivalCurve(dataset=dataset,time_column='time',  event_column='DEATH_EVENT')
exp.plot()


#Mahalanobis K-Nearest Neighbors
features = dataset.drop(columns=['time', 'DEATH_EVENT'])

# Create and fit the model
MKNN_model = MahalanobisKNearestNeighbor(dataset, 'time', 'DEATH_EVENT', k=5)
MKNN_model.fit(features)

# Predict survival curves for the first 5 observations
survival_curves = MKNN_model.plot_survival_curves(MKNN_model.test[0:5], title="Survival Curves for Test Data")

c_index = MKNN_model.calculate_concordance_index(MKNN_model.test, verbose=True)
print(c_index)


heart_failure_dataset = SurvivalDataset(dataset, 'time', 'DEATH_EVENT')

Cox_model = CoxPHModel(heart_failure_dataset.__len__()-2)
Cox_model.fit(heart_failure_dataset, heart_failure_dataset.get_features(), 'time', 'DEATH_EVENT', batch_size = 24, learning_rate=0.001, num_epochs=200)
Cox_model.report_results(feature_names=heart_failure_dataset.get_features())
Cox_model.plot_loss_curve()



unos_dataset = pd.read_csv('/Users/rbhalerao/Desktop/CPH200B/UNOS_train.csv').drop(columns=['Unnamed: 0', 'wl_id_code','days_stat1', 'days_stat1a', 'days_stat2', 'days_stat1b', 'init_date', 'end_date', 'prev_tx', 'num_prev_tx'])
print(unos_dataset.columns)
print(unos_dataset.shape)
unos_class = SurvivalDataset(unos_dataset, 'Survival Time', 'Censor (Censor = 1)')
train_dataset, val_dataset, test_dataset = SurvivalDataset.split_dataset(unos_class, 'Survival Time', 'Censor (Censor = 1)', test_size=0.2, val_size=0.2, random_state=142)

X_train = train_dataset.X.to_numpy()
T_train = train_dataset.durations.to_numpy()
C_train = train_dataset.events.to_numpy()
print(X_train.shape)
print(T_train.shape)
print(C_train.shape)

X_val = val_dataset.X.to_numpy()
T_val = val_dataset.durations.to_numpy()
C_val = val_dataset.events.to_numpy()
print(X_val.shape)
print(T_val.shape)
print(C_val.shape)
X_test = test_dataset.X.to_numpy()
T_test = test_dataset.durations.to_numpy()
C_test = test_dataset.events.to_numpy()
print(X_test.shape)
print(T_test.shape)
print(C_test.shape)
# Filter out negative survival times in the training dataset
train_filter = (T_train >= 0)  & (T_train <=120) # Ensure survival times are non-negative
X_train_filtered = X_train[train_filter]
T_train_filtered = T_train[train_filter]
C_train_filtered = C_train[train_filter]

# Filter out negative survival times in the validation dataset
val_filter = (T_val >= 0) & (T_val <=120)
X_val_filtered = X_val[val_filter]
T_val_filtered = T_val[val_filter]
C_val_filtered = C_val[val_filter]

# Filter out negative survival times in the test dataset
test_filter = (T_test >= 0) & (T_test <=120)
X_test_filtered = X_test[test_filter]
T_test_filtered = T_test[test_filter]
C_test_filtered = C_test[test_filter]

print(len(X_train_filtered))
model = DeepSurv(n_in=X_train.shape[1], dropout=0.5, lr=1e-2, activation='selu', hidden_layers_sizes=[64,64])
model.fit(X_train_filtered, T_train_filtered, C_train_filtered, X_val_filtered, T_val_filtered, C_val_filtered)

# After training
model.save_model('deepsurv_model.pth')

plt.figure()
model.plot_average_survival_curve(X_test)
kmf_unos = KaplanMeierFitter()
kmf_unos.fit(durations=T_train_filtered, event_observed=C_train_filtered)
kmf_unos.plot(label='Kaplan Meier', color='orange')
plt.legend()
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.savefig('KaplanMeierByLifelines_UNOSwithSurvivalCurve.png')

print('Test C-Index:', model.concordance_index(T_test_filtered, C_test_filtered, model.predict_risk(X_test_filtered)))


#PART 1.4
# Load the dataset (e.g., flchain)
metabric_df = dfp.metabric.read_df()
print(metabric_df.columns)
cox = CoxPHFitter()
cox.fit(metabric_df, duration_col='duration', event_col='event')

df_censored = metabric_df[metabric_df['event'] == 0]
df_uncensored = metabric_df[metabric_df['event'] == 1]
metabric_censored = KaplanMeierFitter()
metabric_uncensored = KaplanMeierFitter()


# Fit Kaplan-Meier estimator
metabric_censored.fit(df_censored['duration'])
metabric_uncensored.fit(df_uncensored['duration'])

# Plot the survival curves
plt.figure(figsize=(10, 6))
metabric_censored.plot(label='Censored', color='blue')
metabric_uncensored.plot(label='Uncensored', color='red')
plt.title('Kaplan-Meier Survival Curve (Censored vs Uncensored)')
plt.savefig('KaplanMeier_Censored_vs_Uncensored.png')

# Statistical tests for differences between censored and uncensored covariates
# Perform t-test for a specific feature 
for feature in df_censored.columns:
    censored_group = df_censored[feature]
    uncensored_group = df_uncensored[feature]
    t_stat, p_val = ttest_ind(censored_group, uncensored_group)
    print(f"T-test p-value for {feature}: {p_val}")

# Create synthetic censoring based 
np.random.seed(142)
bias_factor = 0.8 + 0.01 * metabric_df['x0']  

# Generate censoring times by adding random noise
censoring_times = np.random.exponential(scale=100, size=len(metabric_df))
censored = np.random.rand(len(metabric_df)) < 1 / bias_factor

# Create synthetic survival times with censoring
metabric_df['duration_synth'] = np.minimum(metabric_df['duration'], censoring_times)
metabric_df['event_synth'] = np.where(censored, 0, metabric_df['event'])  # Set event to 0 (censored) when censoring occurs

# Shuffle dataset to preserve randomness
df_synth = shuffle(metabric_df)

df_synth.to_csv('metabric_synth.csv', index=False)

X = df_synth.drop(columns=['duration_synth', 'event_synth']).to_numpy()
T = metabric_df['duration_synth'].values
C = metabric_df['event_synth'].values

# Initialize the model
n_in = X.shape[1]  # Number of features in the dataset
model = DeepSurvIPCW(n_in)
orig_model = DeepSurv(n_in)
# Fit the model to the data (you can specify the number of epochs and batch_size)
model.fit(X, T, C, epochs=100, batch_size=64)
orig_model.fit(X, T, C, epochs=100, batch_size=64)
# Make predictions (predicted risk scores)
pred_risk = model.predict_risk(X)
orig_pred_risk = orig_model.predict_risk(X)
# Calculate the C-index for the full dataset (predicted risk vs. true event times and censoring indicator)
c_index = model.concordance_index(T, C, pred_risk)
orig_c_index = orig_model.concordance_index(T, C, orig_pred_risk)

# Print the C-index
print(f"Modified Model C-index: {c_index}", f"Original Model C-index: {orig_c_index}")
