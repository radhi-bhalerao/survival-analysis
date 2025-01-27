#Loading the dataset    
dataset = pd.read_csv('/Users/rbhalerao/Desktop/CPH200B/heart_failure_clinical_records_dataset.csv')
print(len(dataset))
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

#Mahalanobis K-Nearest Neighbors
features = dataset.drop(columns=['time', 'DEATH_EVENT'])

# Create and fit the model
MKNN_model = MahalanobisKNearestNeighbor(dataset, 'time', 'DEATH_EVENT', k=5)
MKNN_model.fit(features)

# Predict survival curves for the first 5 observations
survival_curves = MKNN_model.plot_survival_curves(MKNN_model.test, title="Survival Curves for Test Data")