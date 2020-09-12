import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn import linear_model as lm
from sklearn import preprocessing

full_data = pd.read_csv('https://raw.githubusercontent.com/benchang123/Ames-Housing/master/ames.csv')
training_data, test_data = train_test_split(full_data, random_state=42, test_size=0.2)

full_data.shape

#EDA

nanmean=training_data.isna().mean()*100
nan=nanmean[nanmean>25].sort_values(ascending=False)
print(nan)

training_data.columns.values

sns.jointplot(
    x='Gr_Liv_Area', 
    y='SalePrice', 
    data=training_data,
);

training_data.loc[training_data['Gr_Liv_Area']>5000,['Sale_Condition','SalePrice']]

def remove_outliers(data, variable, upper):
    return data.loc[(data[variable] < upper), :]

training_data = remove_outliers(training_data, 'Gr_Liv_Area', 5000)

training_data.dtypes.value_counts()

training_data.groupby('Neighborhood').size().sort_values(ascending=False).plot(kind='bar')

num_cols = training_data.dtypes[(training_data.dtypes == 'int64') | (training_data.dtypes == 'float64')].index
corr_df = training_data.loc[:,num_cols].corr()

sale_price_corr = corr_df['SalePrice'].drop('SalePrice',axis=0).sort_values(ascending=False)
ax = plt.subplots(figsize=(10,15))
ax = sns.barplot(y=sale_price_corr.keys(),x=sale_price_corr.values)
plt.xlabel("Correlation")
plt.ylabel("Feature")

noise = np.random.normal(0,0.5,training_data.shape[0])
training_data_2=training_data
training_data_2['Bedroom_AbvGr']=training_data_2['Bedroom_AbvGr']+noise
sns.scatterplot(data=training_data_2,x='Bedroom_AbvGr',y='SalePrice')

training_data_2=training_data
training_data_2['Overall_Qual']=training_data_2['Overall_Qual']+noise
sns.scatterplot(data=training_data_2,x='Overall_Qual',y='SalePrice')

def add_total_bathrooms(data):
    """
    Input:
      data (data frame): a data frame containing at least 4 numeric columns 
            Bsmt_Full_Bath, Full_Bath, Bsmt_Half_Bath, and Half_Bath
    """
    with_bathrooms = data.copy()
    bath_vars = ['Bsmt_Full_Bath', 'Full_Bath', 'Bsmt_Half_Bath', 'Half_Bath']
    weights = pd.Series([1, 1, 0.5, 0.5], index=bath_vars)
    with_bathrooms['TotalBathrooms']=with_bathrooms[bath_vars].fillna(0)@weights
    return with_bathrooms

training_data = add_total_bathrooms(training_data)

training_data_2=training_data
training_data_2['TotalBathrooms']=training_data_2['TotalBathrooms']+noise
sns.scatterplot(data=training_data_2,x='TotalBathrooms',y='SalePrice')

def find_rich_neighborhoods(data, n=3, metric=np.mean):
    neighborhoods = data.groupby('Neighborhood').agg(metric).sort_values('SalePrice',ascending=False).iloc[0:n].index.tolist()
    return neighborhoods

richhoods=training_data.groupby('Neighborhood').agg(np.mean).sort_values('SalePrice',ascending=False).iloc[0:20]
ax = plt.subplots(figsize=(5,8))
sns.barplot(y=richhoods.index,x=richhoods.SalePrice)

def add_in_rich_neighborhood(data, neighborhoods):
    data['in_rich_neighborhood'] = data['Neighborhood'].isin(neighborhoods).astype('category')
    return data

rich_neighborhoods = find_rich_neighborhoods(training_data, 4, np.mean)
training_data = add_in_rich_neighborhood(training_data, rich_neighborhoods)

def ohe_column(data,column):
    column_ohe=pd.get_dummies(data[column],drop_first=True,prefix=column)
    data=pd.concat([data, column_ohe], axis=1)
    return data

training_data=ohe_column(training_data,'Functional')
training_data=ohe_column(training_data,'Exter_Qual')

train_error_vs_N = []
cv_error_vs_N = []
linear_model=lm.LinearRegression()
range_of_num_features = range(1, sale_price_corr.shape[0] + 1)

for N in range_of_num_features:
    sale_price_corr_first_N_features = sale_price_corr.iloc[:N]
    saleprice=training_data['SalePrice'].drop(training_data.index[training_data[sale_price_corr.iloc[:N].index].isnull().any(1)])
    indepVar=training_data[sale_price_corr_first_N_features.index].dropna()
    
    cv_results = cross_validate(linear_model, indepVar, saleprice, cv=4,scoring=('r2', 'neg_root_mean_squared_error'),return_train_score=True)
    
    train_error_overfit =-np.mean(cv_results['train_neg_root_mean_squared_error'])
    test_error_overfit=-np.mean(cv_results['test_neg_root_mean_squared_error'])
    train_error_vs_N.append(train_error_overfit)
    cv_error_vs_N.append(test_error_overfit)
    
sns.lineplot(range_of_num_features, train_error_vs_N)
sns.lineplot(range_of_num_features, cv_error_vs_N)
plt.legend(["Training Error", "CV Error"])
plt.xlabel("Number of Features")
plt.ylabel("RMSE");

print(cv_error_vs_N[10:15])

features=sale_price_corr.iloc[:14]

plt.figure(figsize=(10,10))
sns.heatmap(training_data[features.index].corr(),annot=True)

colinear=['TotRms_AbvGrd','Garage_Area','Year_Remod/Add','Full_Bath','Garage_Yr_Blt']

def select_columns(data, columns):
    """Select only columns passed as arguments."""
    return data.loc[:, columns]

def process_data_fm(data):
    data = remove_outliers(data, 'Gr_Liv_Area', 5000)
    data = add_total_bathrooms(data)
    data=ohe_column(data,'Functional')
    data=ohe_column(data,'Exter_Qual')
    
    # Use rich_neighborhoods computed earlier to add in_rich_neighborhoods feature
    data = add_in_rich_neighborhood(data, rich_neighborhoods)
    
    # Transform Data, Select Features
    
    num_features=list(features.drop(colinear).index)
    other_features=['SalePrice', 
                   'TotalBathrooms', 
                   'in_rich_neighborhood',
                   'Exter_Qual_Fa', 
                'Exter_Qual_Gd', 
                'Exter_Qual_TA',
                'Functional_Min1',
                'Functional_Min2',
                'Functional_Mod',
                'Functional_Maj2',
                'Functional_Typ']
    overall_features=num_features+other_features
    
    data = select_columns(data, overall_features)
    # Return predictors and response variables separately
    X = data.drop(['SalePrice'], axis = 1)
    y = data.loc[:, 'SalePrice']
    return X, y

full_data = pd.read_csv('https://raw.githubusercontent.com/benchang123/Ames-Housing/master/ames.csv')

num_features=list(features.drop(colinear).index)
full_clean=full_data.drop(full_data.index[full_data[num_features].isnull().any(1)])

training_data, test_data = train_test_split(full_clean, random_state=42, test_size=0.2)

full_clean.shape

X_train,y_train=process_data_fm(training_data)
X_test,y_test=process_data_fm(test_data)

final_model = lm.LinearRegression()
final_model.fit(X_train, y_train)
y_predicted_train = final_model.predict(X_train)
y_predicted_test = final_model.predict(X_test)

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual-predicted)**2))

training_rmse = rmse(y_predicted_train, y_train)
test_rmse = rmse(y_predicted_test, y_test)
(round(training_rmse,2),round(test_rmse,2))

ax=sns.scatterplot(y_predicted_train,y_predicted_train-y_train,label="Training")
ax=sns.scatterplot(y_predicted_test,y_predicted_test-y_test,label="Test")
sns.lineplot([0,600000],0,color='red')

plt.legend(labels=['Training', 'Test'])
leg = ax.legend()
plt.xlabel('Predicted Sales Price')
plt.ylabel('RMSE')

sns.scatterplot(y_predicted_test,y_test)
sns.lineplot([0,600000],[0,600000],color='red')

plt.xlabel('Predicted Sales Price')
plt.ylabel('Actual Sales Price')

X_train_n,y_train_n=process_data_fm(training_data)
X_test_n,y_test_n=process_data_fm(test_data)

X_train_n = preprocessing.scale(X_train_n)
X_test_n = preprocessing.scale(X_test_n)

y_train_n=(y_train-np.mean(y_train))/np.std(y_train)
y_test_n=(y_test-np.mean(y_test))/np.std(y_test)

param_grid = {'alpha': [0.01, 0.1, 1., 5., 10., 25., 50., 100.]}
final_ridge = GridSearchCV(lm.Ridge(), cv=5, param_grid=param_grid, scoring='neg_mean_squared_error')
final_ridge.fit(X_train_n, y_train_n)
alpha = final_ridge.best_params_['alpha']
alpha

param_gridimp = {'alpha': list(np.linspace(1,10,200))}
final_ridgeimp = GridSearchCV(lm.Ridge(), cv=5, param_grid=param_gridimp, scoring='neg_mean_squared_error')
final_ridgeimp.fit(X_train_n, y_train_n)
alphaimp = final_ridgeimp.best_params_['alpha']
round(alphaimp,2)

y_ridge_train = final_ridgeimp.predict(X_train_n)
y_ridge_test = final_ridgeimp.predict(X_test_n)

ax=sns.scatterplot(y_ridge_train,y_ridge_train-y_train_n,label="Training")
ax=sns.scatterplot(y_ridge_test,y_ridge_test-y_test_n,label="Test")
sns.lineplot([-2,6],0,color='red')

plt.legend(labels=['Training', 'Test'])
leg = ax.legend()
plt.xlabel('Predicted Sales Price')
plt.ylabel('RMSE')

sns.scatterplot(y_ridge_test,y_test_n)
sns.lineplot([-2,5],[-2,5],color='red')

plt.xlabel('Predicted Sales Price')
plt.ylabel('Actual Sales Price')

param_grid = {'alpha': [0.01, 0.1, 1., 5., 10., 25., 50., 100.]}
final_lasso = GridSearchCV(lm.Lasso(), cv=5, param_grid=param_grid, scoring='neg_mean_squared_error')
final_lasso.fit(X_train_n, y_train_n)
alpha = final_lasso.best_params_['alpha']
round(alpha,2)

param_gridimp = {'alpha': list(np.linspace(0.005,0.1,100))}
final_lassoimp = GridSearchCV(lm.Lasso(), cv=5, param_grid=param_gridimp, scoring='neg_mean_squared_error')
final_lassoimp.fit(X_train_n, y_train_n)
alphaimp = final_lassoimp.best_params_['alpha']
round(alphaimp,2)

final_lassoimp.fit(X_train_n, y_train_n)
y_lasso_train = final_lassoimp.predict(X_train_n)
y_lasso_test = final_lassoimp.predict(X_test_n)

sns.lineplot([-2,6],0,color='red')
ax=sns.scatterplot(y_ridge_train,y_ridge_train-y_train_n,label="Training")
ax=sns.scatterplot(y_ridge_test,y_ridge_test-y_test_n,label="Test")

plt.legend(labels=['Training', 'Test'])
leg = ax.legend()
plt.xlabel('Predicted Sales Price')
plt.ylabel('RMSE')

sns.scatterplot(y_ridge_test,y_test_n)
sns.lineplot([-2,5],[-2,5],color='red')

plt.xlabel('Predicted Sales Price')
plt.ylabel('Actual Sales Price')