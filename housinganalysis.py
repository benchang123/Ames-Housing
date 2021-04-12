import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn import linear_model as lm
from sklearn import preprocessing
from sklearn import metrics
from sklearn import tree
from sklearn import ensemble

full_data = pd.read_csv('https://raw.githubusercontent.com/benchang123/Ames-Housing/master/ames.csv')
training_data, test_data = train_test_split(full_data, random_state=42, test_size=0.2)

full_data.shape




#Remove features with lots of NA
nanmean=training_data.isna().mean()*100
nan=nanmean[nanmean>25].sort_values(ascending=False)
print(nan)

training_data.drop(columns=nan.index,inplace=True)

#Number of sold houses per year
year=training_data.groupby('Yr_Sold').count()['Order'].plot(kind='bar')
year

#outliers for living area vs price

training_data.columns.values

sns.jointplot(
    x='Gr_Liv_Area', 
    y='SalePrice', 
    data=training_data,
);

#outliers for basement area vs price

training_data.columns.values

sns.jointplot(
    x='Total_Bsmt_SF', 
    y='SalePrice', 
    data=training_data,
);

#2 SD for sales price

salepricemean=np.mean(training_data['SalePrice'])
salepricestd=np.std(training_data['SalePrice'])

print('Mean Sales Price: ', salepricemean)
print('STD Sales Price: ', salepricestd)

salepricerange=(salepricemean-(2*salepricestd),salepricemean+(2*salepricestd))
salepricerange

#check outliers

training_data.loc[training_data['Gr_Liv_Area']>5000,['Sale_Condition','SalePrice']]

#check outliers

training_data.loc[training_data['Total_Bsmt_SF']>3000,['Total_Bsmt_SF','SalePrice']]

def remove_outliers(data, variable, upper):
    return data.loc[(data[variable] < upper), :]

training_data = remove_outliers(training_data, 'Gr_Liv_Area', 5000)

training_data = remove_outliers(training_data, 'Total_Bsmt_SF', 3000)


#check variable type
training_data.dtypes.value_counts()


#check neighborhood histogram
training_data.groupby('Neighborhood').size().sort_values(ascending=False).plot(kind='bar')


#Check correlation to sales price
num_cols = training_data.dtypes[(training_data.dtypes == 'int64') | (training_data.dtypes == 'float64')].index
corr_df = training_data.loc[:,num_cols].corr()

sale_price_corr = corr_df['SalePrice'].drop('SalePrice',axis=0).sort_values(ascending=False)
ax = plt.subplots(figsize=(10,15))
ax = sns.barplot(y=sale_price_corr.keys(),x=sale_price_corr.values)
plt.xlabel("Correlation")
plt.ylabel("Feature")


#bedroom
noise = np.random.normal(0,0.5,training_data.shape[0])
training_data_2=training_data
training_data_2['Bedroom_AbvGr']=training_data_2['Bedroom_AbvGr']+noise
sns.scatterplot(data=training_data_2,x='Bedroom_AbvGr',y='SalePrice')

#overall quality
training_data_2=training_data
training_data_2['Overall_Qual']=training_data_2['Overall_Qual']+noise
plt.figure()
sns.scatterplot(data=training_data_2,x='Overall_Qual',y='SalePrice')


#feature engineering
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

#check scatter plot
training_data_2=training_data
training_data_2['TotalBathrooms']=training_data_2['TotalBathrooms']+noise
sns.scatterplot(data=training_data_2,x='TotalBathrooms',y='SalePrice')



def find_rich_neighborhoods(data, n=3, metric=np.mean):
    neighborhoods = data.groupby('Neighborhood').agg(metric).sort_values('SalePrice',ascending=False).iloc[0:n].index.tolist()
    return neighborhoods

richhoods=training_data.groupby('Neighborhood').agg(np.mean).sort_values('SalePrice',ascending=False).iloc[0:20]
plt.subplots(figsize=(5,8))
sns.barplot(y=richhoods.index,x=richhoods.SalePrice)
plt.xticks(rotation=45)

def add_in_rich_neighborhood(data, neighborhoods):
    data['in_rich_neighborhood'] = data['Neighborhood'].isin(neighborhoods).astype('category')
    return data

rich_neighborhoods = find_rich_neighborhoods(training_data, 4, np.mean)
training_data = add_in_rich_neighborhood(training_data, rich_neighborhoods)



categorical = (training_data.dtypes == "object")
categorical_list = list(categorical[categorical].index)
print(categorical_list)

def encode():
    for i in categorical_list:
        encode=preprocessing.LabelEncoder()
        training_data[i]=encode.fit_transform(training_data[i])
encode()

#Feature Importance (Tree)

X=training_data.drop(columns=['SalePrice'])
Y=training_data['SalePrice']

X=X.fillna(method="pad")

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
features = X.columns
importances = clf.feature_importances_
idx = np.argsort(importances)

plt.figure(figsize=(15,10))
plt.bar(np.arange(len(idx)),importances[idx])
plt.xticks(range(len(idx)), [features[i] for i in idx], rotation='vertical')

plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Importance of Feature')

#Feature Importance (Gradient Boosting)

clf = ensemble.GradientBoostingClassifier()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
features = X.columns
importances = clf.feature_importances_
idx = np.argsort(importances)

plt.figure(figsize=(15,10))
plt.bar(np.arange(len(idx)),importances[idx])
plt.xticks(range(len(idx)), [features[i] for i in idx], rotation='vertical')

plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Importance of Feature')



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






# # storage vectors
# features=82
# rmse = np.zeros(features-3)

# # loop over the Ks
# trainingdata_nosale=training_data.drop(columns=['SalePrice'])

# for i in range(3,features):
#     trainingx=trainingdata_nosale.iloc[:,2:i]
#     trainingy=training_data[['SalePrice']]

#     linear_model=lm.LinearRegression()
#     linear_model.fit(trainingx, trainingy)
#     ypred = linear_model.predict(trainingx)

#     rmse[i-1]=metrics.mean_squared_error(ypred, trainingy, squared=False)

# #graph
# plt.figure(figsize=(10,5))
# plt.plot(np.arange(1,features),rmse)
# plt.xlabel('Number of Features')
# plt.ylabel('RMSE')
# plt.title('RMSE versus Number of Features')
# plt.show()









#modeling

def select_columns(data, columns):
    """Select only columns passed as arguments."""
    return data.loc[:, columns]

def process_data_fm(data):
    data = remove_outliers(data, 'Gr_Liv_Area', 5000)
    data = add_total_bathrooms(data)
    
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

training_rmse = metrics.mean_squared_error(y_predicted_train, y_train,squared=False)
test_rmse = metrics.mean_squared_error(y_predicted_test, y_test,squared=False)
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