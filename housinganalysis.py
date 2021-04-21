import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn import linear_model as lm
from sklearn import preprocessing
from sklearn import metrics
from sklearn import ensemble

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

full_data = pd.read_csv('https://raw.githubusercontent.com/benchang123/Ames-Housing/master/ames.csv')
training_data, test_data = train_test_split(full_data, random_state=42, test_size=0.2)

full_data.shape


############ EDA #########################

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

#2 SD for sales price

salepricemean=np.mean(training_data['SalePrice'])
salepricestd=np.std(training_data['SalePrice'])

print('Mean Sales Price: ', salepricemean)
print('STD Sales Price: ', salepricestd)

salepricerange=(salepricemean-(2*salepricestd),salepricemean+(2*salepricestd))
salepricerange

plt.subplots(figsize=(10,8))
plt.hist(training_data['SalePrice'])
plt.xlabel("Sales Price")
plt.ylabel("Frequency")

#check outliers

training_data.loc[training_data['Gr_Liv_Area']>4000,['Gr_Liv_Area','SalePrice']]

def remove_outliers(data, variable, upper):
    return data.loc[(data[variable] < upper), :]

training_data = remove_outliers(training_data, 'Gr_Liv_Area', 4000)


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
plt.figure()
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


def add_total_SF(data):
    totalSFdf = data.copy()
    totalSFdf['Total_SF'] = totalSFdf.Total_Bsmt_SF + totalSFdf.Gr_Liv_Area
    return totalSFdf

training_data = add_total_SF(training_data)

def find_rich_neighborhoods(data, n=3, metric=np.mean):
    neighborhoods = data.groupby('Neighborhood').agg(metric).sort_values('SalePrice',ascending=False).iloc[0:n].index.tolist()
    return neighborhoods

richhoods=training_data.groupby('Neighborhood').agg(np.mean).sort_values('SalePrice',ascending=False).iloc[0:20]
plt.subplots(figsize=(5,8))
sns.barplot(y=richhoods.index,x=richhoods.SalePrice)
plt.xticks(rotation=45)

def add_in_rich_neighborhood(data, neighborhoods):
    data['in_rich_neighborhood'] = data['Neighborhood'].isin(neighborhoods).astype(int)
    return data

rich_neighborhoods = find_rich_neighborhoods(training_data, 4, np.mean)
training_data = add_in_rich_neighborhood(training_data, rich_neighborhoods)


categorical = (training_data.dtypes == "object")
categorical_list = list(categorical[categorical].index)
print(categorical_list)

def encode(data):
    categorical = (data.dtypes == "object")
    categorical_list = list(categorical[categorical].index)
    for i in categorical_list:
        encode=preprocessing.LabelEncoder()
        data[i]=encode.fit_transform(data[i])
    return data
training_data=encode(training_data)

#Feature Importance (RF)

X=training_data.drop(columns=['SalePrice'])
Y=training_data['SalePrice']

for i in range(len(num_cols)-1):
    meanvar=np.nanmean(X[num_cols[i]])
    X[num_cols[i]].fillna(meanvar,inplace=True)

X=X.fillna(method="pad")

clf = ensemble.RandomForestClassifier()
clf = clf.fit(X, Y)
features = X.columns
importances = clf.feature_importances_
idxrf = np.argsort(importances)[::-1]

plt.figure(figsize=(15,10))
plt.bar(np.arange(len(idxrf)),importances[idxrf])
plt.xticks(range(len(idxrf)), [features[i] for i in idxrf], rotation='vertical')

plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Importance of Feature')

#number of features based on trees

features=79
rmse = np.zeros(features-1)

# loop over the Ks

train_error_vs_N = []
cv_error_vs_N = []

for i in range(1,features):
    trainingx=X.iloc[:,idxrf[0:i]]
    trainingy=Y

    linear_model=lm.LinearRegression()
    
    cv_results = cross_validate(linear_model, trainingx, trainingy, 
                                cv=5,scoring=('r2', 'neg_root_mean_squared_error'),
                                return_train_score=True)
    
    train_error_overfit =-np.mean(cv_results['train_neg_root_mean_squared_error'])
    test_error_overfit=-np.mean(cv_results['test_neg_root_mean_squared_error'])
    train_error_vs_N.append(train_error_overfit)
    cv_error_vs_N.append(test_error_overfit)

plt.figure(figsize=(10,7))
sns.lineplot(np.arange(1,features), train_error_vs_N)
sns.lineplot(np.arange(1,features), cv_error_vs_N)
plt.legend(["Training Error", "CV Error"])
plt.xlabel("Number of Features")
plt.ylabel("RMSE");
plt.title('Importance of Feature (Random Forest)')

print(cv_error_vs_N[5:15])

numfeaturesrf=13

#Feature Importance (Gradient Boosting)

clf = ensemble.GradientBoostingClassifier(n_estimators=25,verbose=3)
clf = clf.fit(X, Y)
features = X.columns
importances = clf.feature_importances_
idxgb = np.argsort(importances)[::-1]

plt.figure(figsize=(15,10))
plt.bar(np.arange(len(idxgb)),importances[idxgb])
plt.xticks(range(len(idxgb)), [features[i] for i in idxgb], rotation='vertical')

plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Importance of Feature')

#number of features based on gb

features=79
rmse = np.zeros(features-1)

# loop over the Ks

train_error_vs_N = []
cv_error_vs_N = []

for i in range(1,features):
    trainingx=X.iloc[:,idxgb[0:i]]
    trainingy=Y

    linear_model=lm.LinearRegression()
    
    cv_results = cross_validate(linear_model, trainingx, trainingy, 
                                cv=5,scoring=('r2', 'neg_root_mean_squared_error'),
                                return_train_score=True)
    
    train_error_overfit =-np.mean(cv_results['train_neg_root_mean_squared_error'])
    test_error_overfit=-np.mean(cv_results['test_neg_root_mean_squared_error'])
    train_error_vs_N.append(train_error_overfit)
    cv_error_vs_N.append(test_error_overfit)

plt.figure(figsize=(10,7))
sns.lineplot(np.arange(1,features), train_error_vs_N)
sns.lineplot(np.arange(1,features), cv_error_vs_N)
plt.legend(["Training Error", "CV Error"])
plt.xlabel("Number of Features")
plt.ylabel("RMSE");
plt.title('Importance of Feature (Gradient Boosting)')

print(cv_error_vs_N[10:20])

numfeaturesgb=20


#number of features based on corr

train_error_vs_N = []
cv_error_vs_N = []
linear_model=lm.LinearRegression()

num_cols = training_data.dtypes[(training_data.dtypes == 'int64') | (training_data.dtypes == 'float64')].index
corr_df = training_data.loc[:,num_cols].corr()
sale_price_corr = corr_df['SalePrice'].drop('SalePrice',axis=0).sort_values(ascending=False)

range_of_num_features = range(1, sale_price_corr.shape[0] + 1)

for N in range_of_num_features:
    sale_price_corr_first_N_features = sale_price_corr.iloc[:N]
    saleprice=training_data['SalePrice'].drop(training_data.index
                                              [training_data[sale_price_corr.iloc[:N].index]
                                               .isnull().any(1)])
    indepVar=training_data[sale_price_corr_first_N_features.index].dropna()
    
    scaler=preprocessing.StandardScaler()
    indepVar = pd.DataFrame(scaler.fit_transform(indepVar),columns = indepVar.columns)
    
    cv_results = cross_validate(linear_model, indepVar, saleprice, cv=4,
                                scoring=('r2', 'neg_root_mean_squared_error'),
                                return_train_score=True)
    
    train_error_overfit =-np.mean(cv_results['train_neg_root_mean_squared_error'])
    test_error_overfit=-np.mean(cv_results['test_neg_root_mean_squared_error'])
    train_error_vs_N.append(train_error_overfit)
    cv_error_vs_N.append(test_error_overfit)

plt.figure(figsize=(10,7))
sns.lineplot(range_of_num_features, train_error_vs_N)
sns.lineplot(range_of_num_features, cv_error_vs_N)
plt.legend(["Training Error", "CV Error"])
plt.xlabel("Number of Features")
plt.ylabel("RMSE");

print(cv_error_vs_N[10:20])

numfeaturescorr=16

#multicorr test

features=sale_price_corr.iloc[:numfeaturescorr]

plt.figure(figsize=(10,10))
sns.heatmap(training_data[features.index].corr(),annot=True)

colinear=['TotRms_AbvGrd','Garage_Area','Year_Remod/Add','Full_Bath','Garage_Yr_Blt']

idxcorr=[]

for i in range(len(features.index.to_list())):
    idxcorr.append(training_data.columns.get_loc(sale_price_corr.index.to_list()[i]))
idxcorr.insert(0,76)
idxcorr=np.array(idxcorr)


#### MODELING ##################

def select_columns(data, columns):
    """Select only columns passed as arguments."""
    return data.iloc[:, columns]

def process_data_fm(data, overall_features):
    data = remove_outliers(data, 'Gr_Liv_Area', 4000)
    data = add_total_bathrooms(data)
    
    # Transform Data, Select Features
    data = add_in_rich_neighborhood(data, rich_neighborhoods)
    data = select_columns(data, overall_features)
        
    # Return predictors and response variables separately
    X = data.drop(['SalePrice'], axis = 1)
    
    numf=X.dtypes[(X.dtypes == 'int64') | (X.dtypes == 'float64')].index
    scaler=preprocessing.StandardScaler()
    X.loc[:,numf] = scaler.fit_transform(X.loc[:,numf])
    X = encode(X)
    
    y = data.loc[:, 'SalePrice']
    X=X.fillna(method="pad")
    X=X.fillna(method="bfill")
    return X, y

#### OLS ###########

def OLSrun(X_train_n, y_train, X_test_n, y_test):  

    final_model = lm.LinearRegression()
    final_model.fit(X_train_n, y_train)
    y_predicted_train = final_model.predict(X_train_n)
    y_predicted_test = final_model.predict(X_test_n)
    
    training_rmse = metrics.mean_squared_error(y_predicted_train, y_train,squared=False)
    test_rmse = metrics.mean_squared_error(y_predicted_test, y_test,squared=False)
    print('Training and Test Error:',round(training_rmse,2),round(test_rmse,2))
    
    #Line Plot
    plt.figure(figsize=(10,7))
    sns.scatterplot(y_predicted_test,y_test)
    sns.lineplot([0,600000],[0,600000],color='red')
    
    plt.xlabel('Predicted Sales Price')
    plt.ylabel('Actual Sales Price')
    
    #Residual Plots
    plt.figure(figsize=(10,7))
    sns.residplot(y_predicted_train,y_train,label="Training")
    plt.xlabel('Predicted Sales Price')
    plt.ylabel('RMSE')
    plt.title('Residual Plot (Training)')
    
    plt.figure(figsize=(10,7))
    sns.residplot(y_predicted_test,y_test,label="Test")
    plt.xlabel('Predicted Sales Price')
    plt.ylabel('RMSE')
    plt.title('Residual Plot (Test)')


##### Ridge ############

def ridgerun(X_train_n, y_train_n, X_test_n, y_test_n):
    
    param_grid = {'alpha': [0.01, 0.1, 1., 5., 10., 25., 50., 100.]}
    final_ridge = GridSearchCV(lm.Ridge(), cv=5, param_grid=param_grid, scoring='neg_mean_squared_error')
    final_ridge.fit(X_train_n, y_train_n)
    alpha = final_ridge.best_params_['alpha']
    print('Initial Best Alpha', alpha)
    
    param_gridimp = {'alpha': list(np.linspace(alpha-(alpha*0.2),alpha+(alpha*0.2),200))}
    final_ridgeimp = GridSearchCV(lm.Ridge(), cv=5, param_grid=param_gridimp, scoring='neg_mean_squared_error')
    final_ridgeimp.fit(X_train_n, y_train_n)
    alphaimp = final_ridgeimp.best_params_['alpha']
    print('Improved Best Alpha', round(alphaimp,2))
    
    
    y_ridge_train = final_ridgeimp.predict(X_train_n)
    y_ridge_test = final_ridgeimp.predict(X_test_n)
    
    training_rmse = metrics.mean_squared_error(y_ridge_train, y_train_n,squared=False)
    test_rmse = metrics.mean_squared_error(y_ridge_test, y_test_n,squared=False)
    print('Training and Test Error:',round(training_rmse,2),round(test_rmse,2))
    
    
    #Line Plot
    plt.figure(figsize=(10,7))
    sns.scatterplot(y_ridge_test,y_test_n)
    sns.lineplot([0,600000],[0,600000],color='red')
    
    plt.xlabel('Predicted Sales Price')
    plt.ylabel('Actual Sales Price')
    
    #Residual Plots
    plt.figure(figsize=(10,7))
    sns.residplot(y_ridge_train,y_train_n,label="Training")
    plt.xlabel('Predicted Sales Price')
    plt.ylabel('RMSE')
    plt.title('Residual Plot (Training)')
    
    plt.figure(figsize=(10,7))
    sns.residplot(y_ridge_test,y_test_n,label="Test")
    plt.xlabel('Predicted Sales Price')
    plt.ylabel('RMSE')
    plt.title('Residual Plot (Test)')
    
    #Feature Importance
    ridge=final_ridgeimp.best_estimator_
    
    coefs = pd.DataFrame({'coefs':ridge.coef_}, index=X_train_n.columns)
    coefs['coefs_abs'] = np.abs(coefs.coefs)
    
    top_coefs = coefs.sort_values('coefs_abs', ascending=False).head(10)
    plt.figure(figsize=(8,10))
    sns.barplot( top_coefs.coefs_abs, top_coefs.index)
    plt.title('Ridge Regression: Top Features')
    plt.xlabel('Absolute Coefficient')
    plt.show()


##### LASSO ############

def lassorun(X_train_n, y_train_n, X_test_n, y_test_n):  

    param_grid = {'alpha': [0.01, 0.1, 1., 5., 10., 25., 50., 100., 500., 1000.]}
    final_lasso = GridSearchCV(lm.Lasso(), cv=5, param_grid=param_grid, scoring='neg_mean_squared_error')
    final_lasso.fit(X_train_n, y_train_n)
    alpha = final_lasso.best_params_['alpha']
    print('Initial Best Alpha', alpha)
    
    param_gridimp = {'alpha': list(np.linspace(alpha-(alpha*0.2),alpha+(alpha*0.2),1000))}
    final_lassoimp = GridSearchCV(lm.Lasso(), cv=5, param_grid=param_gridimp, scoring='neg_mean_squared_error')
    final_lassoimp.fit(X_train_n, y_train_n)
    alphaimp = final_lassoimp.best_params_['alpha']
    print('Improved Best Alpha', round(alphaimp,2))
    
    y_lasso_train = final_lassoimp.predict(X_train_n)
    y_lasso_test = final_lassoimp.predict(X_test_n)
    
    training_rmse = metrics.mean_squared_error(y_lasso_train, y_train_n,squared=False)
    test_rmse = metrics.mean_squared_error(y_lasso_test, y_test_n,squared=False)
    print('Training and Test Error:',round(training_rmse,2),round(test_rmse,2))
    
    
    #Line Plot
    plt.figure(figsize=(10,7))
    sns.scatterplot(y_lasso_test,y_test_n)
    sns.lineplot([0,600000],[0,600000],color='red')
    
    plt.xlabel('Predicted Sales Price')
    plt.ylabel('Actual Sales Price')
    
    #Residual Plots
    plt.figure(figsize=(10,7))
    sns.residplot(y_lasso_train,y_train_n,label="Training")
    plt.xlabel('Predicted Sales Price')
    plt.ylabel('RMSE')
    plt.title('Residual Plot (Training)')
    
    plt.figure(figsize=(10,7))
    sns.residplot(y_lasso_test,y_test_n,label="Test")
    plt.xlabel('Predicted Sales Price')
    plt.ylabel('RMSE')
    plt.title('Residual Plot (Test)')
    
    #Feature Importance
    lasso=final_lassoimp.best_estimator_
    
    coefs = pd.DataFrame({'coefs':lasso.coef_}, index=X_train_n.columns)
    coefs['coefs_abs'] = np.abs(coefs.coefs)
    
    top_coefs = coefs.sort_values('coefs_abs', ascending=False).head(10)
    plt.figure(figsize=(8,10))
    sns.barplot( top_coefs.coefs_abs, top_coefs.index)
    plt.title('LASSO Regression: Top Features')
    plt.xlabel('Absolute Coefficient')
    plt.show()
    
def runmodels(training_data, test_data, numfeatures, idx):
    training_data = add_total_SF(training_data)
    test_data = add_total_SF(test_data)
    
    X_train, y_train=process_data_fm(training_data,idx[0:numfeatures])
    X_test, y_test=process_data_fm(test_data,idx[0:numfeatures])
    
    X_train_r2 = sm.add_constant(X_train)
    models = sm.OLS(y_train,X_train_r2)
    results = models.fit()
    print(results.summary())
    
    X_train_n = X_train
    X_test_n = X_test
    
    OLSrun(X_train_n, y_train, X_test_n, y_test)
    ridgerun(X_train_n, y_train, X_test_n, y_test)
    lassorun(X_train_n, y_train, X_test_n, y_test)
    
##### Running each model ########

full_data = pd.read_csv('https://raw.githubusercontent.com/benchang123/Ames-Housing/master/ames.csv')

#drop too many nan
nan=nanmean[nanmean>25].sort_values(ascending=False)
full_data.drop(columns=nan.index,inplace=True)

training_data, test_data = train_test_split(full_data, random_state=42, test_size=0.2)

training_data.shape

#RF
runmodels(training_data, test_data, numfeaturesrf, idxrf)

#GB
runmodels(training_data, test_data, numfeaturesgb, idxgb)

runmodels(training_data, test_data, 40, idxgb)

#corr
runmodels(training_data, test_data, numfeaturescorr, idxcorr)
