# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 20:24:49 2022

All Project Code For Submission

@author: Joshua Phelan
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# These are the steps I took to prepare the 2015 Batting Dataframe, where I tested my initial models
# before training on a larger set. I explored various aspects of the data and explain my reasoning for the steps I took.


# Batter data taken from Baseball Reference
batterStandard = pd.read_csv("2015 MLB Batter Standard Stats.csv")
batterSalary = pd.read_csv("2015 MLB Batter Salaries.csv")
batterAdvanced = pd.read_csv("2015 MLB Batter Advanced Stats.csv")
batterCumulative = pd.read_csv("2015 MLB Batter Cumulative Stats.csv")

# Before merging all the dataframes together, I will reduce the columns
# Removing unnecessary columns from batterStandard
# I will keep Position Summary for now so I can remove pitchers later
batterStandard.drop(['Rk','Name', 'Tm','Lg'], axis=1, inplace=True)

# For batterSalary, there are unncessary columns and ones that are duplicates of batterStandard
batterSalary.columns.values
batterSalary.drop(['Rk', 'Name', 'Age', 'Tm', 'G', 'PA', 'Acquired','Pos\xa0Summary'], axis=1, inplace=True)

# batterAdvanced also has duplicate columns
batterAdvanced.drop(['Age', 'Tm', 'PA'], axis=1, inplace=True)

# I only need the column 'Yrs' from batterCumulative
batterCumulative = batterCumulative[['Yrs','Name-additional']]

# Because traded players show multiple rows per team, I will remove all traded players from consideration
(batterAdvanced['Name-additional'].duplicated() == True).sum()
batterAdvanced.drop_duplicates(subset='Name-additional',keep=False,inplace=True)
(batterAdvanced['Name-additional'].duplicated() == True).sum()

# Joining data on bbref key
batting = pd.merge(batterAdvanced, batterStandard, how = "inner", on= "Name-additional")
batting = pd.merge(batting, batterSalary, how = "inner", on= "Name-additional")
batting = pd.merge(batting, batterCumulative, how = "inner", on= "Name-additional")
(batting['Name-additional'].duplicated() == True).sum() # no duplicates in merged dataframe


# Checking null values for salary
batting['Salary'].isnull().sum()
batting.loc[batting['Salary'].isnull()]
batting.loc[batting['Salary'].isnull()]['WAR'].mean()
# The average WAR for null salary players is .20, meaning they are just above a replacement level player

# Dropping rows with any null values for salary
batting.dropna(subset=['Salary'],inplace=True)

# Checking null values for other columns
nullBatting = batting.isnull().sum()
# I will drop these columns with greater than 70 null values
batting.drop(['RS%','SB%','XBT%','GB/FB'],axis=1,inplace=True)
nullBatting = batting.isnull().sum()

# After investigating the rows where 'EV', the column with the most null values remaining,
# is null, it appears the players played no more than 80 games and had null values for
# many other columns as well. I will drop all these rows as they would not be valuable for training the model.
batting[batting.isnull()['EV']]
batting.dropna(subset=['EV'],inplace=True)

# There are 3 null values for BAbip, and the rest of the null values are within these 3 rows.
# I will drop them
nullBatting = batting.isnull().sum()
batting[batting.isnull()['BAbip']]
batting.dropna(subset=['BAbip'],inplace=True)

# There are no more null values
batting.isnull().sum()

# Converting salary from object to integer
batting.dtypes['Salary']
batting['Salary'] = batting['Salary'].apply(lambda x: int(str(x.replace(',','')[1:])))

# Converting percentage predictors to numeric values
nonNumeric = [col for col in batting.columns if (batting[col].dtypes != 'float64') and (batting[col].dtypes != 'int64')]
nonNumeric.remove('Name-additional')
nonNumeric.remove('Pos Summary')
for col in nonNumeric:
    batting[col] = batting[col].apply(lambda x: float(str(x)[:-1])/100)


# Looking at average salary by year

batting.loc[batting['Yrs'] > 6]['Salary'].mean()
batting.loc[batting['Yrs'] < 7]['Salary'].mean()


AvgSalary=pd.DataFrame()
salary = []
for i in range(1,20):
    print('Avg Salary for Year',i,': $',round(batting.loc[batting['Yrs'] == i]['Salary'].mean(),2))
    salary.append(round(batting.loc[batting['Yrs'] == i]['Salary'].mean(),2))
    
AvgSalary['Yrs'] = range(1,20)
AvgSalary['Salary'] = salary

# Two player milestones are evident in the plot below: arbitration after year 3 and free agency after year 6 
# Salary rapidly increases once a player reaches free agency after 6 years
g = sns.lineplot(data=AvgSalary,x='Yrs',y='Salary')
plt.title('Average Salary over Years Played')
plt.xlabel('Years Played')
plt.ticklabel_format(style='plain')
ylabels = ['$'+'{:,.1f}'.format(y) + 'M' for y in g.get_yticks()/1000000]
g.set_yticklabels(ylabels)

# Because I am only interested in free agent players, I will only look at players
# with more than 6 years of playing time
batting = batting.loc[batting['Yrs'] > 6]

# Selecting players with > 400 plate appearances as they would provide close to a full season's worth of data
batting = batting.loc[batting['PA'] > 400]

# Converting Position Summary into 1 position per player

# Taking the first position listed from the Pos Summary column and assigning that position to the player
# If a player is listed solely as a DH, and has no number for a position, they will be assigned '1' for position.
# Although position '1' signifies a pitcher, there are no pitchers in the dataframe due to the filter for PA > 400.
# 
import re

# Adding position dummy variables
pos = []
for i in range(len(batting)):
    try:
        pos.append(re.search('[0-9]', batting['Pos\xa0Summary'].iloc[i] ).group())
    except AttributeError:
        pos.append('1')
        continue

batting['Pos\xa0Summary'] = pos

# Getting dummy variables for the 9 positions
positions = pd.get_dummies(batting['Pos\xa0Summary'])
batting= pd.concat([batting,positions],axis=1)

# Dropping the 'pitcher' column
# The KeyError occurs when there are no values for '1' because they were no DH in the dataframe.
# In this case, the column does not need to be dropped.
try:
    batting.drop('1',axis=1,inplace=True)
except KeyError:
    pass

# Salary by Position boxplot
# This boxplot shows the higher median salary for 1B and RF, and lower median for C.
# I can assume these positions may have significant predictive value of salary.
g = sns.boxplot(x='Pos\xa0Summary',y='Salary',data=batting,order=sorted(batting['Pos\xa0Summary'].unique()))
plt.title('Salary by Position')
plt.xlabel('Position')
xlabels = ['C','1B','2B','3B','SS','LF','CF','RF']
g.set_xticklabels(xlabels)
ylabels = ['$'+'{:,.1f}'.format(y) + 'M' for y in g.get_yticks()/1000000]
g.set_yticklabels(ylabels)

# Dropping the Pos Summary Column for the regression model
batting.drop('Pos\xa0Summary',axis=1,inplace=True)

# A lot of baseball statistics are highly correlated with each other, in part because
# so many statistics are derived from others. For example, a player with more home runs will have more total bases.

# Function that returns a set of correlated columns from a dataframe above a correlated given threshold
def correlation(df, threshold):
    correlated_cols = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_cols.add(colname)
    return correlated_cols

# I used this function to remove a set of columns that have overlapping information and to begin
# narrowing down the model to a reasonable number of predictors.
# However, I ultimately found a more effective approach than selecting columns using the function
# to remove highly correlated columns and create a list of predictors, as I have shown below

# Dropping the highly correlated columns from the dataframe
batting_num = batting.drop(['Salary','Name-additional'],axis=1)
corr_features = correlation(batting_num, .5)
predictors = list(set(batting_num.columns) - corr_features)

# Adding dummy variable for 90th percentile of key stats
# This helps avoid issues of collinearity with key statistics that contain a lot of overlap
# while still being able to include them in the dataframe, unlike the above approach.

dummylist = []
def dummycreator(df,col,percentile):
    for row in range(len(df)):
        if df[col].iloc[row] > np.percentile(df[col],percentile):
            dummylist.append(1)
        else:
            dummylist.append(0)
    df[col+'_dummy'] = dummylist
    dummylist.clear()
                    
dummycreator(batting,'HR',90)          
dummycreator(batting,'RBI',90)
dummycreator(batting,'3B',90)   
dummycreator(batting,'TB',90)  
dummycreator(batting,'WAR',90)  
dummycreator(batting,'OPS',90) 
dummycreator(batting,'H',90)  
dummycreator(batting,'AB',90) 

# Saving the cleaned dataframe
batting.to_csv('Batting Dataframe 2015.csv')



# prepare function that combines all the saved csv files from Baseball Reference and combines and cleans them
# I used this function to prepare the batting dataframes for 2016-2022, excluding 2020
def prepare(year):
    year = f'{year}'
    # Batter data taken from Baseball Reference
    batterStandard = pd.read_csv(year+" MLB Batter Standard Stats.csv")
    batterSalary = pd.read_csv(year+" MLB Batter Salaries.csv")
    batterAdvanced = pd.read_csv(year+" MLB Batter Advanced Stats.csv")
    batterCumulative = pd.read_csv(year+" MLB Batter Cumulative Stats.csv")
    
    # Before merging all the dataframes together, I will reduce the columns
    # Removing unnecessary columns from batterStandard
    # I will keep Position Summary for now so I can remove pitchers later
    batterStandard.drop(['Rk','Name', 'Tm','Lg'], axis=1, inplace=True)
    
    # For batterSalary, there are unncessary columns and ones that are duplicates of batterStandard
    batterSalary.drop(['Rk', 'Name', 'Age', 'Tm', 'G', 'PA', 'Acquired','Pos\xa0Summary'], axis=1, inplace=True)
    
    # batterAdvanced also has duplicate columns
    batterAdvanced.drop(['Rk','Age','Name', 'Tm', 'PA'], axis=1, inplace=True)
    
    # Creating the career statistic variables
    batterCumulative.drop(['Name','Rk'], axis=1, inplace=True)
    batterCumulative = batterCumulative.add_suffix('_Career')
    batterCumulative.drop('Age_Career', axis=1, inplace=True)
    batterCumulative.rename(columns = {'Name-additional_Career':'Name-additional', 
                                       'Yrs_Career':'Yrs'}, inplace=True)
    
    # Because traded players show multiple rows per team, I will remove all traded players from consideration
    batterAdvanced.drop_duplicates(subset='Name-additional',keep=False,inplace=True)
    
    # Joining data on bbref key
    batting = pd.merge(batterAdvanced, batterStandard, how = "inner", on= "Name-additional")
    batting = pd.merge(batting, batterSalary, how = "inner", on= "Name-additional")
    batting = pd.merge(batting, batterCumulative, how = "inner", on= "Name-additional")
    (batting['Name-additional'].duplicated() == True).sum() # no duplicates in merged dataframe
    
    
    # Dropping rows with any null values for salary
    batting.dropna(subset=['Salary'],inplace=True)
    
    # With greater than 70 null values, I will drop these columns with greater than 70 null values
    batting.drop(['RS%','SB%','XBT%','GB/FB'],axis=1,inplace=True)
    
    # After investigating the rows where 'EV', the column with the most null values remaining,
    # is null, it appears the players played no more than 80 games and had null values for
    # many other columns as well. I will drop all these rows
    batting.dropna(subset=['EV'],inplace=True)
    
    # There are 3 null values for BAbip, and the rest of the null values are within these 3 rows.
    # I will drop them
    batting.dropna(subset=['BAbip'],inplace=True)
    
    # Converting salary from object to integer
    batting.dtypes['Salary']
    batting['Salary'] = batting['Salary'].apply(lambda x: int(str(x.replace(',','')[1:])))
    
    # Converting percentage predictors to numeric values
    nonNumeric = [col for col in batting.columns if (batting[col].dtypes != 'float64') and (batting[col].dtypes != 'int64')]
    nonNumeric.remove('Name-additional')
    nonNumeric.remove('Pos Summary')
    for col in nonNumeric:
        batting[col] = batting[col].apply(lambda x: float(str(x)[:-1])/100)
    
    # Because I am only interested in free agent players, I will only look at players
    # with more than 6 years of playing time
    batting = batting.loc[batting['Yrs'] > 6]
    
    # Selecting players with > 400 plate appearances
    batting = batting.loc[batting['PA'] > 400]
    
    import re
    
    # Adding position dummy variables
    pos = []
    for i in range(len(batting)):
        try:
            pos.append(re.search('[0-9]', batting['Pos\xa0Summary'].iloc[i] ).group())
        except AttributeError:
            pos.append('1')
            continue
    
    batting['Pos\xa0Summary'] = pos
    
    positions = pd.get_dummies(batting['Pos\xa0Summary'])
    batting= pd.concat([batting,positions],axis=1)
    
    # Dropping pitcher and position summary column
    try:
        batting.drop('1',axis=1,inplace=True)
    except KeyError:
        pass
    batting.drop('Pos\xa0Summary',axis=1,inplace=True)
    
    # Adding dummy variable for 90th percentile of key stats
    dummylist = []
    def dummycreator(df,col,percentile):
        for row in range(len(df)):
            if df[col].iloc[row] > np.percentile(df[col],percentile):
                dummylist.append(1)
            else:
                dummylist.append(0)
        df[col+'_dummy'] = dummylist
        dummylist.clear()
                        
    dummycreator(batting,'HR',90)          
    dummycreator(batting,'RBI',90)
    dummycreator(batting,'3B',90)   
    dummycreator(batting,'TB',90)  
    dummycreator(batting,'WAR',90)  
    dummycreator(batting,'OPS',90) 
    dummycreator(batting,'H',90)  
    dummycreator(batting,'AB',90) 
    
    batting['Year'] = int(year)
    
    return batting

# Now I can combine all the prepared dataframes into one.
# Below, I combine 2015-2019 data to train a model that can predict 2021 player salary.

from prepare import prepare

# Preparing all the datasets from all years
batting2015 = pd.read_csv('Batting Dataframe 2015.csv')
batting2015['Year'] = 2015
batting2015.drop('Unnamed: 0',axis=1,inplace=True)
batting2016 = prepare(2016)
batting2017 = prepare(2017)
batting2018 = prepare(2018)
batting2019 = prepare(2019)
batting2021 = prepare(2021)


# Combining all the datasets while avoiding duplicates. Here, I am starting with 2021 data and adding
# new rows descending from 20019 to have more data points with more recent salary numbers since
# salary inflation is much higher in baseball than the rest of the world.
# I am not including 2020 data because of statistical inconsistency from the shortened COVID-year season.
batting = pd.concat([batting2021,batting2019]).drop_duplicates(subset=['Name-additional'],keep='first').reset_index(drop=True)
batting = pd.concat([batting,batting2018]).drop_duplicates(subset=['Name-additional'],keep='first').reset_index(drop=True)
batting = pd.concat([batting,batting2017]).drop_duplicates(subset=['Name-additional'],keep='first').reset_index(drop=True)
batting = pd.concat([batting,batting2016]).drop_duplicates(subset=['Name-additional'],keep='first').reset_index(drop=True)
batting = pd.concat([batting,batting2015]).drop_duplicates(subset=['Name-additional'],keep='first').reset_index(drop=True)

# Saving complete dataframe
batting.to_csv('Batting Dataframe 2015-2021.csv')


# Preparing the next dataframe for 2023 Free Agent batters. This will be used so I 
# can evaluate the position players the Rays can target.

# I read the table of 2023 free agents from Spotrac. I only looked at unrestricted
# free agent batters.
freeagents = pd.read_html('https://www.spotrac.com/mlb/free-agents/ufa/batters/')[0]
freeagents = freeagents.droplevel(level=[0,1],axis=1)
freeagents.rename(columns = {'Player (2)':'Name'}, inplace=True)
# Removing unnecessary columns
freeagents.drop(['Bats','Throws', 'From','To','Yrs','Dollars','Average Salary'],axis=1,inplace=True)

# This dataframe includes many minor leaguers and many players who did not meet the 400 PA
# requirement for training the model, including D.J. Burt.
freeagents = freeagents[freeagents['Name'] != 'D.J. Burt']

# Adding Baseball Reference key using the pybaseball package
# The playerid_lookup function with Fuzzy=True returns the 5 most similar names
# Sometimes, the first player returned is not a current player and therefore would return
from pybaseball import playerid_lookup
lastname = []
firstname = []
bbref_list = []
i=0
for player in freeagents['Name']:
    lastname.append(freeagents.loc[(freeagents['Name'] == player)]['Name'].values[0].split()[1])
    firstname.append(freeagents.loc[(freeagents['Name'] == player)]['Name'].values[0].split()[0])
    
    if (playerid_lookup(lastname[i], firstname[i], fuzzy = True)['mlb_played_last'][0]>=2015) == True:
        bbref_list.append(playerid_lookup(lastname[i], firstname[i], fuzzy = True)['key_bbref'][0])
    elif (playerid_lookup(lastname[i], firstname[i], fuzzy = True)['mlb_played_last'][1]>=2015) == True:
        bbref_list.append(playerid_lookup(lastname[i], firstname[i], fuzzy = True)['key_bbref'][1])
    elif (playerid_lookup(lastname[i], firstname[i], fuzzy = True)['mlb_played_last'][2]>=2015) == True:
        bbref_list.append(playerid_lookup(lastname[i], firstname[i], fuzzy = True)['key_bbref'][2])
    else:
        bbref_list.append('error')
    i+=1

# Counting how many players were not returned with a bbref key
bbref_list.count('error')
freeagents['key_bbref'] = bbref_list

# Removing Yolmler Sanchez due to error
freeagents = freeagents[freeagents['Name'] != 'Yolmer Sanchez']

# Saving dataframe with bbref key
freeagents.to_csv('2023 Free Agents.csv')


# Importing cleaned free agent dataframe. Need to make sure the renames and columns are the same here
freeagents = pd.read_csv('2023 Free Agents.csv',index_col='Unnamed: 0')
freeagents.rename(columns = {'key_bbref':'Name-additional'}, inplace=True)
freeagents.drop(['Pos.','Age'],axis=1,inplace=True)

# Adding 2022 stats to freeagents
from prepare import prepare
batting2022 = prepare(2022)
freeagentbatting = pd.merge(freeagents, batting2022, how = "inner", on= "Name-additional")
freeagentbatting.drop(['Salary'],axis=1,inplace=True)

# Saving the final table for predictions, 2023 Free Agent Batting Stats
freeagentbatting.to_csv('2023 Free Agent Batting Stats.csv')



# Next, I visualize the top batting statistics in the batting dataframe.

# Histogram of Salary
sns.histplot(x='Salary',data=batting,bins=25)
plt.title("Salary")

# Histogram of Log(Salary)
logSalary = np.log(batting['Salary'])
sns.histplot(x=logSalary,bins=25)
plt.title("Log(Salary)")

# Histogram of top statistics
sns.histplot(x='HR',data=batting,bins=30)
plt.title("HR")

sns.histplot(x='RBI',data=batting,bins=30)
plt.title("RBI")

sns.histplot(x='TB',data=batting,bins=30)
plt.title("TB")

sns.histplot(x='IBB',data=batting,bins=30)
plt.title("IBB")

sns.histplot(x='R',data=batting,bins=30)
plt.title("R")

# Top statistics vs Salary
plt.scatter(batting['HR'], batting['Salary'])
plt.xlabel('HR')
plt.ylabel('Salary')

plt.scatter(batting['RBI'], batting['Salary'])
plt.xlabel('RBI')
plt.ylabel('Salary')

plt.scatter(batting['TB'], batting['Salary'])
plt.xlabel('TB')
plt.ylabel('Salary')

plt.scatter(batting['IBB'], batting['Salary'])
plt.xlabel('IBB')
plt.ylabel('Salary')

plt.scatter(batting['R'], batting['Salary'])
plt.xlabel('R')
plt.ylabel('Salary')


# Now I will implement various models to find which performs best at predicting salary

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import feature_selection
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Importing the dataset
batting=pd.read_csv('Batting Dataframe 2015-2021.csv',index_col='Unnamed: 0')
batting_num = batting.drop(['Salary','Name-additional'],axis=1)

# Creating a table to compare all model results
modelResults = pd.DataFrame(columns = ['Model','RMSE','Adj R^2','Accuracy'])

# Model 1: Stepwise Regression
from sklearn.linear_model import LinearRegression
# Using R, I performed forward stepwise regression. From the result, I removed insignificant variables
# until VIF < 5 for all . The found the variables below most significant
# The model in R had an adjusted R^2 of .67.
predictors = ["CS_Career", "Rpos","IBB_Career", "Rrep", "HBP_Career", "SO_Career", "ISO", "Age", 
                    "G", "Year", "OPS+_Career", "RBI_dummy", "TB_dummy", "LD%", "OPS_dummy", 
                    "IBB", "Rbaser","H_dummy"]

X = batting[predictors]
y = batting['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3,random_state=101)

lmStepwise = LinearRegression()

lmStepwise.fit(X_train,y_train)

pred = lmStepwise.predict(X_test)

# Predicted vs Actual Plot
plt.scatter(pred,y_test)
plt.title('Stepwise Linear Regression Model')
plt.xlabel('Predicted Y')
plt.ylabel('Y Test')

# Histogram of Residuals
sns.histplot((y_test-pred),bins=20)
plt.title('Actual - Predicted')

# Residuals vs Predicted Plot
plt.scatter(pred, (y_test-pred))
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Y')
plt.ylabel('Residuals')


# Results of model
results1 = pd.DataFrame()
results1['Predictor'] = predictors
results1['F-Score'] = feature_selection.f_regression(X, y, center=True, force_finite=True)[0]
results1['P Value'] = feature_selection.f_regression(X, y, center=True, force_finite=True)[1]
results1['Coefficient'] = lmStepwise.coef_
results1["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
print('MAE:', round(metrics.mean_absolute_error(y_test, pred),2))
print('MSE:', round(metrics.mean_squared_error(y_test, pred),2))
print('RMSE:', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),2))
r2 = metrics.r2_score(y_test, pred)
adj_r2 = 1-(((1-r2)*((len(y))-1))/((len(y))-(len(predictors))-1))
print('R^2:', round(r2,4))
print('Adj R^2:', round(adj_r2,4))
mape = metrics.mean_absolute_percentage_error(y_test, pred)
print('MAPE:', round(mape*100,2), '%')
accuracy = 1 - mape
print('Accuracy:', round(accuracy*100, 2), '%.')

results1.sort_values(by = 'P Value',axis=0)

# Adding results to modelResults dataframe
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, pred)),2)
modelResults.loc[len(modelResults)] = ['Stepwise',rmse, adj_r2, accuracy]



# Lasso regression - from R
# I removed insignificant variables until VIF < 5 for all.
# I found the variables below most significant
# The model in R had an adjusted R^2 of .67.
predictors = ["LD%", "Age", "G", "R", "IBB", "Rbaser", "Rpos", "Rrep", "CS_Career",
              "SO_Career", "OPS+_Career", "HBP_Career", "IBB_Career", "3", "5",     
              "HR_dummy", "RBI_dummy", "TB_dummy", "WAR_dummy", "OPS_dummy", "H_dummy", "Year" ]
X = batting[predictors]
y = batting['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3,random_state=101)

lmLasso = LinearRegression()

lmLasso.fit(X_train,y_train)

pred = lmLasso.predict(X_test)

# Predicted vs Actual Plot
plt.scatter(pred,y_test)
plt.title('Lasso Linear Regression Model')
plt.xlabel('Predicted Y')
plt.ylabel('Y Test')

# Histogram of Residuals
sns.histplot((y_test-pred),bins=20)
plt.title('Actual - Predicted')

# Residuals vs Predicted Plot
plt.scatter(pred, (y_test-pred))
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Y')
plt.ylabel('Residuals')


# Results of model
results2 = pd.DataFrame()
results2['Predictor'] = predictors
results2['F-Score'] = feature_selection.f_regression(X, y, center=True, force_finite=True)[0]
results2['P Value'] = feature_selection.f_regression(X, y, center=True, force_finite=True)[1]
results2['Coefficient'] = lmLasso.coef_
results2["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
print('MAE:', round(metrics.mean_absolute_error(y_test, pred),2))
print('MSE:', round(metrics.mean_squared_error(y_test, pred),2))
print('RMSE:', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),2))
r2 = metrics.r2_score(y_test, pred)
adj_r2 = 1-(((1-r2)*((len(y))-1))/((len(y))-(len(predictors))-1))
print('R^2:', round(r2,4))
print('Adj R^2:', round(adj_r2,4))
mape = metrics.mean_absolute_percentage_error(y_test, pred)
print('MAPE:', round(mape*100,2), '%')
accuracy = 1 - mape
print('Accuracy:', round(accuracy*100, 2), '%.')

results2.sort_values(by = 'P Value',axis=0)

# Adding results to modelResults dataframe
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, pred)),2)
modelResults.loc[len(modelResults)] = ['Lasso',rmse, adj_r2, accuracy]

# For the next model, I performed Principal Component Analysis

# Using Principal Component Analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 
from sklearn.model_selection import RepeatedKFold
from sklearn import model_selection

predictors=batting_num.columns
X = batting[predictors]
y = batting['Salary']

pca = PCA()
X_reduced = pca.fit_transform(scale(X))

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=101)

lmPCA = LinearRegression()
mse = []

# Calculate MSE with only the intercept
score = -1*model_selection.cross_val_score(lmPCA,
           np.ones((len(X_reduced),1)), y, cv=cv,
           scoring='neg_mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using cross-validation, adding one component at a time
for i in np.arange(1, 20):
    score = -1*model_selection.cross_val_score(lmPCA,
               X_reduced[:,:i], y, cv=cv, scoring='neg_mean_squared_error').mean()
    mse.append(score)

# Plot cross-validation results    
plt.figure(figsize=(12, 6))
plt.plot(mse,color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.xlabel('Number of Principal Components')
plt.ylabel('MSE')
plt.title('Salary')

np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

# Fitting results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3,random_state = 101)

# Scale the training and testing data using the first 4 components
X_reduced_train = pca.fit_transform(scale(X_train))
X_reduced_test = pca.transform(scale(X_test))[:,:3]

# Train PCR model on training data using the first 4 components
lmPCA = LinearRegression()
lmPCA.fit(X_reduced_train[:,:3], y_train)

pred = lmPCA.predict(X_reduced_test)

# Results

# Predicted vs Actual Plot
plt.scatter(pred,y_test)
plt.title('PCA Model')
plt.xlabel('Predicted Y')
plt.ylabel('Y Test')

# Histogram of Residuals
sns.histplot((y_test-pred),bins=15)
plt.title('Actual - Predicted')

# Residuals vs Predicted Plot
plt.scatter(pred, (y_test-pred))
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Y')
plt.ylabel('Residuals')

print('MAE:', round(metrics.mean_absolute_error(y_test, pred),2))
print('MSE:', round(metrics.mean_squared_error(y_test, pred),2))
print('RMSE:', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),2))
r2 = metrics.r2_score(y_test, pred)
adj_r2 = 1-(((1-r2)*((len(y))-1))/((len(y))-(4)-1))
print('R^2:', round(r2,4))
print('Adj R^2:', round(adj_r2,4))
mape = metrics.mean_absolute_percentage_error(y_test, pred)
print('MAPE:', round(mape*100,2), '%')
accuracy = 1 - mape
print('Accuracy:', round(accuracy*100, 2), '%.')

# Adding results to modelResults dataframe
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, pred)),2)
modelResults.loc[len(modelResults)] = ['PCA',rmse, adj_r2, accuracy]



# K Nearest Neighbors Regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


predictors =batting_num.columns
X = batting[predictors]
y = batting['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3,random_state=101)


# Finding right number of K
error = []

# Calculating MAE error for K values between 1 and 39
for i in range(1, 40):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, pred_i)
    error.append(mae)
    
# K-Value and MSE
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('K Value MAE')
plt.xlabel('K Value')
plt.ylabel('Mean Absolute Error')

# K-Value and MSE zoomed in
plt.figure(figsize=(12, 6))
plt.plot(range(1, 15), error[:14], color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('K Value MAE')
plt.xlabel('K Value')
plt.ylabel('Mean Absolute Error')

# K=8 had lowest MAE

knn = KNeighborsRegressor(n_neighbors=8)

knn.fit(X_train, y_train)

# Predict on test dataset 
from sklearn import metrics

pred = knn.predict(X_test)

# Results 

# Predicted vs Actual Plot
plt.scatter(pred,y_test)
plt.title('KNN Model')
plt.xlabel('Predicted Y')
plt.ylabel('Y Test')

# Histogram of Residuals
sns.histplot((y_test-pred),bins=15)
plt.title('Actual - Predicted')

# Residuals vs Predicted Plot
plt.scatter(pred, (y_test-pred))
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Y')
plt.ylabel('Residuals')

print('MAE:', round(metrics.mean_absolute_error(y_test, pred),2))
print('MSE:', round(metrics.mean_squared_error(y_test, pred),2))
print('RMSE:', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),2))
r2 = metrics.r2_score(y_test, pred)

adj_r2 = 1-(((1-r2)*((len(y))-1))/((len(y))-(len(predictors))-1))
print('R^2:', round(r2,4))
print('Adj R^2:', round(adj_r2,4))
mape = metrics.mean_absolute_percentage_error(y_test, pred)
print('MAPE:', round(mape*100,2), '%')
accuracy = 1 - mape
print('Accuracy:', round(accuracy*100, 2), '%.')

# Adding results to modelResults dataframe
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, pred)),2)
modelResults.loc[len(modelResults)] = ['KNN',rmse, adj_r2, accuracy]

# Combine PCA with KNN because KNN works better with fewer dimensions

predictors=batting_num.columns
X = batting[predictors]
y = batting['Salary']

pca = PCA()
X_reduced = pca.fit_transform(scale(X))

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=101)

knn_PCA = KNeighborsRegressor(n_neighbors=8)
mse = []

# Calculate MSE with only the intercept
score = -1*model_selection.cross_val_score(knn_PCA,
           np.ones((len(X_reduced),1)), y, cv=cv,
           scoring='neg_mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using cross-validation, adding one component at a time
for i in np.arange(1, 20):
    score = -1*model_selection.cross_val_score(knn_PCA,
               X_reduced[:,:i], y, cv=cv, scoring='neg_mean_squared_error').mean()
    mse.append(score)

# Plot cross-validation results    
plt.figure(figsize=(12, 6))
plt.plot(mse,color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.xlabel('Number of Principal Components')
plt.ylabel('MSE')
plt.title('Salary')

np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

# Fitting results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3,random_state = 101)

# Scale the training and testing data using the first 4 components
X_reduced_train = pca.fit_transform(scale(X_train))
X_reduced_test = pca.transform(scale(X_test))[:,:3]

# Calculating MAE error for K values between 1 and 39
error = []
for i in range(1, 40):
    knn_PCA = KNeighborsRegressor(n_neighbors=i)
    knn_PCA.fit(X_reduced_train[:,:3], y_train)
    pred_i = knn_PCA.predict(X_reduced_test)
    mae = metrics.mean_absolute_error(y_test, pred_i)
    error.append(mae)
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('K Value MAE')
plt.xlabel('K Value')
plt.ylabel('Mean Absolute Error')

# 9 clusters is optimal

# Train PCR model on training data using the first 4 components and 9 clusters
knn_PCA = KNeighborsRegressor(n_neighbors=9)
knn_PCA.fit(X_reduced_train[:,:3], y_train)

pred = knn_PCA.predict(X_reduced_test)

# Results

# Predicted vs Actual Plot
plt.scatter(pred,y_test)
plt.title('KNN Model with PCA')
plt.xlabel('Predicted Y')
plt.ylabel('Y Test')

# Histogram of Residuals
sns.histplot((y_test-pred),bins=15)
plt.title('Actual - Predicted')

# Residuals vs Predicted Plot
plt.scatter(pred, (y_test-pred))
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Y')
plt.ylabel('Residuals')

print('MAE:', round(metrics.mean_absolute_error(y_test, pred),2))
print('MSE:', round(metrics.mean_squared_error(y_test, pred),2))
print('RMSE:', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),2))
r2 = metrics.r2_score(y_test, pred)
adj_r2 = 1-(((1-r2)*((len(y))-1))/((len(y))-(4)-1))
print('R^2:', round(r2,4))
print('Adj R^2:', round(adj_r2,4))
mape = metrics.mean_absolute_percentage_error(y_test, pred)
print('MAPE:', round(mape*100,2), '%')
accuracy = 1 - mape
print('Accuracy:', round(accuracy*100, 2), '%.')

# Adding results to modelResults dataframe
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, pred)),2)
modelResults.loc[len(modelResults)] = ['KNN with PCA',rmse, adj_r2, accuracy]




# For the final model, I performed a random forest regression

batting_num = batting.drop(['Salary','Name-additional'],axis=1)
predictors =batting_num.columns
X = batting[predictors]
y = batting['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3,random_state=101)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 101)
# Train the model on training data
rf.fit(X_train, y_train)

# Use the forest's predict method on the test data
pred = rf.predict(X_test)

# Results

# Predicted vs Actual Plot
plt.scatter(pred,y_test)
plt.title('Random Forest Model')
plt.xlabel('Predicted Y')
plt.ylabel('Y Test')

# Histogram of Residuals
sns.histplot((y_test-pred),bins=15)
plt.title('Actual - Predicted')

# Residuals vs Predicted Plot
plt.scatter(pred, (y_test-pred))
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Y')
plt.ylabel('Residuals')

print('MAE:', round(metrics.mean_absolute_error(y_test, pred),2))
print('MSE:', round(metrics.mean_squared_error(y_test, pred),2))
print('RMSE:', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),2))
r2 = metrics.r2_score(y_test, pred)
print('R^2:', round(r2,4))
mape = metrics.mean_absolute_percentage_error(y_test, pred)
print('MAPE:', round(mape*100,2), '%')
accuracy = 1 - mape
print('Accuracy:', round(accuracy*100, 2), '%.')

# Adding results to modelResults dataframe
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, pred)),2)
modelResults.loc[len(modelResults)] = ['Random Forest',rmse, r2, accuracy]




# Get numerical feature importances
importances = list(rf.feature_importances_)# List of tuples with variable and importance
feature_importances = [(predictor, round(importance, 2)) for predictor, importance in zip(predictors, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# Saving the model results table
modelResults.to_csv('Salary Predictor Models Results.csv')


# Of all these models, I have found the forward linear regression model to have the lowest error
# I will use these models to predict 2023 free agents' salaries. Seeing predictions from various models at once will help
# me see where each model differs and if all models have a similar consensus salary for a player.


# Loading the 2023 Free Agent Batting Stats dataframe
freeagentbatting = pd.read_csv('2023 Free Agent Batting Stats.csv',index_col='Unnamed: 0')
freeagentbatting_num = freeagentbatting.drop(['Name-additional','Name'],axis=1)

# Table for predictions from all models
predictionsTable = pd.DataFrame()
predictionsTable['Name'] = freeagentbatting['Name']

# Stepwise Regression Model
predictors = ["CS_Career", "Rpos","IBB_Career", "Rrep", "HBP_Career", "SO_Career", "ISO", "Age", 
                    "G", "Year", "OPS+_Career", "RBI_dummy", "TB_dummy", "LD%", "OPS_dummy", 
                    "IBB", "Rbaser","H_dummy"]

predictionsTable['Stepwise'] = lmStepwise.predict(freeagentbatting[predictors])

# Lasso Regression Model
predictors = ["LD%", "Age", "G", "R", "IBB", "Rbaser", "Rpos", "Rrep", "CS_Career",
              "SO_Career", "OPS+_Career", "HBP_Career", "IBB_Career", "3", "5",     
              "HR_dummy", "RBI_dummy", "TB_dummy", "WAR_dummy", "OPS_dummy", "H_dummy", "Year" ]
predictionsTable['Lasso'] = lmLasso.predict(freeagentbatting[predictors])

# PCA Model
predictors = freeagentbatting_num.columns
pred_reduced = pca.transform(scale(freeagentbatting[predictors]))[:,:3]
predictionsTable['PCA'] = lmPCA.predict(pred_reduced)

# KNN Model
predictors = freeagentbatting_num.columns
predictionsTable['KNN'] = knn.predict(freeagentbatting[predictors])

# KNN with PCA
predictors = freeagentbatting_num.columns
pred_reduced = pca.transform(scale(freeagentbatting[predictors]))[:,:3]
predictionsTable['KNN with PCA'] = knn_PCA.predict(pred_reduced)

# Random Forest
predictors = freeagentbatting_num.columns
predictionsTable['Random Forest'] = rf.predict(freeagentbatting[predictors])


# Adding position to free agent table

# Taking baseball reference key and position data from the free agents table
predictionsTable['Name-additional'] = freeagentbatting['Name-additional']
positions = pd.read_csv('2023 Free Agents.csv')[['key_bbref','Pos.']]
positions.rename(columns = {'key_bbref':'Name-additional'},inplace=True)

# Merging positions onto predictionsTable
predictionsTable = pd.merge(predictionsTable, positions, how = "inner", on= "Name-additional")

# Moving Pos column to 2nd in dataframe
column_moving = predictionsTable['Pos.']
predictionsTable.insert(1, "Pos", column_moving )
predictionsTable.drop('Pos.',axis=1,inplace=True)

# Saving the salary predictions table
predictionsTable.to_csv('2023 Salary Predictions Table.csv')
