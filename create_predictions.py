# Loading the relevant modules
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from string import ascii_letters
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.preprocessing import scale as scl
from sklearn.model_selection import GridSearchCV
from IPython.display import display, Math, Latex
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score  
import statsmodels.api as sm
from sklearn.preprocessing import scale as scl


# Assuming the new dataset has the same name 

# Read training set
df_train = pd.read_csv("Data/Regression_Supervised_Train.csv")

# Read testing set
df_test = pd.read_csv("Data/Regression_Supervised_Test.csv")


# In the training set keep only columns which are in test dataset 
df_train = df_train[df_test.columns.to_list() + ["parcelvalue"]]

# Removing not needed variables
df_train = df_train.drop(['totaltaxvalue', 'buildvalue', 'landvalue'], axis=1)
df_test = df_test.drop(['totaltaxvalue', 'buildvalue', 'landvalue'], axis=1)

# Create a pandas data frame with two columns a boolean column specifying whether the column has missing data and 
# another column specifying the percentage of missing values. 
mv_train = pd.DataFrame({'with_missing_values': df_train.isna().any(), 
              'percentage_missing_values':df_train.isnull().sum()*100/len(df_train)})


mv_test = pd.DataFrame({'with_missing_values': df_test.isna().any(), 
              'percentage_missing_values':df_test.isnull().sum()*100/len(df_test)})



# Remove columns with more than 90% of missing values
df_train_h = df_train[mv_train[mv_train["percentage_missing_values"] < 90].index.to_list()]

# Remove countycode because it gives the same information as countycode2 and numbath because is highly correlated with numbath2
df_train_h = df_train_h.drop(["countycode", "numfullbath"], axis = 1)

modes_df = df_train_h.groupby("neighborhoodcode")[["aircond", "qualitybuild", "heatingtype"]].agg(pd.Series.mode)

# Function to get values 
def get_value(value):
    if isinstance(value, np.floating) == False:
        if len(value)>1:
            return value[0]
        elif len(value)==0: 
            return None
    else:
        return value
        


modes_df = modes_df.applymap(get_value)


# Reset index to have neighborhood code as a variable
modes_df = modes_df.reset_index()

# Define column names before joining to the training set
modes_df.columns = ["neighborhoodcode", "aircond_mode", "qualitybuild_mode", "heatingtype_mode"]

# Left join with training set
df_train_h = df_train_h.merge(modes_df, on='neighborhoodcode', how='left')


# Fill variables using the mode
df_train_h["aircond"] = df_train_h["aircond"].fillna(df_train_h["aircond_mode"])
df_train_h["qualitybuild"] = df_train_h["qualitybuild"].fillna(df_train_h["qualitybuild_mode"])
df_train_h["heatingtype"] = df_train_h["heatingtype"].fillna(df_train_h["heatingtype_mode"])

# Dropping irrelevant variables
df_train_h = df_train_h.drop(["aircond_mode", "qualitybuild_mode", "heatingtype_mode"], axis = 1)

# Numerical columns to obtain the median
cols_to_me = ["num34bath",  "finishedarea1st", "finishedareaEntry", "numstories", "garagenum", "garagearea", "numbedroom", 
              "unitnum", "lotarea", "finishedarea"]

# Group numerical columns by neigborhoodcode and get the median
me_df = df_train_h.groupby("neighborhoodcode")[cols_to_me].median()

# Reset index to have neighborhoodcode as a variable
me_df = me_df.reset_index()

# Names of columns we are filling with the median
names_me_columns = []
for var in cols_to_me:
    name = var+"_me"
    names_me_columns.append(name)

# Set column names
me_df.columns = ["neighborhoodcode"] + names_me_columns

# Left join with training set
df_train_h = df_train_h.merge(me_df, on='neighborhoodcode', how='left')

# Filling some missing values using the median
for var in cols_to_me:
    df_train_h[var] = df_train_h[var].fillna(df_train_h[var+"_me"])

# Drop the variables we used to fill the missing values with the median
df_train_h = df_train_h.drop(names_me_columns, axis=1)

# Fill with 0s observations with missing values
df_train_h["poolnum"] = df_train_h["poolnum"].fillna(0)

mv_train_h = pd.DataFrame({'with_missing_values': df_train_h.isna().any(), 
              'percentage_missing_values':df_train_h.isnull().sum()*100/len(df_train_h)}).sort_values(by = "percentage_missing_values", ascending = False)


df_train_h = df_train_h[mv_train_h[mv_train_h["percentage_missing_values"] < 41].index.to_list()]
df_train_h = df_train_h.dropna()


# List with columns that have been removed from the testing set
a = list(set(df_train.columns)-set(df_train_h.columns))

# List with columns that we need to leave 
columns_to_keep = list(set(df_test.columns)-set(a))

# Keep only columns in columns_to_keep
df_test_h = df_test[columns_to_keep]


# Fill poolnum with 0s
df_test_h["poolnum"] = df_test_h["poolnum"].fillna(0)

# Obtain the modes by neighborhoodcode for categorical variables
modes_df_test = df_test_h.groupby("neighborhoodcode").agg(pd.Series.mode)[["aircond", "qualitybuild", "heatingtype"]].head(50)
modes_df_test = modes_df_test.applymap(get_value)
modes_df_test = modes_df_test.reset_index()
modes_df_test.columns = ["neighborhoodcode", "aircond_mode", "qualitybuild_mode", "heatingtype_mode"]

# Join with testing set
df_test_h = df_test_h.merge(modes_df_test, on='neighborhoodcode', how='left')

#Replace missing values with the modes by neighborhoodcode
df_test_h["aircond"] = df_test_h["aircond"].fillna(df_test_h["aircond_mode"])
df_test_h["qualitybuild"] = df_test_h["qualitybuild"].fillna(df_test_h["qualitybuild_mode"])
df_test_h["heatingtype"] = df_test_h["heatingtype"].fillna(df_test_h["heatingtype_mode"])

# Drop variables we created to fill the missing values with the mode
df_test_h = df_test_h.drop(["aircond_mode", "qualitybuild_mode", "heatingtype_mode"], axis = 1)

# Columns to obtain the median
cols_to_me = ["unitnum"]

me_df = df_test_h.groupby("neighborhoodcode")[cols_to_me].median()


# Reset index to have the neighborhoodcode
me_df = me_df.reset_index()

# Names of columns that we add the median
names_me_columns = []
for var in cols_to_me:
    name = var+"_me"
    names_me_columns.append(name)

# Set column names
me_df.columns = ["neighborhoodcode"] + names_me_columns

# Left join with testing set
df_test_h = df_test_h.merge(me_df, on='neighborhoodcode', how='left')

# Fill missing values with the median
df_test_h["unitnum"] = df_test_h["unitnum"].fillna(df_test_h["unitnum_me"])


# Drop irrelevant variables that were created to fill missing values with the median 
df_test_h = df_test_h.drop(names_me_columns, axis = 1) 

# If there are still missing values, fill them with the median 
df_test_h = df_test_h.fillna(df_test_h.median())


# For datasets with handled missing values
df_train_h = pd.get_dummies(df_train_h, prefix=['countycode2', 'citycode', 'regioncode', 'neighborhoodcode', 
                                                "taxyear", "heatingtype"], 
               columns=['countycode2','citycode', 'regioncode', 'neighborhoodcode', "taxyear", "heatingtype"], drop_first=True)

df_test_h = pd.get_dummies(df_test_h, prefix=['countycode2', 'citycode', 'regioncode', 'neighborhoodcode', 
                                              "taxyear", "heatingytpe"], 
               columns=['countycode2', 'citycode','regioncode', 'neighborhoodcode', "taxyear", "heatingtype"], drop_first=True)


# Defining X matrix with all regressors and parcelvalue as y as the dependent variable
X_train_h = df_train_h.drop(["lotid", "parcelvalue"], axis = 1)
y_train_h = df_train_h.parcelvalue

X_h = X_train_h.copy()
y_h = y_train_h.copy()

X_h = X_h.drop(["numfireplace", "lotarea", "roomnum"], axis = 1)

# Testing data
# X matrix for testing data
X_test_h = df_test_h.drop("lotid", axis = 1)

# Get missing columns in the testing test
missing_cols = set(X_h.columns) - set(X_test_h.columns)

# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    X_test_h[c] = 0
    
# Ensure the order of column in the test set is in the same order than in train set
X_test_original_h = X_test_h[X_h.columns]


X_test_h = X_test_original_h.copy()


X_h = scl(X_h)
X_test_h = scl(X_test_h)

# Training the model 
model_aic = LassoLarsIC(criterion="aic", normalize=False, max_iter=30)
results_final_model = model_aic.fit(X_h, np.log(y_h))

y_hat_log_h = model_aic.predict(X_h)


# Getting predictions
y_hat_test_log_h = model_aic.predict(X_test_h)
y_hat_test_h = np.exp(y_hat_test_log_h)



# Step 8: Produce .csv for kaggle testing 
test_predictions_submit = pd.DataFrame({"lotid": df_test_h["lotid"], "parcelvalue": y_hat_test_h})
test_predictions_submit.to_csv("test_predictions_submit.csv", index = False)