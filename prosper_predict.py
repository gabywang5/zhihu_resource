import pandas as pd
import numpy as np
import datetime
import math
import datetime
from scipy import stats as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split

from sklearn.metrics import recall_score, precision_score, f1_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns

# Lets's see how the dataset looks like
df = original_data = pd.read_csv("./prosperLoanData.csv")
pd.set_option("display.max_columns", len(df.columns))
df.head()
df.info()

# fix the NaN 
# fill the NaN with Unknown
categorical = df.select_dtypes(include=['object']).columns.values
df[categorical] = df[categorical].fillna('Unknown')
df.select_dtypes(exclude=[np.number]).isnull().sum()
df.info()

# fill the BorrowerAPR
borrower_fees = df['BorrowerAPR'] - df['BorrowerRate']
borrower_fees.median()
df['BorrowerAPR'].fillna(df['BorrowerAPR'] + borrower_fees.median(), inplace=True)
df['BorrowerAPR'].isnull().sum()
df['BorrowerAPR'].fillna(df['BorrowerRate'] + borrower_fees.median(), inplace=True)
df['BorrowerAPR'].isnull().sum()
df.info()

# fill the EstimatedReturn
estimated_loss_from_fees = df['BorrowerRate'] - df['EstimatedEffectiveYield']
estimated_loss_from_fees.median()
df['EstimatedEffectiveYield'].fillna(df['BorrowerRate']- estimated_loss_from_fees.median(), inplace=True)
df['EstimatedEffectiveYield'].isnull().sum()
df['EstimatedLoss'].fillna(df['EstimatedLoss'].median(), inplace=True)
df['EstimatedLoss'].isnull().sum()
df['EstimatedReturn'].fillna(df['EstimatedEffectiveYield']-df['EstimatedLoss'], inplace=True)
df['EstimatedReturn'].isnull().sum()
df.info()


# fille the ProsperRating (numeric) & ProsperScore
df['ProsperRating (numeric)'].fillna(df['ProsperRating (numeric)'].median(), inplace=True)
df['ProsperScore'].fillna(df['ProsperScore'].median(), inplace=True)
df['ProsperRating (numeric)'].isnull().sum(), df['ProsperScore'].isnull().sum()
df.info()

# df.dropna(subset=['EmploymentStatusDuration','CreditScoreRangeLower','FirstRecordedCreditLine','CurrentCreditLines', 
#                   'TotalCreditLinespast7years'], inplace=True)
# df.info()


# fill the DebtToIncomeRatio
df_debt_income_null = df[df['DebtToIncomeRatio'].isnull()]
df_debt_income_null[:5]
df.loc[40]
df.loc[40, 'MonthlyLoanPayment'], df.loc[40, 'StatedMonthlyIncome']
df_debt_income_null = df[df['DebtToIncomeRatio'].isnull()]
df_debt_income_null['MonthlyLoanPayment'].isnull().sum(), df_debt_income_null['StatedMonthlyIncome'].isnull().sum()
df_debt_income_null['IncomeVerifiable'][:10]
df['DebtToIncomeRatio'].fillna(df['MonthlyLoanPayment'] / (df['StatedMonthlyIncome'] + 1), inplace=True)
df['DebtToIncomeRatio'].isnull().sum()

# drop the ScorexChangeAtTimeOfListing
df.drop('ScorexChangeAtTimeOfListing', axis=1, inplace=True)


# fill the missing Prosper histories with 0s.
prosper_vars = ['TotalProsperLoans','TotalProsperPaymentsBilled','OnTimeProsperPayments',
                'ProsperPaymentsLessThanOneMonthLate','ProsperPaymentsOneMonthPlusLate',
                'ProsperPrincipalBorrowed', 'ProsperPrincipalOutstanding']
df[prosper_vars] = df[prosper_vars].fillna(0)


df.drop(['ListingKey', 'ListingNumber', 'LoanKey', 'LoanNumber'], axis=1, inplace=True)
df.drop(["ListingCreationDate", "ClosedDate", "DateCreditPulled", "LoanOriginationDate",
         "LoanOriginationQuarter", "MemberKey"], axis=1, inplace=True)
df.drop(["LoanCurrentDaysDelinquent", "LoanFirstDefaultedCycleNumber", "LoanMonthsSinceOrigination", 
         "LP_CustomerPayments","LP_CustomerPrincipalPayments", "LP_InterestandFees",
         "LP_ServiceFees", "LP_CollectionFees", "LP_GrossPrincipalLoss","LP_NetPrincipalLoss", 
         "LP_NonPrincipalRecoverypayments"], axis=1, inplace=True)
df.info()

# define the default and completed
df['LoanStatus'].value_counts()
df_historical = df[df['LoanStatus'] != 'Current']
df_historical['LoanStatus'].value_counts()
df_historical['LoanStatus'] = (df_historical['LoanStatus'] == 'Completed').astype(int)
df_historical['LoanStatus'][:10]
fig = plt.figure()

get_ipython().magic(u'matplotlib inline')
sns.set_style("whitegrid")
sns.set_context({"figure.figsize":(15, 7.5)})

fig = plt.figure()
ax1 = fig.add_subplot(221)
sns.countplot(df_historical['LoanStatus'])
ax2 = fig.add_subplot(222)
sns.barplot(y=df_historical['LoanStatus']).set_ylim([0,1])
df_historical['LoanStatus'].mean(), 1-df_historical['LoanStatus'].mean()
df_historical.describe()

sns.barplot(x='BorrowerState', y='LoanStatus', data=df_historical)
al, dc = df_historical[df_historical['BorrowerState'] == 'AL'], df_historical[df_historical['BorrowerState'] == 'DC']
1- al['LoanStatus'].mean(), 1-dc['LoanStatus'].mean()

df_historical.replace(to_replace={"ListingCategory": {0: "Unknown", 1: "Debt", 2: "Reno", 3: "Business", 4: "Personal",
                                                                5: "Student", 6: "Auto", 7: "Other", 8: "Baby", 9: "Boat", 
                                                                10: "Cosmetic", 11: "Engagement", 12: "Green", 13: "Household",
                                                                14: "LargePurchase", 15: "Medical", 16: "Motorcycle", 17: "RV",
                                                                18: "Taxes", 19: "Vacation", 20: "Wedding"}}, inplace=True)
df_historical["ListingCategory"][:10]
sns.barplot(x='ListingCategory', y='LoanStatus', data=df_historical)
rv, green = df_historical[df_historical['ListingCategory'] == 'RV'], df_historical[df_historical['ListingCategory'] == 'Green']
1-rv['LoanStatus'].mean(), 1-green['LoanStatus'].mean()

fig = plt.figure()
ax1 = fig.add_subplot(221)
sns.barplot(x='ProsperRating (numeric)', y='LoanStatus', data=df_historical)
ax2 = fig.add_subplot(222)
sns.barplot(x='ProsperScore', y='LoanStatus', data=df_historical)
ax2 = fig.add_subplot(223)
sns.barplot(x='CreditScoreRangeLower', y='LoanStatus', data=df_historical)
ax2 = fig.add_subplot(223)

sns.barplot(x='CreditScoreRangeUpper', y='LoanStatus', data=df_historical)
credit_score_range = df_historical['CreditScoreRangeUpper'] - df_historical['CreditScoreRangeLower']
credit_score_range.value_counts()
df_historical.drop('CreditScoreRangeUpper', axis=1, inplace=True)
df_historical.rename(index=str,columns={'CreditScoreRangeLower':'CreditScore'}, inplace=True)
df_historical.info()

fig = plt.figure()
ax1 = fig.add_subplot(221)
sns.barplot(x='EmploymentStatus', y='LoanStatus', data=df_historical)
ax2 = fig.add_subplot(222)
sns.boxplot(x='LoanStatus', y='EmploymentStatusDuration', data=df_historical).set_ylim([0,400])
x = df_historical['EmploymentStatusDuration']
y = df_historical['LoanStatus']
r, p = st.pearsonr(x, y)
print('The correlation between employment status duration and loan default is {}, with a p-value of {}'.format(r, p))
df_historical.drop('EmploymentStatusDuration', axis=1, inplace=True)

fig = plt.figure()
ax1 = fig.add_subplot(221)
sns.barplot(x='LoanStatus', y='StatedMonthlyIncome', data=df_historical).set_ylim([0,20000])
ax2 = fig.add_subplot(222)
sns.boxplot(x='LoanStatus', y='DebtToIncomeRatio', data=df_historical).set_ylim([0,0.8])
fig = plt.figure()
ax1 = fig.add_subplot(221)
sns.barplot(x='LoanStatus', y='AvailableBankcardCredit', data=df_historical).set_ylim([-1000,50000])
ax2 = fig.add_subplot(222)
sns.boxplot(x='LoanStatus', y='BankcardUtilization', data=df_historical).set_ylim([0,2.5])
sns.barplot(x='IsBorrowerHomeowner', y='LoanStatus', data=df_historical).set_ylim([0,1])

fig = plt.figure()
ax1 = fig.add_subplot(221)
sns.barplot(x='LoanStatus', y='CurrentCreditLines', data=df_historical).set_ylim([0,40])
ax2 = fig.add_subplot(222)
sns.boxplot(x='LoanStatus', y='TotalCreditLinespast7years', data=df_historical).set_ylim([0,100])
ax3 = fig.add_subplot(223)
sns.barplot(x='LoanStatus', y='CurrentDelinquencies', data=df_historical).set_ylim([0,5])
ax4 = fig.add_subplot(224)
sns.boxplot(x='LoanStatus', y='DelinquenciesLast7Years', data=df_historical).set_ylim([0,20])
delinquency = ['CurrentDelinquencies','DelinquenciesLast7Years']
y = df_historical['LoanStatus']
for var in delinquency:
    x = df_historical[var]
    r, p = st.pearsonr(x, y)
    print("The correlation between {} and loan default is {}, with a p-value of {}".format(var, r, p))

fig = plt.figure()
ax1 = fig.add_subplot(221)
sns.violinplot(x='LoanStatus', y='LoanOriginalAmount', data=df_historical).set_ylim([0,35000])
ax2 = fig.add_subplot(222)
sns.violinplot(x='LoanStatus', y='Term', data=df_historical).set_ylim([10,60])
df_historical['Term'].value_counts()
sns.distplot(df_historical['BorrowerRate'])
df_historical['BorrowerRate'].describe()
sns.boxplot(x='CreditScore', y='BorrowerRate', hue='LoanStatus', data=df_historical)
df_historical.describe()

df_historical['PercentFunded'].mean()
df_historical['IncomeRange'].head()
df_historical['Occupation'].value_counts()
df_historical.drop(["CreditGrade", "BorrowerAPR", "LenderYield", "EstimatedEffectiveYield", 
                    "EstimatedLoss", "EstimatedReturn", "ProsperRating (Alpha)", "Occupation",
                    "CurrentlyInGroup", "GroupKey", "IncomeRange", "PercentFunded"],axis=1, inplace=True)
df_historical.info()

# transform the bool type into int type
df_historical['IsBorrowerHomeowner'] = df_historical['IsBorrowerHomeowner'].astype(int)
df_historical['IncomeVerifiable'] = df_historical['IncomeVerifiable'].astype(int)
df_historical['IsBorrowerHomeowner'][:10], df_historical['IncomeVerifiable'][:10]

# transform the datetime type
df_historical['FirstRecordedCreditLine'][:10]
first_credit_year = df_historical['FirstRecordedCreditLine'].str[:4]
df_historical['YearsWithCredit'] = 2014 - pd.to_numeric(first_credit_year)
df_historical.drop('FirstRecordedCreditLine', axis=1, inplace=True)
df_historical['YearsWithCredit'][:10]


category = pd.get_dummies(df_historical['ListingCategory'])
df_historical = df_historical.join(category, rsuffix='_category')
df_historical.drop('ListingCategory', axis=1, inplace=True)
df_historical.info()

employment = pd.get_dummies(df_historical['EmploymentStatus'])
df_historical = df_historical.join(employment, rsuffix='employmentstatus')
df_historical.drop('EmploymentStatus', axis=1, inplace=True)
df_historical.info()

# transform the BorrowerState
df_historical['BorrowerState'].value_counts()
state_defaults = df_historical.groupby('BorrowerRate')['LoanStatus'].mean()
vlow_risk = sorted(state_defaults)[51]
low_risk = sorted(state_defaults)[40]
mid_risk = sorted(state_defaults)[29]
high_risk = sorted(state_defaults)[19]
vhigh_risk = sorted(state_defaults)[9]
new_geography = {}
for state in state_defaults.index:
    if high_risk > state_defaults[state]:
        v = 'StateVeryHighRisk'
    elif mid_risk > state_defaults[state] >= high_risk:
        v = 'StateHighRisk'
    elif low_risk > state_defaults[state] >= mid_risk:
        v = 'StateMidRisk'
    elif vlow_risk > state_defaults[state] >= low_risk:
        v = 'StateLowRisk'
    else:
        v = 'StateVeryLowRisk'
    new_geography[state] = v
df_historical.replace(to_replace={'BorrowerState': new_geography}, inplace=True)

df_historical['BorrowerState'][:10]
state = pd.get_dummies(df_historical['BorrowerState'])
df_historical = df_historical.join(state, rsuffix='_state')
df_historical.drop('BorrowerState', axis=1, inplace=True)
df_historical.info()

# split the training set and testing set
X = df_historical.drop('LoanStatus', axis=1)
y = df_historical['LoanStatus']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# re-scale the data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
df_historical.info()

# reduce some features by PCA and univariate feature selection
pca = PCA(n_components=10)
pca.fit(X_train_scaled)
X_train_pca10 = pca.transform(X_train_scaled)
X_test_pca10 = pca.transform(X_test_scaled)
pca.explained_variance_ratio_

pca = PCA(n_components=3)
pca.fit(X_train)
X_train_pca3 = pca.transform(X_train_scaled)
X_test_pca3 = pca.transform(X_test_scaled)
pca.explained_variance_ratio_

X_train_reduce50 = SelectPercentile(percentile=50).fit_transform(X_train_scaled, y_train)
X_test_reduce50 = SelectPercentile(percentile=50).fit_transform(X_test_scaled, y_test)
X_train_reduce10 = SelectPercentile(percentile=10).fit_transform(X_train_scaled, y_train)
X_test_reduce10 = SelectPercentile(percentile=10).fit_transform(X_test_scaled, y_test)


# start the training
import warnings
models = {"Logistic Regression": LogisticRegression(),
         "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
         "Gaussian Naive Bayes": GaussianNB(),
         "Decision Tree Classifier": DecisionTreeClassifier(),
         "Random Forest Classifier": RandomForestClassifier(),
         "AdaBoost Classifier": AdaBoostClassifier()}
data = {"":X_train,
       "(scaled data)": X_train_scaled,
       "(PCA 10 components)": X_train_pca10,
       "(PCA 3 components)": X_train_pca3,
       "(50% of features)": X_train_reduce50,
       "(10% of features)": X_train_reduce10}
res = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, model in models.items():
        for scale, X in data.items():
            n = name + " " + scale
            clf = model
            s0, m0 = datetime.datetime.now().time().second, datetime.datetime.now().time().minute
            acc = cross_val_score(clf, X, y_train).mean()
            s1, m1 = datetime.datetime.now().time().second, datetime.datetime.now().time().minute
            y_pred = cross_val_predict(clf, X, y_train)
            default_recall = make_scorer(recall_score, pos_label=0, average="binary")
            default_precision = make_scorer(precision_score, pos_label=0, average="binary")
            default_fscore = make_scorer(f1_score, pos_label=0,average="binary")
            rec = cross_val_score(clf, X, y_train, scoring=default_recall).mean()
            prec = cross_val_score(clf, X, y_train, scoring=default_precision).mean()
            f1 = cross_val_score(clf, X, y_train, scoring=default_fscore).mean()
            m_delta = m1 - m0
            s_delta = s1 - s0
            if s_delta < 0:
                s_delta = 60 + s_delta
                m_delta -= 1
            res[n] = {"MeanAccuracy": acc,
                     "Precision": prec,
                     "Recall": rec,
                     "F1Score": f1,
                     "TrainingTime(m)": m_delta,
                     "TrainingTime(s)": s_delta}
results = pd.DataFrame.from_dict(res, orient="index")
results = results[["MeanAccuracy", "Precision", "Recall", "F1Score", "TrainingTime(m)", "TrainingTime(s)"]]
results


# find which model produces the best recall
results['Recall'].argmax(), results['Recall'].max()
# results['F1Score'].argmax(), results['F1Score'].max()

# tunning
# I choose the second best model : random forest classfier, instead of tunning the GaussianNB
default_recall = make_scorer(recall_score, pos_label=0)
param_grid = {"class_weight":
                              ["balanced",{0:2, 1:1},
                                          {0:5, 1:1},
                                          {0:10, 1:1}],
             "min_samples_split":[2, 3, 5, 10, 20]}
forest = RandomForestClassifier()
clf = GridSearchCV(forest, param_grid=param_grid, scoring=default_recall)
clf.fit(X_train_reduce50, y_train)
clf.best_params_, clf.best_score_
(.7371 - .4586)/ .4586

# I guess it may hurt the precision when the recall increases
default_precision = make_scorer(precision_score, pos_label=0, average="binary")
default_fscore = make_scorer(f1_score, pos_label=0, average="binary")
rec = cross_val_score(clf, X_train_reduce50, y_train, scoring=default_recall).mean()
prec = cross_val_score(clf, X_train_reduce50, y_train, scoring=default_precision).mean()
f = cross_val_score(clf, X_train_reduce50, y_train, scoring=default_fscore).mean()
print ("Recall: {} \n Precision: {} \n F1 Score: {}".format(rec, prec, f))

# predicting defaults
clf = RandomForestClassifier(class_weight={0: 10, 1: 1}, min_samples_split=20)
clf.fit(X_train_reduce50, y_train)
pred = clf.predict(X_test_reduce50)
pred = pd.Series(pred, index=[y_test.index], name='Predicted')
# I got errors at this step and didn't go on
