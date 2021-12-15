import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import json

def get_numpy_data(dataframe, features, label):
    dataframe.loc[:, 'intercept'] = 1
    features = ['intercept'] + features
    feature_matrix = dataframe.loc[:, features].values
    label_array = dataframe.loc[:, label].values
    return (feature_matrix, label_array)

def remove_punctuation(text):
    import string
    return text.translate(string.punctuation)

def get_product_reviews_data():
    products_df = pandas.read_csv('dataset/amazon_baby_subset.csv')

    with open('dataset/important_words.json', 'r') as f:
        important_words = json.loads(f.read())

    products_df = products_df.fillna({'review':''})  # fill in N/A's in the review column
    products_df.loc[:, 'review_clean'] = products_df['review'].apply(remove_punctuation)

    for word in important_words:
        products_df.loc[:, word] = products_df['review_clean'].apply(lambda s : s.split().count(word))

    sentiment_train_data = products_df.sample(frac=0.8, random_state=100)
    sentiment_validation_data = products_df.drop(sentiment_train_data.index)

    sentiment_X_train, sentiment_y_train = get_numpy_data(sentiment_train_data, important_words, 'sentiment')
    sentiment_X_valid, sentiment_y_valid = get_numpy_data(sentiment_validation_data, important_words, 'sentiment')

    print ('*****Sentiment data shape*****')
    print ('sentiment_X_train.shape: ', sentiment_X_train.shape)
    print ('sentiment_y_train.shape: ', sentiment_y_train.shape)
    print ('sentiment_X_valid.shape: ', sentiment_X_valid.shape)
    print ('sentiment_y_valid.shape: ', sentiment_y_valid.shape)

    return (sentiment_X_train, sentiment_y_train), (sentiment_X_valid, sentiment_y_valid)

def get_loans_data():
    loans_df = pandas.read_csv('dataset/lending-club-data.csv', low_memory=False)
    # safe_loans =  1 => safe
    # safe_loans = -1 => risky
    loans_df.loc[:, 'safe_loans'] = loans_df['bad_loans'].apply(lambda x : +1 if x==0 else -1)
    loans_df.drop(columns=['bad_loans'])
    target = 'safe_loans'
    features = ['grade',                     # grade of the loan (categorical)
                'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
                'short_emp',                 # one year or less of employment
                'emp_length_num',            # number of years of employment
                'home_ownership',            # home_ownership status: own, mortgage or rent
                'dti',                       # debt to income ratio
                'purpose',                   # the purpose of the loan
                'payment_inc_ratio',         # ratio of the monthly payment to income
                'delinq_2yrs',               # number of delinquincies
                'delinq_2yrs_zero',          # no delinquincies in last 2 years
                'inq_last_6mths',            # number of creditor inquiries in last 6 months
                'last_delinq_none',          # has borrower had a delinquincy
                'last_major_derog_none',     # has borrower had 90 day or worse rating
                'open_acc',                  # number of open credit accounts
                'pub_rec',                   # number of derogatory public records
                'pub_rec_zero',              # no derogatory public records
                'revol_util',                # percent of available credit being used
                'total_rec_late_fee',        # total late fees received to day
                'int_rate',                  # interest rate of the loan
                'total_rec_int',             # interest received to date
                'annual_inc',                # annual income of borrower
                'funded_amnt',               # amount committed to the loan
                'funded_amnt_inv',           # amount committed by investors for the loan
                'installment',               # monthly payment owed by the borrower
            ]
    # drop na
    loans_df = loans_df[features + [target]].dropna()

    safe_loans_raw = loans_df[loans_df[target] == +1]
    risky_loans_raw = loans_df[loans_df[target] == -1]

    # Since there are fewer risky loans than safe loans, find the ratio of the sizes
    # and use that percentage to undersample the safe loans.
    percentage = risky_loans_raw.shape[0]/safe_loans_raw.shape[0]

    risky_loans = risky_loans_raw
    safe_loans = safe_loans_raw.sample(frac=percentage, random_state=1)

    # Append the risky_loans with the downsampled version of safe_loans
    loans_data = risky_loans.append(safe_loans)

    categorical_variables = list(loans_data.select_dtypes(include=['object']).columns)

    one_hot_data = pandas.get_dummies(loans_data[categorical_variables], prefix=categorical_variables)
    # need to add inplace in oreder to drop columns.
    loans_data.drop(columns=categorical_variables, axis=1, inplace=True)
    loans_data = pandas.concat([loans_data, one_hot_data], axis=1)

    loans_train_data = loans_data.sample(frac=0.8, random_state=100)
    loans_validation_data = loans_data.drop(loans_train_data.index)

    loans_feature_colums = list(loans_train_data.columns)
    loans_feature_colums.remove(target)
    loans_X_train, loans_y_train = get_numpy_data(loans_train_data, loans_feature_colums, target)
    loans_X_valid, loans_y_valid = get_numpy_data(loans_validation_data, loans_feature_colums, target)

    print ('*****Loans data shape*****')
    print ('loans_X_train.shape: ', loans_X_train.shape)
    print ('loans_y_train.shape: ', loans_y_train.shape)
    print ('loans_X_valid.shape: ', loans_X_valid.shape)
    print ('loans_y_valid.shape: ', loans_y_valid.shape)

    return (loans_X_train, loans_y_train), (loans_X_valid, loans_y_valid)