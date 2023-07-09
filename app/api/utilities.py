import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open("app/api/models/churn_model.pkl", "rb"))

TARGET = "Exited"

numerical = ['CreditScore',
             'Age',
             'Tenure',
             'Balance',
             'NumOfProducts',
             'EstimatedSalary']
categorical =['RowNumber',
              'CustomerId',
              'Surname',
              'Geography',
              'Gender',
              'HasCrCard',
              'IsActiveMember',
              'Exited']

#Feature Engineering
def credit_score_table(row):
    credit_score = row.CreditScore
    if credit_score < 300:
        return "Deep"
    elif credit_score < 500:
        return "Very_Poor"
    elif credit_score < 601:
        return "Poor"
    elif credit_score < 661:
        return "Fair"
    elif credit_score < 781:
        return "Good"
    elif credit_score < 851:
        return "Excellent"
    else:
        return "Top"
def age_categorization(age):
    if age < 0:
        return "Invalid"
    elif age < 13:
        return "Child"
    elif age < 18:
        return "Teenager"
    elif age < 30:
        return "Young Adult"
    elif age < 60:
        return "Adult"
    else:
        return "Senior"

def age_tenure(row):
    age = row.Age
    tenure = row.Tenure
    return age*tenure

def product_utilization_rate_by_year(row):
    number_of_products = row.NumOfProducts
    tenure = row.Tenure
    
    if tenure == 0:
        return number_of_products
    else:
        rate = number_of_products / tenure
        return rate
    
def product_utilization_rate_by_estimated_salary(row):
    number_of_products = row.NumOfProducts
    estimated_salary = row.EstimatedSalary
    
    if estimated_salary == 0:
        return "Undefined"
    else:
        rate = number_of_products / estimated_salary
        return rate

def countries_monthly_average_salaries(row):
    # Monthly average salary taken from https://www.worlddata.info/average-income.php
    avg_salaries = {'France': 3680, 'Germany': 4305, 'Spain': 2474}
    
    salary = row.EstimatedSalary / 12
    country = row.Geography
    
    if country in avg_salaries:
        return salary / avg_salaries[country]
    else:
        return None
    
def balance_salary(row):
    balance = row.Balance
    salary = row.EstimatedSalary
    return balance/salary

def credit_score_squared(row):
    return (row.CreditScore)**2

def mean_financials(row):
    credit_score = row.CreditScore
    balance = row.Balance
    estimated_salary = row.EstimatedSalary
    
    mean_financials = (credit_score + balance + estimated_salary) / 3
    return mean_financials

def feature_engineering(df, is_show_graph=False):
    df_fe = df.copy()

    # balance_salary_rate
    df_fe['balance_salary_rate'] = df_fe.apply(lambda x: balance_salary(x), axis=1)

    # product_utilization_rate_by_year
    df_fe = df_fe.assign(product_utilization_rate_by_year=df_fe.apply(lambda x: product_utilization_rate_by_year(x), axis=1))

    # product_utilization_rate_by_estimated_salary
    df_fe = df_fe.assign(product_utilization_rate_by_estimated_salary=df_fe.apply(lambda x: product_utilization_rate_by_estimated_salary(x), axis=1))

    # tenure_rate_by_age
    df_fe['tenure_rate_by_age'] = df_fe.Tenure / (df_fe.Age - 17)

    # credit_score_rate_by_age
    df_fe['credit_score_rate_by_age'] = df_fe.CreditScore / (df_fe.Age - 17)

    # product_utilization_rate_by_salary
    df_fe['product_utilization_rate_by_salary'] = df_fe.Tenure / (df_fe.EstimatedSalary)

    # credit_score_rate_by_salary
    df_fe['credit_score_rate_by_salary'] = df_fe.CreditScore / (df_fe.EstimatedSalary)

    # mean_financials
    df_fe['mean_financials'] = df_fe.apply(lambda x: mean_financials(x), axis=1)

    # credit_score_squared
    df_fe['credit_score_squared'] = df_fe.apply(lambda x: credit_score_squared(x), axis=1)

    # age_tenure
    df_fe['age_tenure'] = df_fe.apply(lambda x: age_tenure(x), axis=1)

    # age_categorization
    df_fe['age_categorization'] = df_fe.Age.apply(lambda x: age_categorization(x))

    # credit_score_table
    df_fe = df_fe.assign(credit_score_table=df_fe.apply(lambda x: credit_score_table(x), axis=1))

    # countries_monthly_average_salaries
    df_fe = df_fe.assign(countries_monthly_average_salaries=df_fe.apply(lambda x: countries_monthly_average_salaries(x), axis=1))

    return df_fe

def data_encoding(df):
    df_model = df.copy()
    
    # Categorical columns
    non_encoding_columns = ["Geography", "HasCrCard", "IsActiveMember", "Gender", "NumOfProducts", "Tenure", "credit_score_table","age_categorization"]
    df_non_encoding = df_model[non_encoding_columns]
    df_model = df_model.drop(non_encoding_columns, axis=1)
    
    df_encoding = df_non_encoding.copy()
    
    encoder = LabelEncoder()
    df_encoding["gender_category"] = encoder.fit_transform(df_non_encoding.Gender)
    df_encoding["country_category"] = encoder.fit_transform(df_non_encoding.Geography)
    df_encoding["credit_score_category"] = encoder.fit_transform(df_non_encoding.credit_score_table)
    df_encoding["age_category"] = encoder.fit_transform(df_non_encoding.age_categorization)

    df_encoding.reset_index(drop=True, inplace=True)
    df_model.reset_index(drop=True, inplace=True)
    df_model = pd.concat([df_model, df_encoding], axis=1)

    # Drop unnecessary columns
    df_model = df_model.drop(["Geography", "Gender", "CustomerId", "Surname", "credit_score_table","age_categorization"], axis=1)

    # Encode HasCrCard and IsActiveMember as -1 for 0 values
    df_model.loc[df_model.HasCrCard == 0, 'HasCrCard'] = -1
    df_model.loc[df_model.IsActiveMember == 0, 'IsActiveMember'] = -1

    return df_model

def predictions(model, df):
    df_feature = feature_engineering(df)
    df_encoded = data_encoding(df_feature)
    data = model.predict(df_encoded)
    return data

def prediction_pipeline(df):
    return predictions(model,df)

if __name__ == "__main__":
    data = {'RowNumber': 5,
            'CustomerId': 15737888,
            'Surname': 'Mitchell',
            'CreditScore': 850,
            'Geography': 'Spain',
            'Gender': 'Female',
            'Age': 43,
            'Tenure': 2,
            'Balance': 125510.82,
            'NumOfProducts': 1,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 79084.1}
    data_new = pd.DataFrame(data, index = [0])
    result = predictions(model,data_new)
    print(result)
    