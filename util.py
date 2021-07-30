import pandas as pd
from sklearn.preprocessing import StandardScaler

creditcard_filepath="/home/ivanm/Documents/Github/Credit card fraud detection/creditcard.csv"

def read_data(creditcard_filepath):
    df = pd.read_csv('creditcard.csv')
    df.drop('Time', axis = 1, inplace = True)

    return df

if __name__=='__main__':
    df = read_data(creditcard_filepath)

    print(df.head())

    cases = len(df)
    nonfraud_count = len(df[df.Class == 0])
    fraud_count = len(df[df.Class == 1])
    fraud_percentage = round(fraud_count/nonfraud_count*100, 2)

    print('CASE COUNT')
    print('--------------------------------------------')
    print('Total number of cases are {}'.format(cases))
    print('Number of Non-fraud cases are {}'.format(nonfraud_count))
    print('Number of Non-fraud cases are {}'.format(fraud_count))
    print('Percentage of fraud cases is {}'.format(fraud_percentage))
    print('--------------------------------------------')

    nonfraud_cases = df[df.Class == 0]
    fraud_cases = df[df.Class == 1]

    print('CASE AMOUNT STATISTICS')
    print('--------------------------------------------')
    print('NON-FRAUD CASE AMOUNT STATS')
    print(nonfraud_cases.Amount.describe())
    print('--------------------------------------------')
    print('FRAUD CASE AMOUNT STATS')
    print(fraud_cases.Amount.describe())
    print('--------------------------------------------')
