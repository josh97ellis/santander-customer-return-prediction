"""
Create duplicate counts features for each raw feature

Duplicates are considered from all of the 200k training features
in addition to the 100k real test data
"""
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def main():
    # Import raw training and test data
    df_train = pd.read_csv('C:/Users/Josh Ellis/projects/2023/santander-customer-transaction-prediction/data/input/train.csv')
    df_test = pd.read_csv('C:/Users/Josh Ellis/projects/2023/santander-customer-transaction-prediction/data/input/test.csv')

    raw_features = [c for c in df_train.columns if c not in ['ID_code', 'target']]
    test_values = df_test[raw_features].values

    # Creates a matrix of 0s in the shape of the test data
    unique_count = np.zeros_like(test_values)

    # Samples which have unique values are real the others are fake
    for feature in range(test_values.shape[1]):
        value_, index_, count_ = np.unique(test_values[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    # Identify the indexes of real and fake data in the test set
    real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

    # Create test frame with only real test data
    real_test = df_test.iloc[real_samples_indexes]

    # Combine the training and real testing data into one dataframe
    full_df = pd.concat([df_train, real_test])

    # Create a value counts feature for each raw feature (+200 new features)
    for feature in raw_features:
        val_counts = full_df[feature].value_counts().to_dict()
        df_train[f'{feature}_counts'] = df_train[feature].map(val_counts)
        df_test[f'{feature}_counts'] = df_test[feature].map(val_counts)

    df_test = df_test.drop(columns=['ID_code'])
    df_train = df_train.drop(columns=['ID_code'])

    # Write data to processed folder
    df_test.to_parquet('C:/Users/Josh Ellis/projects/2023/santander-customer-transaction-prediction/data/processed/test_processed.parquet')
    df_train.to_parquet('C:/Users/Josh Ellis/projects/2023/santander-customer-transaction-prediction/data/processed/train_processed.parquet')


if __name__ == '__main__':
    main()
