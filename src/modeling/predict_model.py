from joblib import load
import pandas as pd


def main():
    test_cleaned = pd.read_parquet('../data/processed/test_processed.parquet')
    pipe = load('../models/final_stacking.joblib') 

    # Make predictions with test data predictors
    yhat = pipe.predict_proba(test_cleaned)[:, 1:]
    yhat = [item for sublist in yhat for item in sublist]

    submission = pd.DataFrame({
        'ID_code': pd.read_csv('../data/input/test.csv', usecols=['ID_code']).iloc[:, 0].to_list(),
        'target': yhat
    })

    submission.to_csv('../models/submissions/submission.csv', index=False)


if __name__ == '__main__':
    main()