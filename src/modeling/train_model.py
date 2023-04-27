from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
from joblib import dump


def main():
    train_cleaned = pd.read_parquet('../data/processed/train_processed.parquet')

    # Split X and y from training data
    X_train = train_cleaned.drop(columns='target')  # 400 Predictors
    y_train = train_cleaned['target']  # 1 Binary Response

    # Initialize Classifier -> Stacking Classifier
    estimators = [
        ('lgbm', LGBMClassifier(num_leaves=30, n_estimators=200, max_depth=5, learning_rate=0.1)),
        ('xgb', XGBClassifier(subsample=1, n_estimators=200, max_depth=3, learning_rate=0.2, gamma=0.1, colsample_bytree=1))]
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)

    # Initialize ML Pipeline
    ml_pipeline = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', MinMaxScaler()),
        ('clf', stacking_clf)])

    # Train ML Pipeline with full training data
    ml_pipeline.fit(X_train, y_train)

    # Save the pickled object to disk.
    dump(ml_pipeline, '../models/final_stacking.joblib')


if __name__ ==  '__main__':
    main()
    