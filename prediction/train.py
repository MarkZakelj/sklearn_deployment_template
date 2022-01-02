from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def main():
    dataset = load_iris()
    X, y = dataset.data, dataset.target
    model = RandomForestClassifier(n_estimators=150, max_depth=3)
    model.fit(X, y)
    dump(model, 'rf_model.joblib')


if __name__=='__main__':
    main()