import mlflow, datetime, os, pickle, random
from joblib import dump
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sys
from sklearn.ensemble import RandomForestClassifier
import argparse

sys.path.insert(0, os.path.abspath('..'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    # Access the timestamp
    timestamp = args.timestamp

    # Use the timestamp in your script
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    # CHANGE 1: real Iris flower data with 4 known columns
    # (sepal length, sepal width, petal length, petal width, in cm)
    # instead of 6 unknown fake columns.
    iris = load_iris(as_frame=True)
    X = iris.data
    # Answer: 1 = the flower is Iris virginica, 0 = it is another species
    y = (iris.target == 2).astype(int)

    # CHANGE 2: keep 20% of the flowers aside as test data the model never sees
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save the TEST data, so evaluate_model.py can use it
    if os.path.exists('data'):
        with open('data/data.pickle', 'wb') as data:
            pickle.dump(test_X, data)

        with open('data/target.pickle', 'wb') as data:
            pickle.dump(test_y, data)
    else:
        os.makedirs('data/')
        with open('data/data.pickle', 'wb') as data:
            pickle.dump(test_X, data)

        with open('data/target.pickle', 'wb') as data:
            pickle.dump(test_y, data)

    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Iris Flowers"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    with mlflow.start_run(experiment_id=experiment_id,
                        run_name= f"{dataset_name}"):

        params = {
                    "dataset_name": dataset_name,
                    "number of datapoint": X.shape[0],
                    "number of dimensions": X.shape[1]}

        mlflow.log_params(params)

        forest = RandomForestClassifier(random_state=0)
        # CHANGE 3: learn from the training part only
        forest.fit(train_X, train_y)

        y_predict = forest.predict(train_X)
        mlflow.log_metrics({'Accuracy': accuracy_score(train_y, y_predict),
                            'F1 Score': f1_score(train_y, y_predict)})

        if not os.path.exists('models/'):
            # then create it.
            os.makedirs("models/")

        # After retraining the model
        model_version = f'model_{timestamp}'  # Use a timestamp as the version
        model_filename = f'{model_version}_dt_model.joblib'
        dump(forest, model_filename)
