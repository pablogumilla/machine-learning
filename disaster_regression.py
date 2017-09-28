import csv
import os

import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATASET_PATH = "{base_path}/datasets/titanic/train.csv".format(
    base_path=BASE_PATH
)
TEST_DATASET_PATH = "{base_path}/datasets/titanic/test.csv".format(
    base_path=BASE_PATH
)
DATASET_COLUMNS = [
    'Pclass',
    'Sex',
    'Age',
]


def _create_dataset(training=False):
    dataset_path = TEST_DATASET_PATH
    if training:
        dataset_path = TRAIN_DATASET_PATH

    # We read csv file to create dataset and
    # we clean data a little bit
    dataset = pandas.read_csv(dataset_path)
    dataset['Sex'] = dataset['Sex'].apply(
        lambda sex:1 if sex == 'male' else 0
    )
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())

    return dataset


def _create_train_dataset():
    train_dataset = _create_dataset(training=True)

    # We pick each passenger's data and
    # whether they survived or not
    data = train_dataset[DATASET_COLUMNS].values
    survived = train_dataset['Survived'].values

    return survived, data


def _create_test_dataset():
    test_dataset = _create_dataset()

    # We pick each passenger's data
    # This time we'll predict whether they survive or not
    data = test_dataset[DATASET_COLUMNS].values
    passenger_id = test_dataset["PassengerId"].values
    passenger_name = test_dataset["Name"].values

    return data, passenger_id, passenger_name


survived, training_data = _create_train_dataset()

# We create the regressor and train it
regressor = LogisticRegression()
regressor.fit(training_data, survived)

testing_data, passenger_id, passenger_name = _create_test_dataset()
predicted = regressor.predict(testing_data)

results = pandas.DataFrame(columns=['PassengerId', 'PassengerName', 'Survived'])
results['PassengerId'] = passenger_id
results['PassengerName'] = passenger_name
results['Survived'] = predicted.astype(bool)

print results
