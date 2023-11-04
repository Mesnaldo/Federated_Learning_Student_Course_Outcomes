from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    n_classes = 2  # MNIST has 10 classes
    n_features = 45  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


#def load_mnist() -> Dataset:
    
#    df = pd.read_csv('/Users/oumsai/Desktop/FL_FINAL 2/student_dataset.csv')
#    scaler = MinMaxScaler()
#    df[['Self_Study_Daily', 'Tution_Monthly']] = scaler.fit_transform(df[['Self_Study_Daily', 'Tution_Monthly']])
#    X = df.iloc[:, :-1]
#    y = df.iloc[:, -1]
#    X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.05, random_state=0)
#    return (X_train, y_train), (X_test, y_test)


# Load the dataset
url = '/Users/oumsai/Desktop/FL_FINAL 2/student_binary_classification.csv'
data = pd.read_csv(url, sep=",")

# Encode binary categorical variables (e.g., "school", "sex", "address")
binary_vars = ['school','sex','address','famsize','Pstatus','reason','traveltime','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
label_encoder = LabelEncoder()
for var in binary_vars:
    data[var] = label_encoder.fit_transform(data[var])

# Encode nominal categorical variables using one-hot encoding (e.g., "Mjob", "Fjob", "reason", "guardian")
data = pd.get_dummies(data, columns=["Mjob", "Fjob", "reason", "guardian"])

# Split data into features and target variable (X and y)
X = data.drop(columns=["G3"])  # Features
y = data["G3"]  # Target variable (final grade)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def load_mnist() -> Dataset:
     return (X_train, y_train), (X_test, y_test)



def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
