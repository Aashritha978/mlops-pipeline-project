from sklearn.datasets import load_iris

def load_data():
    data = load_iris()
    return data.data, data.target