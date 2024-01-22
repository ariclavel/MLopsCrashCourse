# web_service/lib/training_utils.py
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

def train_model(x_train: csr_matrix, y_train: np.ndarray):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    return lr
