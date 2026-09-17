class CarPriceModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None

    def train_baseline(self, x_train, y_train):
        raise NotImplementedError

    def train_random_forest(self, x_train, y_train):
        raise NotImplementedError

    def train_xgboost(self, x_train, y_train):
        raise NotImplementedError

    def evaluate(self, x_test, y_test):
        raise NotImplementedError

    def save_model(self, filepath):
        raise NotImplementedError
