class CarDataCleaner:
    def __init__(self, input_path):
        self.input_path = input_path

    def standardize_price(self, df):
        raise NotImplementedError

    def standardize_odometer(self, df):
        raise NotImplementedError

    def normalize_provinces(self, df):
        raise NotImplementedError

    def remove_outliers_iqr(self, df, columns):
        raise NotImplementedError

    def run_pipeline(self):
        raise NotImplementedError
