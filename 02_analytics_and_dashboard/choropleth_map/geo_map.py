class VietnamCarMap:
    def __init__(self, geojson_path):
        self.geojson_path = geojson_path

    def build_density_map(self, df):
        raise NotImplementedError

    def build_price_map(self, df):
        raise NotImplementedError
