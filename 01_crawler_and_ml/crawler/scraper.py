class CarScraper:
    def __init__(self, base_url="https://gateway.chotot.com/v1/public/ad-listing"):
        self.base_url = base_url

    def fetch_page(self, page_index, limit=50):
        raise NotImplementedError

    def parse_listing(self, raw_data):
        raise NotImplementedError

    def run(self, max_records=15000):
        raise NotImplementedError
