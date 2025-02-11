import requests
import random
import time

class StockInventoryManager:
    def __init__(self, api_url):
        self.api_url = api_url

    def update_stock(self, product_id):
        stock_data = {"product_id": product_id, "stock_level": random.randint(10, 100)}
        response = requests.post(self.api_url, json=stock_data)
        return response.json()

if __name__ == "__main__":
    manager = StockInventoryManager("http://localhost:8000/update_stock")
    while True:
        product_id = random.randint(1, 100)
        stock_status = manager.update_stock(product_id)
        print("Updated Stock:", stock_status)
        time.sleep(5)
