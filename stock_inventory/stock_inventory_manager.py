import requests
import random
import time

class StockInventoryManager:
    """
    Simulates and sends stock level updates to a specified API endpoint.

    :param api_url: The URL of the endpoint that handles stock updates.
    """
    def __init__(self, api_url):
        self.api_url = api_url

    def update_stock(self, product_id):
        """
        Sends a POST request with a randomly generated stock level for a given product_id.

        :param product_id: Integer ID of the product to update.
        :return: Parsed JSON response from the server, or None if there's an error.
        """
        stock_data = {
            "product_id": product_id,
            "stock_level": random.randint(10, 100)
        }

        try:
            response = requests.post(self.api_url, json=stock_data)
            response.raise_for_status()  # Raises an HTTPError if status != 200-299
            return response.json()
        except requests.RequestException as e:
            print(f"Error updating stock for product {product_id}: {e}")
            return None

if __name__ == "__main__":
    # Adjust URL to match your actual API endpoint
    manager = StockInventoryManager("http://localhost:8000/update_stock")

    print("Starting Stock Inventory Manager. Press Ctrl+C to stop.")
    while True:
        product_id = random.randint(1, 100)
        stock_status = manager.update_stock(product_id)

        if stock_status is not None:
            print(f"Updated Stock for product {product_id}:", stock_status)
        else:
            print(f"Failed to update stock for product {product_id}")

        # Update stock every 5 seconds
        time.sleep(5)
