from kafka import KafkaConsumer
import json

class KafkaSalesConsumer:
    def __init__(self, topic, bootstrap_servers="localhost:9092"):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode("utf-8"))
        )

    def consume_messages(self):
        print("Listening for incoming sales data...")
        for message in self.consumer:
            print("Received:", message.value)

if __name__ == "__main__":
    consumer = KafkaSalesConsumer("sales_data")
    consumer.consume_messages()
