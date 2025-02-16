from kafka import KafkaConsumer
import json

class KafkaSalesConsumer:
    def __init__(
        self,
        topic,
        bootstrap_servers="localhost:9092",
        group_id="sales_consumer_group",
        auto_offset_reset="earliest"
    ):
        """
        Initializes a Kafka consumer for reading JSON messages.
        
        :param topic: Name of the Kafka topic to consume.
        :param bootstrap_servers: Kafka bootstrap server addresses.
        :param group_id: Consumer group ID for coordinating with other consumers.
        :param auto_offset_reset: What to do if there is no initial offset in Kafka.
        """
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            value_deserializer=lambda x: json.loads(x.decode("utf-8"))
        )

    def consume_messages(self):
        print("Listening for incoming sales data...")
        for message in self.consumer:
            # message.value will be the deserialized JSON data
            print("Received:", message.value)

if __name__ == "__main__":
    consumer = KafkaSalesConsumer("sales_data")
    consumer.consume_messages()
