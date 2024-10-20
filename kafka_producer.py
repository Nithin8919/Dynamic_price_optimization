from kafka import KafkaProducer
import json
import pandas as pd

import pandas as pd
from sqlalchemy import create_engine

# Connect to PostgreSQL
engine = create_engine('postgresql://nitin:baddy@localhost:5432/Dynamic_price_optimization')

# Read data in chunks
chunk_size = 1000  # Define the chunk size
for chunk in pd.read_sql('SELECT * FROM dynamic_pricing_data', engine, chunksize=chunk_size):
    print(chunk)
    # Send each chunk to Kafka for streaming...


# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send chunks to Kafka
for chunk in pd.read_sql('SELECT * FROM dynamic_pricing_data', engine, chunksize=chunk_size):
    chunk_dict = chunk.to_dict(orient='records')
    for row in chunk_dict:
        producer.send('dynamic_pricing_topic', value=row)
        producer.flush()

print("Data sent to Kafka topic.")
