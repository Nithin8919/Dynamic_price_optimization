

import pandas as pd
from sqlalchemy import create_engine

# Load your dataset
data = pd.read_csv('Data/dynamic_pricing.csv')

# Create a connection to PostgreSQL
engine = create_engine('postgresql://nitin:baddy@localhost:5432/Dynamic_price_optimization')

# Save to PostgreSQL (you can create the table name and schema)
data.to_sql('dynamic_pricing_data', engine, index=False, if_exists='replace')

print("Data saved to PostgreSQL successfully!")
