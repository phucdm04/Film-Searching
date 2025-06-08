# Film-Searching


## 📥 Data Retrieval Script

To keep our credentials secure, we use a `.env` file to store the MongoDB connection URI.
```env
MONGO_URI=[your_mongodb_uri]
```

Then use the below script to retrieve data.
```python
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the MongoDB URI from environment variable
uri = os.getenv("MONGO_URI")

client = MongoClient(
    uri,
    tls=True,
    tlsAllowInvalidCertificates=True
)

db = client["Film"]
collection = db["Data"]
cursor = collection.find({}, {"_id": 0})
df = pd.DataFrame(list(cursor))
```
