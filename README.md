
# Access to Qdrant
```python
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Get the MongoDB URI from environment variable
url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_KEY")
# Access to VectorDB
qdrant_client  =  QdrantClient(
	url=url,
	api_key=api_key,
)
```