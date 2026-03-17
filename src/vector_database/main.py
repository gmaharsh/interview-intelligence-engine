from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url= os.getenv("QDRANT_URL"),
    api_key= os.getenv("QDRANT_API_KEY")
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="test_collection",
    embedding_function=OpenAIEmbeddings(),
)