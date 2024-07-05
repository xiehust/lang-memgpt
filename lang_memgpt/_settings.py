from pydantic_settings import BaseSettings
import dotenv
import os
assert dotenv.load_dotenv(".env", override=True) == True


class Settings(BaseSettings):
    pinecone_api_key: str = ""
    pinecone_index_name: str = ""
    pinecone_namespace: str = "ns1"
    model: str = "claude-3-5-sonnet-20240620"


SETTINGS = Settings(pinecone_api_key=os.environ["PINECONE_API_KEY"],
                    pinecone_index_name=os.environ["PINECONE_INDEX_NAME"],
                    pinecone_namespace=os.environ["PINECONE_NAMESPACE"])