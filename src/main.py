import os
import tiktoken
import hashlib
import time
from apify import Actor
from tqdm.auto import tqdm
from pinecone import ServerlessSpec
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient
from pinecone_text.sparse import BM25Encoder

def get_nested_value(data_dict, keys_str):
    keys = keys_str.split('.')
    result = data_dict
    for key in keys:
        if key in result:
            result = result[key]
        else:
            return None
    return result

async def main():
    async with Actor:
        try:
            # Get the value of the actor input
            actor_input = await Actor.get_input() or {}
        except ValueError as e:
            print(f"Error getting actor input: {e}")
            actor_input = {}

        print("Actor input:", actor_input)

        os.environ['OPENAI_API_KEY'] = actor_input.get('openai_token')

        PINECONE_API_KEY = actor_input.get('pinecone_token')
        PINECONE_ENV = actor_input.get('pinecone_env')
        OPENAI_API_KEY = actor_input.get('openai_token')

        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is missing from the input")

        print("Loading dataset")

        # Get the request queue ID from the actor input
        request_queue_id = actor_input.get('resource', {}).get('defaultRequestQueueId')
        
        if not request_queue_id:
            raise ValueError("No request queue ID provided in the actor input")

        # Fetch the dataset using the request queue ID
        dataset = await Actor.open_dataset(request_queue_id)
        scraped_data = await dataset.get_items()

        documents = []
        for item in scraped_data:
            m = hashlib.sha256()
            m.update(item['url'].encode('utf-8'))
            uid = m.hexdigest()[:12]
            documents.append(Document(
                page_content=item['markdown'],
                metadata={
                    "source": item['url'],
                    "id": uid,
                    **item.get('metadata', {})
                }
            ))

        print(f"Loaded {len(documents)} documents")

        # Rest of the code remains the same...

        # Initialize Pinecone
        pc = PineconeClient(api_key=PINECONE_API_KEY)
        index_name = actor_input.get("index_name", "default-index")
        namespace_name = actor_input.get("namespace_name", "default-namespace")

        # Create index if it doesn't exist
        if index_name not in pc.list_indexes().names():
            print(f"Creating index: {index_name}")
            pc.create_index(
                index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric='dotproduct',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        else:
            print(f"Index {index_name} already exists")

        index = pc.Index(index_name)

        # Define builder function
        def builder(records: list):
            child_contents = [x["child_content"] for x in records]
            dense_vecs = dense_model.embed_documents(child_contents)
            sparse_vecs = bm25_encoder.encode_documents(child_contents)

            return [
                {
                    "id": record["metadata"]["child_id"],
                    "values": dense_vec,
                    "sparse_values": sparse_vec,
                    "metadata": {
                        **record["metadata"],
                        "child_content": record["child_content"],
                        "parent_content": record["parent_content"]
                    }
                }
                for record, dense_vec, sparse_vec in zip(records, dense_vecs, sparse_vecs)
            ]

        # Upsert documents
        batch_size = 60
        for i in tqdm(range(0, len(parent_child_documents), batch_size)):
            batch = parent_child_documents[i:i+batch_size]
            try:
                index.upsert(vectors=builder(batch), namespace=namespace_name)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error during upserting batch {i//batch_size}: {e}")

        print("Documents successfully upserted to Pinecone index")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
