import os
import tiktoken
import hashlib
import time
from functools import reduce
from apify import Actor
from tqdm.auto import tqdm
from uuid import uuid4
from getpass import getpass
from pinecone import ServerlessSpec
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
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
        actor_input = await Actor.get_input() or {}
        print(actor_input)

        os.environ['OPENAI_API_KEY'] = actor_input.get('openai_token')

        fields = actor_input.get('fields') or []
        metadata_fields = actor_input.get('metadata_fields') or {}
        metadata_values = actor_input.get('metadata_values') or {}

        PINECONE_API_KEY = actor_input.get('pinecone_token')
        PINECONE_ENV = actor_input.get('pinecone_env')
        OPENAI_API_KEY = actor_input.get('openai_token')

        print("Loading dataset")

        for field in metadata_fields:
            metadata_fields[field] = get_nested_value(actor_input.get('resource'), metadata_fields[field])
        
        def document_iterator(dataset_item):
            m = hashlib.sha256()
            m.update(dataset_item['url'].encode('utf-8'))
            uid = m.hexdigest()[:12]
            return Document(
                page_content=dataset_item['text'],
                metadata={"source": dataset_item['url'], "id": uid}
            )

        loader = ApifyDatasetLoader(
            dataset_id=actor_input.get('resource')['defaultDatasetId'],
            dataset_mapping_function=document_iterator
        )
                
        tiktoken_model_name = 'gpt-3.5-turbo'
        tiktoken.encoding_for_model(tiktoken_model_name)
        tokenizer = tiktoken.get_encoding('cl100k_base')

        def tiktoken_len(text):
            tokens = tokenizer.encode(text)
            return len(tokens)

        # Initialize BM25Encoder
        bm25_encoder = BM25Encoder().default()

        text_splitter1 = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=25,
            length_function=tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )

        bert_limit = 512
        text_splitter2 = RecursiveCharacterTextSplitter(
            chunk_size=bert_limit,
            chunk_overlap=25,
            length_function=tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )

        documents = loader.load()
        print("docs loaded")

        parent_child_documents = []
        for doc_id, doc in enumerate(documents):
            parent_chunks = text_splitter1.split_text(doc.page_content)
            for parent_id, parent_chunk in enumerate(parent_chunks):
                child_chunks = text_splitter2.split_text(parent_chunk)
                for child_id, child_chunk in enumerate(child_chunks):
                    parent_child_documents.append(
                        {
                            "parent_content": parent_chunk,
                            "child_content": child_chunk,
                            "metadata": {
                                "source": doc.metadata["source"],
                                "doc_id": doc_id,
                                "parent_id": f"{doc_id}-{parent_id}",
                                "child_id": f"{doc_id}-{parent_id}-{child_id}"
                            }
                        }
                    )

        print("documents split successfully!")

        encoder = OpenAIEmbeddings()
        dense_model = encoder

        print("dense model loaded")

        print("Initializing pinecone")
        pc = PineconeClient(api_key=PINECONE_API_KEY)
        print("Pinecone initialized")

        index_name = actor_input.get("index_name")
        namespace_name = actor_input.get("namespace_name")

        spec = ServerlessSpec(cloud='aws', region='us-west-2')

        if index_name not in pc.list_indexes().names():
            print("Creating index")
            pc.create_index(
                index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric='dotproduct',
                spec=spec
            )
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            print("Index created!")
        else:
            print("Index already exists, updating index.")

        index = pc.Index(index_name)

        def builder(records: list):
            child_contents = [x["child_content"] for x in records]

            # Create dense vectors
            dense_vecs = dense_model.embed_documents(child_contents)

            # Create sparse vectors using BM25
            sparse_vecs = bm25_encoder.encode_documents(child_contents)

            upserts = []

            for record, dense_vec, sparse_vec in zip(records, dense_vecs, sparse_vecs):
                _id = record["metadata"]["child_id"]
                source = record["metadata"]["source"]
                child_content = record["child_content"]
                parent_content = record["parent_content"]

                upserts.append({
                    "id": _id,
                    "values": dense_vec,
                    "sparse_values": sparse_vec,
                    "metadata": {
                        "source": source,
                        "child_content": child_content,
                        "parent_content": parent_content
                    }
                })
            return upserts

        print("generating embeddings")

        batch_size = 60

        def generate_and_upsert_documents(parent_child_documents, index, namespace_name):
            document_batches = [parent_child_documents[i:i + batch_size] for i in range(0, len(parent_child_documents), batch_size)]
            for batch in tqdm(document_batches):
                try:
                    index.upsert(vectors=builder(batch), namespace=namespace_name)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error during embedding or upserting: {e}")

        await generate_and_upsert_documents(parent_child_documents, index, namespace_name)
        print("Documents successfully upserted to Pinecone index")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())