import os
import tiktoken
import hashlib
import time
from apify import Actor
from tqdm.auto import tqdm
from pinecone import ServerlessSpec
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ApifyDatasetLoader
from pinecone import Pinecone as PineconeClient
from pinecone_text.sparse import BM25Encoder

def get_nested_value(data_dict, keys_str):
    keys = keys_str.split('.')
    result = data_dict

    for key in keys:
        if key in result:
            result = result[key]
        else:
            # If any of the keys are not found, return None
            return None

    return result

async def main():
    async with Actor:
        try:
            # Get the value of the actor input
            actor_input = await Actor.get_input() or {}
            print("Actor Input:", actor_input)

            os.environ['OPENAI_API_KEY'] = actor_input.get('openai_token')

            fields = actor_input.get('fields') or []
            metadata_fields = actor_input.get('metadata_fields') or {}
            metadata_values = actor_input.get('metadata_values') or {}

            PINECONE_API_KEY = actor_input.get('pinecone_token')
            PINECONE_ENV = actor_input.get('pinecone_env')
            OPENAI_API_KEY = actor_input.get('openai_token')

            print("Loading dataset")

            # Iterator over metadata fields
            for field in metadata_fields:
                metadata_fields[field] = get_nested_value(actor_input.get('resource'), metadata_fields[field])

            # If you want to process data from Apify dataset before sending it to pinecone, do it here inside iterator function
            def document_iterator(dataset_item):
                m = hashlib.sha256()
                m.update(dataset_item['url'].encode('utf-8'))
                uid = m.hexdigest()[:12]
                return Document(
                    page_content=dataset_item['markdown'],
                    metadata={"source": dataset_item['url'], "id": uid}
                )

            loader = ApifyDatasetLoader(
                dataset_id=actor_input.get('resource')['defaultDatasetId'],
                dataset_mapping_function=document_iterator
            )

            # Cleaning data before intializing pinecone
            tiktoken_model_name = 'gpt-3.5-turbo'
            tiktoken.encoding_for_model(tiktoken_model_name)
            tokenizer = tiktoken.get_encoding('cl100k_base')

            # create the length function
            def tiktoken_len(text):
                tokens = tokenizer.encode(text)
                return len(tokens)

            # Create text splitter based on length function
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

            # load documents from Apify
            documents = loader.load()
            print("docs loaded")

            # Collect all markdown content for BM25 fitting
            all_markdown_content = [doc.page_content for doc in documents]

            # Fit BM25 model on the collected markdown content
            bm25_encoder = BM25Encoder()
            bm25_encoder.fit(all_markdown_content)
            bm25_encoder.dump("bm25_values.json")
            bm25_encoder = BM25Encoder().load("bm25_values.json")

            # Split documents into chunks
            parent_child_documents = []
            for doc_id, doc in enumerate(documents):
                print(doc)
                print(type(doc))
                # First, split document into parent chunks
                parent_chunks = text_splitter1.split_text(doc.page_content)
                for parent_id, parent_chunk in enumerate(parent_chunks):
                    # For each parent chunk, split further into child chunks
                    child_chunks = text_splitter2.split_text(parent_chunk)
                    for child_id, child_chunk in enumerate(child_chunks):
                        # Append parent chunk (larger) and child chunk (smaller) together with metadata
                        parent_child_documents.append(
                            {
                                "parent_content": parent_chunk,
                                "context": child_chunk,
                                "metadata": {
                                    "source": doc.metadata["source"],
                                    "doc_id": doc_id,
                                    "parent_id": f"{doc_id}-{parent_id}",
                                    "child_id": f"{doc_id}-{parent_id}-{child_id}"
                                }
                            }
                        )

                print("documents split successfully!")

            # Initialize Pinecone
            pc = PineconeClient(api_key=PINECONE_API_KEY)
            index_name = actor_input.get("index_name", "default-index")
            namespace_name = actor_input.get("namespace_name", "default-namespace")

            # Create index if it doesn't exist
            if index_name not in pc.list_indexes().names():
                print(f"Creating index: {index_name}")
                pc.create_index(
                    index_name,
                    dimension=768,
                    metric='dotproduct',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                while not pc.describe_index(index_name).status['ready']:
                    time.sleep(1)
            else:
                print(f"Index {index_name} already exists")

            index = pc.Index(index_name)

            dense_model = OpenAIEmbeddings(model='text-embedding-3-small', dimensions='768')

            # Define builder function
            def builder(records: list):
                child_contents = [x["context"] for x in records]
                dense_vecs = dense_model.embed_documents(child_contents)
                sparse_vecs = bm25_encoder.encode_documents(child_contents)

                return [
                    {
                        "id": record["metadata"]["child_id"],
                        "values": dense_vec,
                        "sparse_values": sparse_vec,
                        "metadata": {
                            **record["metadata"],
                            "source": doc.metadata["source"],
                            "context": record["context"],
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
        
        except Exception as e:
            print(f"Error in actor execution: {e}")
            raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
