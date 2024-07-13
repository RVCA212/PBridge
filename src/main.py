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
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient
from pinecone_text.sparse import SpladeEncoder


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
            tokens = tokenizer.encode(
                text,
            )
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
        
        splade = SpladeEncoder()
        sparse_model_id = splade

        print("sparse model loaded")



        # Revised print statements to match the Document object structure

        # above loading is equivalent to following
        #    chunks = text_splitter.split_text(doc['page_content'])  # get page content from 'page_content' key
        #    for i, chunk in enumerate(chunks):
        #        documents.append({
        #            'id': f'{uid}-{i}',
        #            'text': chunk,
        #            'source': url
        #})

        print("Initializing pinecone")
        pc = PineconeClient(api_key=PINECONE_API_KEY)
        print("Pinecone initialized")
        print(pc)

        index_name = actor_input.get("index_name")
        namespace_name = actor_input.get("namespace_name")

        spec = ServerlessSpec(cloud='aws', region='us-east-1')

        # Check if our index already exists. If it doesn't, we create it
        if index_name not in pc.list_indexes().names():
            print("Creating index")
            # Create a new index
            pc.create_index(
                index_name,
                dimension=768,
                metric='dotproduct',
                spec=spec
            )
            # Wait for index to be initialized
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            print("Index created!")
        else:
            print("Index already exists, updating index.")

        index = pc.Index(index_name)

        def builder(records: list):
            # search is carried out via the smaller child content
            child_contents = [x["child_content"] for x in records]

            # create dense vecs
            dense_vecs = dense_model.encode(child_contents).tolist()

            # create sparse vecs
            input_ids = tokenizer(
                child_contents, return_tensors="pt",
                padding=True, truncation=True
            )
            with torch.no_grad():
                sparse_vecs = sparse_model(
                    d_kwargs=input_ids.to(device)
                )["d_rep"].squeeze()

            # convert to upsert format
            upserts = []

            for record, dense_vec, sparse_vec in zip(records, dense_vecs, sparse_vecs):
                _id = record["metadata"]["child_id"]
                source = record["metadata"]["source"]
                child_content = record["child_content"]
                parent_content = record["parent_content"]

                # extract columns where there are non-zero weights
                indices = sparse_vec.nonzero().squeeze().cpu().tolist() # positions
                values = sparse_vec[indices].cpu().tolist()             # weights/scores

                # append all to upserts list as pinecone.Vector (or GRPCVector)
                upserts.append({
                    "id": _id,
                    "values": dense_vec,
                    "sparse_values": {
                        "indices": indices,
                        "values": values
                    },
                    "metadata": {
                    "source": source,
                    "child_content": child_content,
                    "parent_content": parent_content
                    }
                })
            return upserts


        print("generating embeddings:", sparse_model_id)
        # Generate embeddings and prepare documents for upserting


        # there is a 2mb total batch upsert capacity -- keep batch_size < 100
        batch_size = 60

        # Updated to be async
        async def generate_and_upsert_documents(parent_child_documents, index, namespace_name):
            document_batches = [parent_child_documents[i:i + batch_size] for i in range(0, len(parent_child_documents), batch_size)]
            for batch in tqdm(document_batches):
                try:
                    index.upsert(vectors=builder(batch), namespace=namespace_name)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error during embedding or upserting: {e}")


        # Upsert documents
        await generate_and_upsert_documents(parent_child_documents, index, namespace_name)
        print("Documents successfully upserted to Pinecone index")

