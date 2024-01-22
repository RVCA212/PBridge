import os
import pinecone
import tiktoken
import hashlib
from pinecone import Pinecone, ServerlessSpec
from functools import reduce
from apify import Actor
from tqdm.auto import tqdm
from uuid import uuid4
from getpass import getpass
from langchain.document_loaders import ApifyDatasetLoader
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

        print(actor_input)

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
                page_content=dataset_item['text'],
                metadata={"source":dataset_item['url'],"id": uid}
            )

        loader = ApifyDatasetLoader(
            dataset_id=actor_input.get('resource')['defaultDatasetId'],
            dataset_mapping_function=document_iterator
        )
        print("Dataset loaded ")

        # Cleaning data before intializing pinecone
        tiktoken_model_name = 'gpt-3.5-turbo'
        tiktoken.encoding_for_model(tiktoken_model_name)
        tokenizer = tiktoken.get_encoding('cl100k_base')

        # create the length function
        def tiktoken_len(text):
            tokens = tokenizer.encode(
                text,
                disallowed_special=()
            )
            return len(tokens)

        # Create text splitter based on length function
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )


        print("Loading documents ")

        # load documents based on text splitter
        documents = loader.load_and_split(text_splitter)
        print("Documents loaded:", len(documents))

        # above loading is equivalent to following
        #    chunks = text_splitter.split_text(doc['page_content'])  # get page content from 'page_content' key
        #    for i, chunk in enumerate(chunks):
        #        documents.append({
        #            'id': f'{uid}-{i}',
        #            'text': chunk,
        #            'source': url
        #})


        # Initialize Pinecone
        print("Initializing pinecone")
        pc = Pinecone(api_key=PINECONE_API_KEY)  # replace with your actual Pinecone API key
        print("Pinecone initialized")

        index_name = actor_input.get("index_name")

        embeddings = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=OPENAI_API_KEY  # replace with your actual OpenAI API key
        )
        print(embeddings)

        # Check if our index already exists. If it doesn't, we create it
        existing_indexes = pc.list_indexes()
        if index_name not in existing_indexes:
            print("Creating index")
            # Create a new serverless index
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',  # specify your cloud provider
                    region='us-west-2'  # specify the region
                )
            )

        # Add documents to the index
        # Here you might need to adjust the code based on how you want to handle embeddings and document insertion
        Pinecone.from_documents(documents, embeddings, index_name=index_name)
        print("Documents added")
