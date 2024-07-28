
# %%
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os
from llama_index.llms.together import TogetherLLM
from llama_index.core.llms import ChatMessage
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from datetime import datetime
# Text QA Prompt template construction
#change date format to "April 22, 2024 (mm-dd-yyyy)"
chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You are a HR assistant chatbot system specifically developed for Quadwave.\n"
             "Always answer the query using the provided context information, "
            "and not prior knowledge.\n"
            f"Today is {datetime.now().strftime('%Y-%m-%d')}." 
        ),
    ), 
    ChatMessage(
        role=MessageRole.USER,
        content=(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str} in bullet points or numbered list where appropriate.\n"
        ),
    ),
]
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

def embed(): 
    # #The below is for images and is not used in HR analytics. Leaving this here for reference. Only the text collection is stored and accessed for this project - Sasi
    # Settings.embed_model = "clip"
    # #Load CLIP model
    # model = SentenceTransformer('clip-ViT-B-32')

    import qdrant_client
    from llama_index.core import SimpleDirectoryReader


    # Create a local Qdrant vector store
    client = qdrant_client.QdrantClient(path="qdrant_img_db_2")

    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection"
    )
    image_store = QdrantVectorStore(
        client=client, collection_name="image_collection"
    )
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )
    #hugging face embedding mmodel is used using the llama index langchain integration (LangchainEmbedding)
    lc_embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
    )
    embed_model = LangchainEmbedding(lc_embed_model)

    documents = SimpleDirectoryReader("./SOURCE_DOCUMENTS").load_data()
    index2 = VectorStoreIndex.from_documents(documents, embed_model=embed_model,storage_context=storage_context)
    return index2


# index2 is saved as a global variable to avoid re-creating it every time the query function is called
index2 = embed()
def handle_greetings(query):
    greetings = ["hi", "hello", "hey", "greetings", "morning", "afternoon", "evening"]
    return any(greeting == query.lower() for greeting in greetings)

def query(input):
    while True:
            query = input
            #top_k is set to 3 for the purpose of this project. This can be changed to a higher or lower value for better accuracy. This is the number of documents to be retrieved from the index.
            retriever2 = index2.as_retriever(similarity_top_k=3)
            retrieval_results = retriever2.retrieve(query)
            llm = TogetherLLM(
                    model="META-LLAMA/LLAMA-3-70B-CHAT-HF", api_key="b0264598e8c7613afd2a6f437775f0ff207adcde227255d1f7b03fde0903abd5"
                )
            # %%
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            # llm is set to togethercomputer/llama-2-70b-chat. Default is openAI. This can be changed to any other supported model.
            Settings.llm = llm
            response = index2.as_query_engine(text_qa_template=text_qa_template).query(query)
            print("user query: ", query)
            print("answer: ",response)
            return response

