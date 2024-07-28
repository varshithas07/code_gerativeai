from django.shortcuts import render
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage,MessageRole
from django.conf import settings
from llama_index.llms.together import TogetherLLM
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
import re
from django.http import HttpResponseServerError
from llama_index.core import StorageContext
from llama_index.core.postprocessor import SentenceTransformerRerank

# Create your views here.



# rerank = SentenceTransformerRerank( model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
def build_document_summary_index(documents):
    # Initialize the embedding model
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=settings.SENTENCE_EMBEDDING_MODEL))
    settings.embed_model = embed_model

    # Initialize the LLM model
    llm_e_chat = TogetherLLM(model=settings.LLM_MODEL, api_key=settings.LLM_API_KEY_CHAT)
    # print("llm_e_chat",llm_e_chat)
    settings.llm = llm_e_chat

    # Set context window size
    settings.context_window = settings.CONTEXT_WINDOW_SIZE


    # # Initialize client, setting path to save data
    # db = chromadb.PersistentClient(path="./chroma_db")
    #
    # # Create collection
    # chroma_collection = db.get_or_create_collection("quickstart")
    #
    # # Assign Chroma as the vector_store to the context
    # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)

    client = qdrant_client.QdrantClient(path="qdrant_edoc")

    # Create a local Qdrant vector store
    vector_store = QdrantVectorStore(
        client=client, collection_name="text_collection")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create your index
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    return index


def formatted_answer(answer):
    lines = answer.split('\n')
    formatted_lines = []
    in_list = False
    list_type = None

    for line in lines:
        # Check for numbered list
        numbered_match = re.match(r'^(\d+\.\s)(.+)', line)
        # Check for asterisk list
        asterisk_match = re.match(r'^(\*\s)(.+)', line)
        # Split asterisk list items that are on the same line
        asterisk_items = re.findall(r'\*\s(.+?)(?=(\*\s|$))', line)

        if numbered_match:
            if not in_list or list_type != 'ol':
                if in_list:  # Close the previous list
                    formatted_lines.append('</ul>' if list_type == 'ul' else '</ol>')
                formatted_lines.append('<ol>')
                in_list = True
                list_type = 'ol'
            formatted_lines.append(f'<li>{numbered_match.group(2).strip()}</li>')

        elif asterisk_match or asterisk_items:
            if not in_list or list_type != 'ul':
                if in_list:  # Close the previous list
                    formatted_lines.append('</ol>' if list_type == 'ol' else '</ul>')
                formatted_lines.append('<ul>')
                in_list = True
                list_type = 'ul'
            if asterisk_items:
                for item, _ in asterisk_items:
                    formatted_lines.append(f'<li>{item.strip()}</li>')
            else:
                formatted_lines.append(f'<li>{asterisk_match.group(2).strip()}</li>')

        else:
            if in_list:  # Close the previous list
                formatted_lines.append('</ul>' if list_type == 'ul' else '</ol>')
                in_list = False
            # Wrap non-list lines in paragraphs or handle them appropriately
            formatted_lines.append(f'<p>{line.strip()}</p>')

    # Close any open list tags
    if in_list:
        formatted_lines.append('</ul>' if list_type == 'ul' else '</ol>')

    # Combine all formatted lines
    formatted_output = ''.join(formatted_lines)

    return formatted_output

query_history=[]
index_edoc = None

documents = SimpleDirectoryReader("./e_document").load_data()
index_edoc = build_document_summary_index(documents)

def e_doc(request):
    global index_edoc

    global query_history

    if request.method == 'POST':
        if request.POST.get('user_input'):
            # Handle user input
            user_input = request.POST.get('user_input', '')
            print(user_input)

            if user_input:
                print(user_input,"entered the if condition")
                try:
                    chat_text_qa_msgs = [
                        ChatMessage(
                            role=MessageRole.SYSTEM,
                            content=(
                                """You are an e-government online assistant chatbot system specifically developed for e-government policy. 
                                Your goal is to answer questions as accurately as possible based on the instructions and context provided.\n"""
                                "If the question is not related to e-government policy, respond with 'I can only answer questions related to e-government policy.'\n"
                                "For general questions like 'Hi', 'How are you?', or 'Who are you?', respond accordingly, mentioning that you're here to assist with e-government policy using the context provided in the document and how you can help the user.\n"
                                 
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
                                
                                "answer the question: {query_str} provided in bullet points or numbered list where appropriate.\n"
                                
                                

                               
                            ),
                        ),
                    ]
                    text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
                    query_2 = f"""{user_input}."""""
                    # query_engine = RetrieverQueryEngine(
                    #     retriever=retriever,
                    #     response_synthesizer=response_synthesizer
                    # )
                    print(index_edoc)
                    response__2 = index_edoc.as_query_engine(text_qa_template=text_qa_template).query(query_2)
                    # query_engine = index_edoc.as_query_engine(similarity_top_k=3, node_postprocessors=[rerank],
                    #                                       text_qa_template=text_qa_template)
                    # response__2 = query_engine.query(query_2)

                    print("response__2",response__2)
                    answer = formatted_answer(str(response__2))
                    query_history.append({"question": user_input, "answer": answer})
                    return render(request, 'e_gov_chat.html', {'query_history': query_history})


                except AttributeError:
                    # Handle the case when doc_summary_index is None
                    return HttpResponseServerError("reload the page.")
        # Render a response for the GET request
    return render(request, 'e_gov_chat.html')
def e_chat(request):
    # return render(request, 'doc_chat.html')

    return render(request, 'e_gov_chat.html')
