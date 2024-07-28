from django.http import HttpResponseServerError, JsonResponse
from django.shortcuts import render
from django.conf import settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import re
import os
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.llms.together import TogetherLLM
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from django.utils import timezone
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from django.core.mail import EmailMessage
from llama_index.core.postprocessor import SentenceTransformerRerank

# Create your views here.


rerank = SentenceTransformerRerank( model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=7)

#function to render home page
def home(request):
    return render(request, 'index.html')

#function of hr_analytics
def embed(documents):
    # Initialize the embedding model
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=settings.SENTENCE_EMBEDDING_MODEL))
    settings.embed_model = embed_model

    # Initialize the LLM model
    llm_hr = TogetherLLM(model=settings.LLM_MODEL, api_key=settings.LLM_API_KEY)
    # print("llm_hr",llm_hr)
    settings.llm = llm_hr

    # Set context window size
    settings.context_window = settings.CONTEXT_WINDOW_SIZE

    # Create a local Qdrant vector store
    client = qdrant_client.QdrantClient(path="qdrant_hr")


    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection"
    )

    storage_context = StorageContext.from_defaults(
        vector_store=text_store
    )

    index2 = VectorStoreIndex.from_documents(documents, embed_model=embed_model,storage_context=storage_context)
    return index2

query_history=[]
index_hr = None
documents = SimpleDirectoryReader("./hr_documents").load_data()
index_hr = embed(documents)


def athena_chat(request):
    global index_hr

    global query_history

    if request.method == 'POST':
        if request.POST.get('user_input'):
            # Handle user input
            user_input = request.POST.get('user_input', '')

            print(user_input)


            if user_input:
                print(user_input, "entered the if condition")
                try:
                    chat_text_qa_msgs = [
                        ChatMessage(
                            role=MessageRole.SYSTEM,
                            content=(
                               
                                """You are a  HR assistant chatbot system specifically developed for Quadwave.. Your goal is to answer questions as accurately as possible based on
                                   the instructions and context provided.\n"""
                                "Always answer the query using the provided context information, "
                                "and not prior knowledge.\n"
                                f"Today is {timezone.now().strftime('%Y-%m-%d')}."
                                "If the question is not related to Quadwave policy, respond with 'I can only answer questions related to Quadwave policy.'\n"
                                "If asked about the next holiday, consider the present date and fetch the next holiday date from the document.\n"
                                "For general questions like 'How are you?' or 'Who are you?', respond accordingly, mentioning that you're here to assist with Human resources within the company.\n"

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
                    query_engine = index_hr.as_query_engine(similarity_top_k=3, node_postprocessors=[rerank],text_qa_template=text_qa_template )
                    query_2 = f"""{user_input}."""""
                   
                    print(timezone.now().strftime('%Y-%m-%d'))
                    response__2 = query_engine.query(query_2)
                    print("response__2", response__2)
                    answer = format_answer(str(response__2))
                    query_history.append({"question": user_input, "answer": answer})

                    return render(request, 'athena_chat.html',{'query_history':query_history})

                except AttributeError:
                    # Handle the case when doc_summary_index is None
                    return HttpResponseServerError("Something went wrong. Please retry uploading the PDF.")


    return render(request,'athena_chat.html')


def save_query_history_to_pdf(query_history):
    try:
        pdf_path = './chat_history.pdf'  # Specify the path where the PDF will be saved
        if os.path.exists(pdf_path):
            os.remove(pdf_path)  # Delete the existing PDF file

        doc = SimpleDocTemplate(pdf_path)
        styles = getSampleStyleSheet()
        flowables = []

        # Add query history to PDF
        for entry in query_history:
            question = entry['question']
            answer = entry['answer']
            flowables.append(Paragraph(f"Question: {question}", styles['Normal']))
            flowables.append(Paragraph(f"Answer: {answer}", styles['Normal']))
            # flowables.append(Paragraph("", styles['Normal']))  # Add empty line for separation
            flowables.append(Spacer(1, 12))

        doc.build(flowables)
        return pdf_path
    except Exception as e:
        raise HttpResponseServerError(f"Error creating PDF: {str(e)}")


def format_answer(answer):
    current_number = 1
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
        bold_match = re.match(r'^(\*\*)(.+?)(\*\*)', line)

        if bold_match:
                if not in_list or list_type != 'ol':
                    if in_list:  # Close the previous list
                        formatted_lines.append('</ul>' if list_type == 'ul' else '</ol>')
                    formatted_lines.append('<ol>') 
                    in_list = True
                    list_type = 'ol'
                formatted_lines.append(f'<p style="font-weight: bold;">{current_number}. {bold_match.group(2).strip()}</p>')
                current_number += 1  # Increment current numbering
        
        elif numbered_match:
            if not in_list or list_type != 'ul':
                if in_list:  # Close the previous list
                    formatted_lines.append('</ol>' if list_type == 'ul' else '</ul>')
                formatted_lines.append('<ul>') 
                in_list = True
                list_type = 'ul'
            formatted_lines.append(f'<li style="margin-left: 50px;">{numbered_match.group(2).strip()}</li>')

        elif asterisk_match or asterisk_items:
            if not in_list or list_type != 'ul':
                if in_list:  # Close the previous list
                    formatted_lines.append('</ol>' if list_type == 'ol' else '</ul>')
                formatted_lines.append('<ul>')
                in_list = True
                list_type = 'ul'
            if asterisk_items:
                for item, _ in asterisk_items:
                    formatted_lines.append(f'<li style="margin-left: 50px;">{item.strip()}</li>')
            else:
                formatted_lines.append(f'{asterisk_match.group(2).strip()}</li>')
    

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


def save_email_content(request):
    if request.method == 'POST':
        email_body = request.POST.get('email_hr_body', '')
        print("email_body",email_body)
        pdf_path = save_query_history_to_pdf(query_history)

        # Send email to the user's email address
        subject = 'User Query'  # Specify your subject
        message = f"Hello HR \n\n {email_body} \n\n Please find the chat history attached. \n\n Regards"  # Use the email body as the message
        from_email = 'sivasuro.1234@gmail.com'  # Sender's email address
        to_email = 'varshithas.512@gmail.com'  # User's email address

        # Send the email
        # send_mail(subject, message, from_email, [to_email])
        email = EmailMessage(subject, message, from_email, [to_email])
        if os.path.exists(pdf_path):
            email.attach_file(pdf_path)

            # Send the email
        email.send()
        return JsonResponse({'result': 'Email content received successfully.'})

    # Return an error response if the request method is not POST
    return JsonResponse({'error': 'Method not allowed.'}, status=405)
