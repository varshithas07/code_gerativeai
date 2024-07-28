from django.shortcuts import render, redirect
from django.urls import reverse
from llama_index.core import SimpleDirectoryReader, get_response_synthesizer
from llama_index.core import DocumentSummaryIndex
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage,MessageRole
from django.conf import settings
from llama_index.llms.together import TogetherLLM
import os
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
)
import re
from llama_index.core.query_engine import RetrieverQueryEngine
from django.http import HttpResponseRedirect
import PyPDF2
import shutil
from django.http import HttpResponseServerError
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext





def home(request):
    return render(request, 'index.html')

# def doc_view(request):
#     return render(request, 'doc.html')

def doc_view(request):


    return render(request, 'doc.html')

def split_pdf_by_page(pdf_file, output_folder):
    # Delete the output folder if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Create the output folder
    os.makedirs(output_folder)

    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    pdf_names = []

    for page_num in range(num_pages):
        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[page_num])
        output_file_path = os.path.join(output_folder, f'page_{page_num + 1}.pdf')
        pdf_names.append(f'page_{page_num + 1}')
        with open(output_file_path, 'wb') as output_pdf:
            pdf_writer.write(output_pdf)

    return pdf_names


def build_document_summary_index(city_docs):
    # Initialize the embedding model
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=settings.SENTENCE_EMBEDDING_MODEL))
    settings.embed_model = embed_model

    # Initialize the LLM model
    llm = TogetherLLM(model=settings.LLM_MODEL, api_key=settings.LLM_API_KEY)
    settings.llm = llm

    # Set context window size
    settings.context_window = settings.CONTEXT_WINDOW_SIZE

    # Initialize the splitter
    splitter = SentenceSplitter(chunk_size=settings.SPLITTER_CHUNK_SIZE)

    # Initialize the response synthesizer
    response_synthesizer = get_response_synthesizer(response_mode="tree_summarize", use_async=True)

    # Build the document summary index
    doc_summary_index = DocumentSummaryIndex.from_documents(
        city_docs,
        llm=llm,
        transformations=[splitter],
        response_synthesizer=response_synthesizer,
        show_progress=True
    )

    return doc_summary_index






doc_summary_index = None
city_docs = []

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
def process_file(request):
    global doc_summary_index
    global city_docs
    global query_history

    if request.method == 'POST':
        if request.FILES.get('pdf_file'):
            # Handle PDF file upload
            pdf_file = request.FILES['pdf_file']
            output_folder = 'output_folder'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            city_docs = []
            output_folder = 'output_folder'
            pdf_names = split_pdf_by_page(pdf_file, output_folder)
            for wiki_title in pdf_names:
                docs = SimpleDirectoryReader(
                    input_files=[f"{output_folder}/{wiki_title}.pdf"]
                ).load_data()
                for doc in docs:
                    doc.doc_id = wiki_title
                city_docs.extend(docs)
            doc_summary_index = build_document_summary_index(city_docs)
            doc_summary_index.storage_context.persist("index")
            return render(request, 'doc_chat.html')

        elif request.POST.get('user_input'):
            # Handle user input
            user_input = request.POST.get('user_input', '')
            # query_history=[]
            if user_input:
               try:
                   retriever = DocumentSummaryIndexLLMRetriever(
                       doc_summary_index,
                       choice_top_k=3,
                   )
                   response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

                   chat_text_qa_msgs = [
                       ChatMessage(
                           role=MessageRole.SYSTEM,
                           content=(
                               "Summarize the documents.\n"
                               "Always answer the query using the provided context information, "
                           ),
                       ),
                       ChatMessage(
                           role=MessageRole.USER,
                           content=(
                               "Context information is below.\n"
                               "---------------------\n"
                               "{context_str}\n"
                               "---------------------\n"
                              # "Given the context information, Provide a summary to the document .\n"
                               # " Please write a passage to answer the question\n"
                               # "Try to include as many key details as possible.\n"
                               """You will generate increasingly concise, entity-dense summaries of the above
 

  Repeat the following 2 steps 5 times.

  - Step 1: Identify 1-3 informative Entities from the Article
  which are missing from the previously generated summary and are the most
  relevant.

  - Step 2: Write a new, denser summary of identical length which covers
  every entity and detail from the previous summary plus the missing entities

  A Missing Entity is:

  - Relevant: to the main story
  - Specific: descriptive yet concise (5 words or fewer)
  - Novel: not in the previous summary
  - Faithful: present in the Article
  - Anywhere: located anywhere in the Article

  Guidelines:
  - The first summary should be long (4-5 sentences, approx. 80 words) yet
  highly non-specific, containing little information beyond the entities
  marked as missing.

  - Use overly verbose language and fillers (e.g. "this article discusses") to
  reach approx. 80 words.

  - Make every word count: re-write the previous summary to improve flow and
  make space for additional entities.

  - Make space with fusion, compression, and removal of uninformative phrases
  like "the article discusses"

  - The summaries should become highly dense and concise yet self-contained,
  e.g., easily understood without the Article.

  - Missing entities can appear anywhere in the new summary.

  - Never drop entities from the previous summary. If space cannot be made,
  add fewer new entities.

  > Remember to use the exact same number of words for each summary.
  Answer in passage.

  > The JSON in `summaries_per_step` should be a list (length 5) of
  dictionaries whose keys are "missing_entities" and "denser_summary".
"""
                           ),
                       ),
                   ]

                   text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
                   query_2 = f"""{user_input} """
                   query_engine = RetrieverQueryEngine(
                       retriever=retriever,
                       response_synthesizer=response_synthesizer
                   )
                   response__2 = doc_summary_index.as_query_engine(text_qa_template=text_qa_template).query(query_2)
                   answer = formatted_answer(str(response__2))
                   query_history.append({"question": user_input, "answer": answer})
                   return render(request, 'doc_chat.html', {'query_history': query_history})
               except AttributeError:
                   # Handle the case when doc_summary_index is None
                   return HttpResponseServerError("something went wrong. Please retry uploading the PDF.")
    # Render a response for the GET request
    return render(request, 'doc_chat.html')

def doc_chat(request):
    # return render(request, 'doc_chat.html')

    return render(request, 'doc_chat.html')


