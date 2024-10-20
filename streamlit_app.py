import streamlit as st
import openai
import tempfile
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import TextLoader


# Set your OpenAI API key here
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

st.title('RAG Search')

#from langchain_openai import ChatOpenAI
#llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature = 0.5)
#response = llm.invoke("What is LLM")
#print(response)
#st.write("Before Search using RAG, RESPONSE : ", response.content)

if not openai_api_key.startswith('sk-'):
   st.warning('Please enter your OpenAI API key!', icon='⚠')
if openai_api_key.startswith('sk-'):
    uploaded_file=st.file_uploader("Upload text files", type=["txt", "pdf"], accept_multiple_files=False)
    user_question=st.text_input("Enter your question:")
    
    def get_docs_from_file(uploaded_file):
        docs = []
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
             tmp_file.write(uploaded_file.read())
             tmp_file_path = tmp_file.name
        # Load the document using TextLoader
        loader = TextLoader(tmp_file_path)
        docs.extend(loader.load())
    
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
             chunk_size=200,
             chunk_overlap=20
        )
        split_docs = text_splitter.split_documents(docs)
        return split_docs
        
#def get_docs():
#    loader = WebBaseLoader('https://www.techtarget.com/whatis/feature/Foundation-models-explained-Everything-you-need-to-know')
#    docs = loader.load()

#    text_splitter = RecursiveCharacterTextSplitter(
#        chunk_size=200,
#        chunk_overlap=20
#    )

#    splitDocs = text_splitter.split_documents(docs)
#    return splitDocs

def create_vector_store(docs):
    embedding = OpenAIEmbeddings(api_key=openai_api_key)
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(api_key=openai_api_key,
        temperature=0.4,
        model='gpt-3.5-turbo-1106'
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question.
    Context: {context}
    Question: {input}
    """)

    #print(prompt)

    # chain = prompt | model
    # We are creating the chain to add documents
    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    # Retrieving the top 1 relevant document from the vector store , We can change k to 2 and get top 2 and so on
    retriever = vectorStore.as_retriever(search_kwargs={"k": 1})

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain
    
if st.button("Query Doc"):
       if uploaded_file and user_question:
           with st.spinner("Processing..."):
              try:
                  split_docs = get_docs_from_file(uploaded_file)
                  vector_store = create_vector_store(split_docs)
                  chain = create_chain(vector_store)

                  context = " ".join([doc.page_content for doc in split_docs])
                  #st.write(f"Context: {context}")
                  #st.write(f"Question: {user_question}")
                  response = chain.invoke({"context": context, "input": user_question})
                  #st.write("### Full Response")
                  #st.write(response)
                  if 'answer' in response:
                      st.write("### Answer")
                      st.write(response['answer'])
              except Exception as e:
                  st.error(f"An error occurred: {e}")
       else:
          st.warning("Please enter your OpenAI API Key")
           
