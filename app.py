from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# for getting embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# for storing vectors in vector store
from langchain_community.vectorstores import InMemoryVectorStore

# document loading....
loader=PyPDFLoader("./Manish_test.pdf")
docs=loader.load()
print(len(docs))

# spliting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(docs)
print(len(docs))
# embedding and vectorstore
import os

embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL")
embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
vector_db = InMemoryVectorStore.from_documents(
    documents=docs,
    embedding=embeddings
)
# user query
query = "what is the password of Name: Charlotte Lopez?"

documents = vector_db.similarity_search(query=query, k=1)

context = ""
for doc in documents:
    context += doc.page_content + "\n\n"

prompt = f"""You are a helpful assistant.
Answer the user's question directly and concisely based on the context. If the question asks for a specific value (like a password), return ONLY that value/password with no additional text.

Context:
{context}

Question: {query}
Answer:"""

chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.1-flash-lite")
llm = ChatGoogleGenerativeAI(model=chat_model)
answer = llm.invoke(prompt)

# Safely extract text content from the response
text_content = ""
if isinstance(answer.content, str):
    text_content = answer.content
elif isinstance(answer.content, list):
    for part in answer.content:
        if isinstance(part, dict) and part.get("type") == "text":
            text_content += part.get("text", "")
        elif isinstance(part, str):
            text_content += part
else:
    text_content = str(answer.content)

print("\n--- Direct Answer ---")
print(text_content.strip())
print("---------------------\n")


