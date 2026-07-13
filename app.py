import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore

app = FastAPI(
    title="PDF Chat API",
    description="A simple API that allows uploading a PDF and asking questions from it."
)

@app.get("/", response_class=FileResponse)
async def get_ui():
    return FileResponse("index.html")

@app.post("/api/chat-pdf")
async def chat_pdf(
    file: UploadFile = File(..., description="The PDF file to upload"),
    question: str = Form(..., description="The question you want to ask from the PDF")
):
    # Validate that it is a PDF file
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Create a temporary file to save the uploaded PDF content
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save the uploaded file: {str(e)}"
        )

    try:
        # Load the PDF content
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        if not docs:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF file is empty or could not be parsed."
            )

        # Split the document content into manageable chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        # Retrieve configuration from environment variables
        embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

        # Set up InMemoryVectorStore and index the chunks
        vector_db = InMemoryVectorStore.from_documents(
            documents=split_docs,
            embedding=embeddings
        )

        # Search for context relevant to the user query
        # Using k=3 to get a reasonable amount of context for answering
        documents = vector_db.similarity_search(query=question, k=3)

        context = ""
        for doc in documents:
            context += doc.page_content + "\n\n"

        # Build prompt
        prompt = f"""You are a helpful assistant.
Answer the user's question directly and concisely based on the context. If the question asks for a specific value (like a password), return ONLY that value/password with no additional text.

Context:
{context}

Question: {question}
Answer:"""

        # Initialize the Chat LLM and run the query
        chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.1-flash-lite")
        llm = ChatGoogleGenerativeAI(model=chat_model)
        answer = llm.invoke(prompt)

        # Extract text content from response
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

        return {
            "success": True,
            "filename": file.filename,
            "question": question,
            "answer": text_content.strip()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the PDF or query: {str(e)}"
        )

    finally:
        # Always clean up the temporary file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
