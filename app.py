import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pypdf import PdfReader

# Load environment variables
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

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
    tmp_path = None
    
    # 1. Validate API Key exists
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="Google Gemini API key is missing. Please set GEMINI_API_KEY or GOOGLE_API_KEY in the environment."
        )

    # 2. Validate question is not empty or whitespace-only
    if not question or not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty or only whitespace."
        )

    # 3. Validate file exists and is a PDF
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # 4. Limit file size (10 MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    content_length = file.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds the 10 MB limit."
        )
    if hasattr(file, "size") and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds the 10 MB limit."
        )

    # 5. Create a temporary file to save the uploaded PDF content
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

    # 6. Verify file size on disk (if not caught by headers)
    if tmp_path:
        file_size = os.path.getsize(tmp_path)
        if file_size > MAX_FILE_SIZE:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise HTTPException(
                status_code=400,
                detail="File size exceeds the 10 MB limit."
            )

    try:
        # 7. Check if PDF is encrypted or password-protected
        try:
            reader = PdfReader(tmp_path)
            if reader.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded PDF is password-protected or encrypted. Please upload an unprotected version."
                )
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or corrupted PDF file: {str(e)}"
            )

        # 8. Load and parse PDF content
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        if not docs:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF file is empty or could not be parsed."
            )

        # 9. Verify PDF contains extractable/readable text (not purely scanned/blank images)
        total_text = "".join([doc.page_content for doc in docs]).strip()
        if len(total_text) < 10:
            raise HTTPException(
                status_code=400,
                detail="No readable text could be extracted from the PDF. It might be scanned/image-only or require OCR (Optical Character Recognition)."
            )

        # Split the document content into manageable chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        # Retrieve configuration from environment variables
        embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

        # Build a FAISS index over the document chunks (per-request, in-process)
        # FAISS uses optimized C++ ANN search — production-grade for CPU deployments
        vector_db = FAISS.from_documents(
            documents=split_docs,
            embedding=embeddings
        )

        # Search for context relevant to the user query
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

    except HTTPException as he:
        raise he
    except Exception as e:
        err_msg = str(e)
        if "API_KEY_INVALID" in err_msg or "API key not valid" in err_msg or "invalid API key" in err_msg.lower():
            raise HTTPException(
                status_code=500,
                detail="Invalid Google Gemini API key. Please check your GEMINI_API_KEY / GOOGLE_API_KEY environment variables."
            )
        elif "quota" in err_msg.lower() or "429" in err_msg or "resource_exhausted" in err_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="API rate limit or quota exceeded. Please try again in a few moments."
            )
        elif "service unavailable" in err_msg.lower() or "503" in err_msg:
            raise HTTPException(
                status_code=503,
                detail="Gemini API service is temporarily unavailable. Please try again later."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while processing the PDF or query: {err_msg}"
            )

    finally:
        # Always clean up the temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
