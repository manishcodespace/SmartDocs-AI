import requests
import os
import sys

def main():
    url = "http://127.0.0.1:8000/api/chat-pdf"
    file_path = "Manish_test.pdf"
    question = "what is the password of Name: Charlotte Lopez?"

    if not os.path.exists(file_path):
        print(f"Error: Test file '{file_path}' not found in the current directory.")
        sys.exit(1)

    print(f"Sending request to {url}...")
    print(f"Uploading file: {file_path}")
    print(f"Question: {question}\n")

    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/pdf")}
            data = {"question": question}
            response = requests.post(url, files=files, data=data)

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:")
            import json
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error Response:")
            print(response.text)
    except Exception as e:
        print(f"Failed to connect or perform request: {e}")
        print("Ensure the FastAPI server is running on http://127.0.0.1:8000")

if __name__ == "__main__":
    main()
