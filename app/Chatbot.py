import os
import random
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_or_create_vectorstore(documents, embeddings, index_path="faiss_index"):
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating FAISS index...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(index_path)
        return vectorstore

def generate_questions(llm, docs_text, num_questions=10):
    prompt = f"""
Here is some document content:

\"\"\"
{docs_text}
\"\"\"

Please generate {num_questions} clear quiz questions based ONLY on this content.  
If possible, make some questions multiple-choice with options labeled A), B), C), D).  
List questions separated by blank lines.
"""
    response = llm.invoke(prompt)
    # Split by double newlines, strip whitespace
    questions = [q.strip() for q in response.strip().split("\n\n") if q.strip()]
    return questions

def is_dont_know(ans):
    return ans.strip().lower() in ["dont know", "i don't know", "no idea", "", "idk"]

def check_answer(llm, question, user_answer):
    prompt = f"""
You are a helpful assistant. Evaluate if the user's answer is correct or not.

Question: {question}
User's answer: {user_answer}

Reply only with "Correct" or "Incorrect".
"""
    res = llm.invoke(prompt).strip().lower()
    return res == "correct"

def get_correct_answer(llm, question):
    prompt = f"""
What is the correct answer to this question? Please give a short answer or correct option letter.

Question: {question}
"""
    return llm.invoke(prompt).strip()

def main():
    pdf_path = input("Enter PDF path: ").strip()
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = load_or_create_vectorstore(split_docs, embeddings)

    llm = OllamaLLM(model="llama2")

    # Join text (limit length)
    docs_text = "\n\n".join([d.page_content for d in split_docs])[:3500]

    print("Generating quiz questions...\n")
    questions = generate_questions(llm, docs_text, num_questions=10)

    if not questions:
        print("No questions generated. Exiting.")
        return

    # Shuffle questions so order differs each run
    random.shuffle(questions)

    print("Quiz time! Answer briefly. Type 'end' to quit anytime.\n")
    score = 0
    asked = 0

    while True:
        question = questions[asked % len(questions)]  # cycle through questions endlessly
        print(f"Q{asked + 1}: {question}")
        ans = input("Your answer: ").strip()
        if ans.lower() == "end":
            print("Ending quiz. Thanks for playing!")
            break

        if is_dont_know(ans):
            print(" Incorrect (no answer provided).\n")
        else:
            if check_answer(llm, question, ans):
                print(" Correct!\n")
                score += 1
            else:
                correct_ans = get_correct_answer(llm, question)
                print(f" Incorrect. Correct answer: {correct_ans}\n")

        asked += 1

    print(f"Final score: {score} out of {asked}")

if __name__ == "__main__":
    main()
