import os
import time
import pickle
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


from ..llm import llm_model
from ..main import script_dir_path

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

def fetch_docs(urls: List[str]) -> List[str]: 

# docs, docs_list = [], []
# if not docs:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    return docs_list
# text = "The model is finetuned to only predict $y_n$ where conditioned on the sequence prefix, such that the model can self-reflect to produce better output based on the feedback sequence. The model can optionally receive multiple rounds of instructions with human annotators at test time.\nTo avoid overfitting, CoH adds a regularization term to maximize the log-likelihood of the pre-training dataset. To avoid shortcutting and copying (because there are many common words in feedback sequences), they randomly mask 0% - 5% of past tokens during training.\nThe training dataset in their experiments is a combination of WebGPT comparisons, summarization from human feedback and human preference dataset"
# llm_model.embeddings
def get_document_chunks(chunks: List[str]): #**** Verify that the typehint matches ****#
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=0
    )

    pages_split = text_splitter.split_documents(chunks)

    return pages_split
# print("Pages split process completed")
# formated_pages_split = [(f"{i}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. { item}") for i, item in enumerate(pages_split)]
# print(len(formated_pages_split))
# print(formated_pages_split)
# print(formated_pages_split[13])
# return pages_split
# fetch_docs(urls)

faiss_dir_path = os.path.join(script_dir_path,"faiss_index")
faiss_doc_pkl_file_path = os.path.join(script_dir_path, "faiss_doc.pkl")
index_faiss_file_path = os.path.join(faiss_dir_path, "index.faiss")
index_pkl_file_path = os.path.join(faiss_dir_path, "index.pkl")


# print(pages_split)
try:
    if os.path.exists(faiss_dir_path) and \
    os.path.isfile(index_faiss_file_path) and \
    os.path.isfile(index_pkl_file_path):
        print("loading from cache since vector db already exist")

        vector_db = FAISS.load_local(
            folder_path=faiss_dir_path,
            embeddings=llm_model.embeddings,
            # index_name="adaptive_vector_index",
            allow_dangerous_deserialization=True
        )

    else:
        print("Standby, Creating new vector store for project")
        docs_list = fetch_docs(urls)
        pages_split = get_document_chunks(docs_list)

        docs_length = len(pages_split)

        if docs_length > 100:
            # print("docs length is bulky, performing batch process")
            BATCH_SIZE = 75
            vector_db = None

            for i in range(0, docs_length, BATCH_SIZE):
                batch_pages_split = pages_split[i:(i + BATCH_SIZE)]

                # print(f"current batch content is thus: {batch_pages_split}")
                batch_vector_db = FAISS.from_documents(
                    documents=batch_pages_split,
                    embedding=llm_model.embeddings
                )

                # print("batch_vector_db contents are thus: ", batch_vector_db)
                if vector_db is None:
                    vector_db = batch_vector_db
                else:
                    vector_db.merge_from(batch_vector_db)
                # This is needed for avoid Error 429
                time.sleep(45)

            vector_db.save_local(
                folder_path=faiss_dir_path,
            # index_name="adaptive_vector_index"
            )
        else:
            vector_db = FAISS.from_documents(
                    documents=pages_split,
                    embedding=llm_model.embeddings
                )


            vector_db.save_local(folder_path=faiss_dir_path,
            # index_name="adaptive_vector_index"
            )

        with open(faiss_doc_pkl_file_path, "wb") as f:
            pickle.dump(pages_split, f)

        print("Successfully created a vector database.")

except Exception as e:
    print(f"Error setting up FAISS: {str(e)}")
    raise e


retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={'k':1} #'k' is the amount of chunks to return: in this  case 5
)
