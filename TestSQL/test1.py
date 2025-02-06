from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import multiprocessing
from multiprocessing import Queue
import json

template = """Use the following pieces of context and metadata to answer the question at the end. Answer in the same language the question was asked.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        You will format your answer in json, with the keys "answer" and "frames_ids". Always include these keys, even if you did not find anything. Just say it and return an empty list for "frames_ids".
        The value of "answer" will be the answer to the question, and the value of "frames_ids" will be a list of frame_ids from which you got the information from using the metadata.

        Context: {context}
        Question: {question}
        Metadata: {md}

        Helpful Answer:"""

prompt = PromptTemplate(
    input_variables=["question", "context", "md"], template=template
)

chromadb = Chroma(
                persist_directory="utils.CACHE_PATH",
                embedding_function=OpenAIEmbeddings(),
                collection_name="memento_db",
            )

retriever = chromadb.as_retriever()
memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, input_key="question"
        )
qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature=0.8),
            retriever,
            memory=memory,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": prompt},
        )


def process_chat_query(self):
    print("Starting chat query process")
    while True:
        q = self.query_queue.get()
        inp = q["input"]

        print("Query:", inp)
        print("Retrieving relevant documents")
        docs = self.retriever.get_relevant_documents(inp)
        print("done")
        md = {}
        for doc in docs:
            frame_id = doc.metadata["id"]
            window_title = doc.metadata["id"]
            date = doc.metadata["time"]
            md[frame_id] = {
                "window_title": window_title,
                "date": date,
            }

        result = self.qa(inputs={"question": inp, "md": md})

        try:
            result = json.loads(result["answer"])
        except json.decoder.JSONDecodeError as e:
            print("Error decoding json:", e)
            result = {"answer": "Error decoding json", "frames_ids": []}

        print("Answer:", result["answer"])
        print("frames_ids:", result["frames_ids"])
        self.answer_queue.put(result)



def query_llm(self):
    if len(self.textinput.value) > 0:
        chat_history_entry = {}
        chat_history_entry["question"] = self.textinput.value
        chat_history_entry["answer"] = None
        chat_history_entry["frames"] = {}
        self.chat_history.append(chat_history_entry)
        self.query_queue.put({"input": self.textinput.value})
        self.textinput.value = ""

query_queue = Queue()
answer_queue = Queue()
multiprocessing.Process(target=process_chat_query, args=()).start()