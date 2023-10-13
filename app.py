from flask import Flask, jsonify, request
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import constants
from flask_cors import CORS, cross_origin

os.environ["OPENAI_API_KEY"] = constants.APIKEY
PERSIST = False
app = Flask(__name__)
CORS(app)

# Enable to save to disk & reuse the model (for repeated queries on the same data)
@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return "hello"
@app.route("/chat", methods=["POST"])
@cross_origin()
def chat():
    json_data = request.get_json()
    query = json_data.get('query')
    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
        loader = DirectoryLoader("data/")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    prompt_template = PromptTemplate.from_template(
        "You are a helpful assistant named {name}.Answer these question:{user_input}"  
    )
    messages = prompt_template.format(
        name="Dawa",
        user_input=query 
    )
    result = chain({"question": messages, "chat_history": chat_history})
    answer=result['answer']
    chat_history.append((messages, result['answer']))
    return jsonify(answer)

if __name__ == '__main__':
    app.run()

