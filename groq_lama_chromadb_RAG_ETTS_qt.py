import sys
import threading
from qtpy import QtCore, QtWidgets

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

import groq_lama_chromadb_RAG_ETTS as rag


class ChatWorker(QtCore.QThread):
    response_ready = QtCore.Signal(str)

    def __init__(self, prompt, system_message, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.system_message = system_message

    def run(self):
        response = rag.chat_with_model(
            self.prompt,
            self.system_message,
            rag.groq_model,
            rag.ollama_model,
            conversation_history=rag.conversation_history,
        )
        self.response_ready.emit(response)


class ChatWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Groq Chat GUI")
        self.resize(600, 400)

        layout = QtWidgets.QVBoxLayout(self)

        self.chat_display = QtWidgets.QTextEdit(readOnly=True)
        layout.addWidget(self.chat_display)

        input_layout = QtWidgets.QHBoxLayout()
        self.input_field = QtWidgets.QLineEdit()
        self.send_button = QtWidgets.QPushButton("Send")
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)

        self.send_button.clicked.connect(self.handle_send)
        self.input_field.returnPressed.connect(self.handle_send)

        self.system_message = (
            "You are a helpful assistant. You will give precise and concise "
            "answers from the given context. if the context doesnot have the "
            "answer then give it from your knowledge"
        )

    def handle_send(self):
        text = self.input_field.text().strip()
        if not text:
            return
        self.chat_display.append(f"You: {text}")
        self.input_field.clear()

        self.worker = ChatWorker(text, self.system_message)
        self.worker.response_ready.connect(self.display_response)
        self.worker.start()

    def display_response(self, response):
        self.chat_display.append(f"Assistant: {response}")


def initialize():
    rag.dont_read_tts = False
    rag.just_query_file_search = False
    rag.conversation_history = [{"role": "system", "content": "Welcome to Ollama Chat!"}]

    ollama_thread = threading.Thread(target=rag.check_and_start_ollama, daemon=True)
    ollama_thread.start()

    client = chromadb.PersistentClient(
        path=rag.CHROMADB_PATH,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    rag.collection = client.get_collection(rag.collection_name)

    rag.get_relevant_context_hybrid(
        user_input="just loading ollama embeddings model and chromadb, dont respond"
    )


if __name__ == "__main__":
    initialize()
    app = QtWidgets.QApplication(sys.argv)
    win = ChatWindow()
    win.show()
    sys.exit(app.exec())
