import sys
from transformers import AutoTokenizer, LlamaTokenizer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QComboBox, QFrame, QHBoxLayout
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class TokenChecker(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        title_label = QLabel("Tokenize your prompt", self)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setMargin(10)
        main_layout.addWidget(title_label)

        combo_layout = QHBoxLayout()
        combo_label = QLabel("Select Tokenizer:")
        combo_label.setFont(QFont("Arial", 14))
        self.tokenizer_selector = QComboBox(self)
        self.tokenizer_selector.addItems(["Mistral", "Llama 2"])
        self.tokenizer_selector.setFont(QFont("Arial", 12))
        combo_layout.addWidget(combo_label)
        combo_layout.addWidget(self.tokenizer_selector)
        combo_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(combo_layout)

        self.prompt_input = QTextEdit(self)
        self.prompt_input.setPlaceholderText("Enter your prompt here...")
        self.prompt_input.setFont(QFont("Arial", 12))
        main_layout.addWidget(self.prompt_input)

        self.check_button = QPushButton('Tokenize', self)
        self.check_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.check_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        self.check_button.clicked.connect(self.check_tokens)
        main_layout.addWidget(self.check_button)

        output_layout = QVBoxLayout()

        self.token_label = QLabel("Tokens: ", self)
        self.token_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.token_label.setStyleSheet("background-color: #FFFFCC;")
        output_layout.addWidget(self.token_label)

        self.char_label = QLabel("Characters: ", self)
        self.char_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.char_label.setStyleSheet("background-color: #CCFFFF;")
        output_layout.addWidget(self.char_label)

        main_layout.addLayout(output_layout)

        self.result_label = QLabel(self)
        self.result_label.setFont(QFont("Arial", 14))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        main_layout.addWidget(self.result_label)

        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(main_layout)
        self.setWindowTitle('Tokenizer')
        self.setFixedSize(1200, 1000)

    def check_tokens(self):
        selected_tokenizer = self.tokenizer_selector.currentText()
        
        if selected_tokenizer == "Mistral":
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        elif selected_tokenizer == "Llama 2":
            tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
        
        prompt = self.prompt_input.toPlainText()
        tokenized_prompt = tokenizer.encode(prompt, truncation=True)
        token_length = len(tokenized_prompt)
        char_length = len(prompt)

        self.token_label.setText(f"Tokens: {token_length}")
        self.char_label.setText(f"Characters: {char_length}")

        if selected_tokenizer == "Mistral":
            max_tokens = 8000
        elif selected_tokenizer == "Llama 2":
            max_tokens = 4000

        if token_length > max_tokens:
            self.result_label.setText(f"Warning: The prompt exceeds the token limit by {token_length - max_tokens} tokens for {selected_tokenizer}")
        else:
            self.result_label.setText(f"Good to go! The prompt is within the {max_tokens} token limit for {selected_tokenizer}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TokenChecker()
    window.show()
    sys.exit(app.exec_())
