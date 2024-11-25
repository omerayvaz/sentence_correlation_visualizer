from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, \
    QTextEdit, QLineEdit, QFileDialog
from PyQt5.QtCore import QFile, QTextStream
import matplotlib.pyplot as plt
import re
import functions
import nltk
import networkx as nx
from rouge import Rouge
import math
import sys

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.fig = plt.figure(figsize=(9, 6))
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        graph_widget = QWidget()
        graph_layout = QVBoxLayout()
        graph_layout.addWidget(self.toolbar)
        graph_layout.addWidget(self.canvas)
        graph_widget.setLayout(graph_layout)

        layout_left = QVBoxLayout()

        self.label_similarity_threshold = QLabel("Sent. Corr. Threshold Value:")
        layout_left.addWidget(self.label_similarity_threshold)
        self.input_similarity_threshold = QLineEdit()
        layout_left.addWidget(self.input_similarity_threshold)

        self.label_score_threshold = QLabel("Sent. Score Threshold Value:")
        layout_left.addWidget(self.label_score_threshold)

        self.input_score_threshold = QLineEdit()
        layout_left.addWidget(self.input_score_threshold)

        self.button_print_values = QPushButton("Save(Values)")
        self.button_print_values.clicked.connect(self.print_values)
        layout_left.addWidget(self.button_print_values)

        self.button_clear_values = QPushButton("Clear(Values)")
        self.button_clear_values.clicked.connect(self.clear_values)
        layout_left.addWidget(self.button_clear_values)

        layout_left.addStretch()

        self.button_draw = QPushButton("Draw")
        self.button_draw.clicked.connect(self.draw_graph)
        layout_left.addWidget(self.button_draw)

        self.button_clear = QPushButton("Reset Graph")
        self.button_clear.clicked.connect(self.clear_graph)
        layout_left.addWidget(self.button_clear)

        self.button_upload_sum = QPushButton("Add Summary")
        self.button_upload_sum.clicked.connect(self.upload_sum)
        layout_left.addWidget(self.button_upload_sum)

        self.labelDoc_sum = QLabel("Summary Text:")
        layout_left.addWidget(self.labelDoc_sum)

        self.DocSumArea = QTextEdit()
        layout_left.addWidget(self.DocSumArea)

        self.button_upload = QPushButton("Add Document")
        self.button_upload.clicked.connect(self.upload_document)
        layout_left.addWidget(self.button_upload)

        self.labelDoc = QLabel("Main Text:")
        layout_left.addWidget(self.labelDoc)

        self.DocArea = QTextEdit()
        layout_left.addWidget(self.DocArea)

        self.labelSum = QLabel("Summary Text:")
        layout_left.addWidget(self.labelSum)

        self.textarea = QTextEdit()
        layout_left.addWidget(self.textarea)

        self.labelRogue = QLabel("Rouge Score:")
        layout_left.addWidget(self.labelRogue)

        main_layout = QHBoxLayout()
        main_layout.addLayout(layout_left)
        main_layout.addWidget(graph_widget)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        self.ax.axis('off')
        self.G = None
        self.document = None

    def draw_graph(self):
        self.ax.clear()
        self.ax.axis('off')

        self.labelRogue.setText("Rouge Score:")

        threshold = float(self.input_similarity_threshold.text())
        score_threshold = float(self.input_score_threshold.text())
        print(threshold, score_threshold)

        paragraph = self.DocArea.toPlainText()
        paragraph_sum = self.DocSumArea.toPlainText()

        paragraph = paragraph.replace(".", " .")

        header = re.search(r'^(.*?)\n\n', paragraph, flags=re.DOTALL)
        header = header.group(1) if header else ""
        print("Title:", header)

        paragraph = re.sub(r'^(.*?)\n\n', '', paragraph, flags=re.DOTALL)
        print("Final Text:", paragraph)
        sentences = nltk.sent_tokenize(paragraph)
        sum_sent = ""

        threshold_counter = [0] * len(sentences)

        preprocessed_paragraph = functions.pre_process(paragraph)
        print(preprocessed_paragraph)
        similarities = functions.calculate_sim(preprocessed_paragraph)
        print(similarities)

        if self.G is None:
            self.G = nx.Graph()

            for i in range(len(similarities)):
                for j in range(i + 1, len(similarities)):
                    similarity = similarities[i][j]
                    print("Similarity between sentence", i + 1, "and sentence", j + 1, "is:", similarity.item())
                    if similarity > threshold:
                        threshold_counter[i] += 1
                        threshold_counter[j] += 1
            print("threshold_counter", threshold_counter)
            scores = functions.calc_score(preprocessed_paragraph, threshold_counter, sentences, header, paragraph)
            print("scores:", scores)

            for i, sentence in enumerate(sentences):
                self.G.add_node(i + 1, text=sentence, threshold_count=threshold_counter[i], score=scores[i])

            for i in range(len(similarities)):
                for j in range(i + 1, len(similarities)):
                    similarity = similarities[i][j]
                    self.G.add_edge(i + 1, j + 1, weight=similarity.item())

            sent_color = [0] * len(scores)
            for i, parts in enumerate(sentences):
                if scores[i] > score_threshold:
                    if threshold_counter[i] < math.ceil(int(len(sentences)/3)):
                        sum_sent += parts + " "
                        sent_color[i] = 1

                    elif threshold_counter[i] > math.ceil(int(len(sentences)/2)):
                        sum_sent += parts + " "
                        sent_color[i] = 1

                    elif scores[i] in sorted(scores, reverse=True)[:1]:
                        sum_sent += parts + " "
                        sent_color[i] = 1

                elif scores[i] in sorted(scores, reverse=True)[:math.ceil(len(scores) / 5)]:
                    sum_sent += parts + " "
                    sent_color[i] = 1

            print("sum text:", sum_sent)

            if sum_sent != "" and paragraph_sum != "":
                rouge = Rouge()
                rogue_scores = rouge.get_scores(sum_sent, paragraph_sum)
                rouge_1_f_score = rogue_scores[0]['rouge-1']['f']
                print("rouge score: ", rouge_1_f_score)
            else:
                rouge_1_f_score = 0
                print("*rouge score: ", rouge_1_f_score)

            self.labelRogue.setText("Rouge Score: " + str(rouge_1_f_score))
            self.textarea.setPlainText(header + "\n\n" + sum_sent)

            pos = nx.spring_layout(self.G)
            edge_colors = ['red' if self.G[u][v]['weight'] > threshold else 'black' for u, v in self.G.edges()]
            node_colors = ['yellow' if val == 1 else '#1f78b4' for val in sent_color]

            node_labels = {
                n: f"Sentence {n}\nConnection Count:{self.G.nodes[n]['threshold_count']}\nSkor: {self.G.nodes[n]['score']:.2f}"
                for n in self.G.nodes()}

            nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, node_size=2000)
            nx.draw_networkx_labels(self.G, pos, labels=node_labels, font_size=10, font_family="sans-serif")
            nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, width=3, alpha=0.5)

            edge_labels = nx.get_edge_attributes(self.G, 'weight')
            edge_labels = {(u, v): f'{weight:.2f}' for (u, v), weight in edge_labels.items()}

            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

        self.canvas.draw()

    def clear_graph(self):

        self.ax.clear()
        self.ax.axis('off')
        self.G = None
        self.canvas.draw()

    def print_values(self):

        threshold = self.input_similarity_threshold.text()
        score_threshold = self.input_score_threshold.text()

        print("Sent. Corr. Threshold Value:", threshold)
        print("Sent. Score Threshold Value:", score_threshold)

    def clear_values(self):

        self.input_similarity_threshold.clear()
        self.input_score_threshold.clear()

    def upload_document(self):

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose a Document", "", "All Files (*.*)", options=options)
        if file_name:
            file = QFile(file_name)
            if file.open(QFile.ReadOnly | QFile.Text):
                stream = QTextStream(file)
                self.document = stream.readAll()
                file.close()
                self.DocArea.setPlainText(self.document)
                print("Selected Document Content:")
                print(self.document)
            else:
                print("Document Could Not Be Opened:", file.errorString())

    def upload_sum(self):

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Choose a Document", "", "All Files (*.*)", options=options)
        if file_name:
            file = QFile(file_name)
            if file.open(QFile.ReadOnly | QFile.Text):
                stream = QTextStream(file)
                self.sum = stream.readAll()
                file.close()
                self.DocSumArea.setPlainText(self.sum)
                print("SUMMARY:")
                print(self.sum)
            else:
                print("Document Could Not Be Opened:", file.errorString())

    def closeEvent(self, event):

        self.clear_graph()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.setFixedSize(1440, 900)
    window.show()
    sys.exit(app.exec_())