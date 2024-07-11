import io
import re
import base64
import string
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, Response
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

pd.set_option("display.max_colwidth", None)

# initialize app
app = Flask(__name__)

# Define global variables for storing plot data
plot_data = None


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Gathering file from form
        uploaded_file = request.files['txt_file']

        # Making sure it's not empty
        if uploaded_file.filename != '':
            # Reading the file
            pdf_file = PdfReader(uploaded_file)
            num_pages = len(pdf_file.pages)

            # Extract text from each page
            text = ""
            for page_number in range(num_pages):
                page = pdf_file.pages[page_number]
                text += page.extract_text()

            # Converting to a string
            text = str(text)

            # Clean the text
            text = clean_text(text)

            # Perform clustering and generate the plot
            clustered_sentences, label_counts = perform_clustering(text)

            # Generate the plot
            plot_data = generate_plot(label_counts, len(clustered_sentences))

            # Return the index.html template with the plot data
            return render_template('index.html', PageTitle="Landing page", plot_data=plot_data)

    return render_template('index.html', PageTitle="Landing page")


def clean_text(text):
    """Clean the raw text."""
    # Remove non ASCII characters
    printables = set(string.printable)
    text = "".join(filter(lambda x: x in printables, text))

    # Replace tabs with spaces
    text = re.sub(r"\t+", r" ", text)

    # Aggregate lines where the sentence wraps
    # Also, lines in CAPITALS is counted as a header
    fragments = []
    prev = ""
    for line in re.split(r"\n+", text):
        if line.isupper():
            prev = "."  # skip it
        elif line and (line.startswith(" ") or line[0].islower() or not prev.endswith(".")):
            prev = f"{prev} {line}"  # make into one line
        else:
            fragments.append(prev)
            prev = line
    fragments.append(prev)

    # Clean the lines into sentences
    sentences = []
    for line in fragments:
        # Use regular expressions to clean text
        url_str = (
            r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\."
            r"([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
        )
        line = re.sub(url_str, r" ", line)  # URLs
        line = re.sub(r"^\s?\d+(.*)$", r"\1", line)  # headers
        line = re.sub(r"\d{5,}", r" ", line)  # figures
        line = re.sub(r"\.+", ".", line)  # multiple periods

        line = line.strip()  # leading & trailing spaces
        line = re.sub(r"\s+", " ", line)  # multiple spaces
        line = re.sub(r"\s?([,:;\.])", r"\1", line)  # punctuation spaces
        line = re.sub(r"\s?-\s?", "-", line)  # split-line words

        # Use nltk to split the line into sentences
        for sentence in nltk.sent_tokenize(line):
            s = str(sentence).strip().lower()  # lower case
            # Exclude tables of contents and short sentences
            if "table of contents" not in s and len(s) > 5:
                sentences.append(s)
    return sentences


class UnsupervisedClassifier:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer()
        self.kmeans = KMeans(n_clusters=self.n_clusters)

    def cluster_sentences(self, sentences):
        # Transform sentences into TF-IDF features
        X = self.vectorizer.fit_transform(sentences)

        # Apply K-means clustering
        self.kmeans.fit(X)

        # Assign cluster labels to sentences
        clustered_sentences = pd.DataFrame({"sentence": sentences, "cluster": self.kmeans.labels_})

        return clustered_sentences


def perform_clustering(text):
    """Perform clustering on the text and return the clustered sentences and label counts."""
    # Define the labels for the clusters
    labels = [
        "work Environment",
        "training and education",
        "Corporate Compliance",
        "environment friendly",
        "health and safety",
    ]

    # Instantiate the unsupervised classifier with the number of clusters
    classifier = UnsupervisedClassifier(n_clusters=len(labels))

    # Cluster the sentences
    clustered_sentences = classifier.cluster_sentences(text)

    # Assign labels to the clusters
    clustered_sentences["label"] = clustered_sentences["cluster"].map(lambda x: labels[x])

    label_counts = clustered_sentences["label"].value_counts()
    return clustered_sentences, label_counts


def generate_plot(label_counts, total_sentences):
    """Generate the bar chart plot and return it as a base64 encoded image."""
    # Create the bar chart
    plt.figure(figsize=(10, 12))
    ax = label_counts.plot(kind="bar")
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.title("Label Distribution")
    plt.xticks(rotation=45)

    # Set y-axis limits to 0-100
    plt.ylim(0, (total_sentences / 2))

    # Convert percentage values to custom text labels
    ytick_labels = ["Very Low", "Low", "Good", "Excellent"]
    ytick_positions = [
        total_sentences / 10,
        total_sentences / 5,
        3 * (total_sentences / 10),
        2 * (total_sentences / 5),
    ]
    plt.yticks(ytick_positions, ytick_labels)

    # Add label percentages to the bar chart for labels that exceed the threshold
    threshold = 0
    for i, count in enumerate(label_counts):
        if count >= threshold:
            percentage = count / total_sentences * 100
            ax.text(i, count, f"{percentage:.1f}%", ha="center", va="bottom")

    # Save the plot to a bytes buffer
    output = io.BytesIO()
    plt.savefig(output, format="png")
    plt.close()
    output.seek(0)

    # Convert the plot to a base64 encoded image
    plot_data = base64.b64encode(output.getvalue()).decode('utf-8')

    return plot_data


if __name__ == '__main__':
    app.run(debug=True)