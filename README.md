Flask web application that performs clustering and generates a bar chart plot based on the uploaded PDF file. Here's a breakdown of the code:

The necessary imports are included at the beginning, such as io, re, string, nltk, matplotlib.pyplot, pandas, Flask, render_template, request, Response, PdfReader, TfidfVectorizer, and KMeans.

The Flask application is initialized with the name app.

The / route is defined using the @app.route decorator. It handles both GET and POST requests. Inside the route, it checks if the request method is POST.

If it's a POST request, the uploaded file is obtained from the form data, and if the file is not empty, the text is extracted from the PDF file using PdfReader. The extracted text is then cleaned using the clean_text function.

The perform_clustering function is called with the cleaned text to cluster the sentences and generate label counts.

The generate_plot function is called with the label counts and the total number of sentences to generate a bar chart plot. The plot is returned as a response.

If the uploaded file is empty or it's not a POST request, the render_template function is called to render the index.html template with the title "Landing page".

The clean_text function is defined to clean the raw text. It removes non-ASCII characters, replaces tabs with spaces, aggregates lines where the sentence wraps, cleans the lines into sentences using regular expressions, and tokenizes the sentences using nltk.

The UnsupervisedClassifier class is defined to perform clustering using TF-IDF features and K-means clustering.

The perform_clustering function is defined to perform clustering on the text using the UnsupervisedClassifier. It assigns labels to the clusters and returns the clustered sentences and label counts.

The generate_plot function is defined to generate a bar chart plot using Matplotlib. It sets up the plot, customizes the labels and title, adds text labels to the bars, and saves the plot to a bytes buffer. The plot is returned as a response.
