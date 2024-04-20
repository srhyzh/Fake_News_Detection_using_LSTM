# Fake_News_Detection_using_LSTM
 Using NLP / LSTM for Fake News Detection

### Summary of the Notebook: "DL_Fake News Detection LSTM"

This Jupyter notebook presents a comprehensive approach to detecting fake news using deep learning techniques, specifically employing Long Short-Term Memory (LSTM) networks. Key Python libraries utilized in this notebook include `tensorflow.keras` for building and training the LSTM model, `nltk` for natural language processing, `pandas` and `numpy` for data manipulation, and `matplotlib` and `seaborn` for visualization.

### Detailed Explanation

The notebook starts by importing necessary libraries and preparing the dataset which includes fake and real news articles. It involves preprocessing steps like tokenization, removal of stopwords, and stemming to clean and prepare the text data for modeling.

Key steps in the analysis include:
- **Data Visualization**: Using `matplotlib` and `seaborn` for visualizing data distributions and `WordCloud` to represent text data visually, helping understand the most frequent words in both fake and real news.
- **Data Preparation**: The text data is converted into sequences using `Tokenizer` from `tensorflow.keras.preprocessing.text`, and then padded to ensure uniform input size for the LSTM model.
- **Model Building**: A Sequential LSTM model is built using `tensorflow.keras`. The model includes layers like `Embedding`, `LSTM`, `Dense`, and `Dropout` to manage overfitting and learn from the data effectively.
- **Model Training and Evaluation**: The model is compiled and trained on the preprocessed text data, followed by evaluating its performance using metrics such as accuracy, confusion matrix, and classification report from `sklearn`.

The notebook emphasizes the power of LSTM in handling sequences and its effectiveness in tasks requiring understanding of long-term dependencies in text data. The model's training and validation results are visualized to assess the learning process, and predictions are made on test data to evaluate the model's practical performance in distinguishing between fake and real news.

Overall, the notebook is well-organized and structured to provide a clear workflow from data preprocessing to model evaluation, demonstrating an effective use of deep learning for the critical task of fake news detection.