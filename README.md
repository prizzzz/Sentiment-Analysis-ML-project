# ✈️ Airline Tweet Sentiment Analysis using LSTM (Keras)

This project is a sentiment classification model that predicts whether a tweet about an airline is **positive** or **negative** using deep learning. It leverages **LSTM networks** and **Keras** for sequence modeling on real-world tweet data.

---

## 📁 Dataset

- Source: [`Tweets.csv`](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- Columns used:
  - `text`: The actual tweet
  - `airline_sentiment`: The sentiment label (`positive`, `neutral`, `negative`)

➡️ *Neutral tweets are removed to simplify the task to binary classification.*

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- TensorFlow / Keras
- LSTM Neural Networks
- Tokenizer & Padding
- Matplotlib for Visualization
- Google Colab
- GitHub
- Pickle (for saving tokenizer)

---

## 📊 Model Architecture

- **Embedding Layer**: Converts words to dense vectors
- **SpatialDropout1D**: Adds regularization
- **LSTM Layer**: Learns sequence dependencies
- **Dense Layer**: Sigmoid activation for binary output

```python
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

✅ Results
Model trained with binary_crossentropy loss and Adam optimizer.

Plots generated:
📈 Accuracy plot.jpg: Training vs. Validation Accuracy
📉 Loss plt.jpg: Training vs. Validation Loss

🔍 How to Predict Sentiment
Use the saved tokenizer and model to predict sentiment for any new tweet.

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])

🔄 Example:
predict_sentiment("Awesome experience! Loved it.")

💾 Saved Files
sentiment_model.keras – Trained LSTM model
tokenizer.pkl – Tokenizer used for encoding
Accuracy plot.jpg, Loss plt.jpg – Training performance graphs

🚀 How to Run This Project
Upload the dataset Tweets.csv to your Google Drive.
Open the Colab notebook and mount the drive.
Train the model using the notebook code.
Use predict_sentiment() function to test on custom tweets.

🧠 Future Enhancements
Add multi-class classification support (positive, neutral, negative)
Implement BERT or other transformer-based models
Deploy using Flask, Django, or Streamlit

👩‍💻 Author
Priyanka Manohar Chougule
Machine Learning & AI Enthusiast

📜 License
This project is for educational and academic purposes only.
