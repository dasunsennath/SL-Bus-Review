from flask import Flask, request, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import pickle
from flask_cors import CORS

nltk.download('stopwords')

set(stopwords.words('english'))

app = Flask(__name__)
CORS(app)

@app.route('/')
def my_form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    vectorizer = pickle.load(open("models/emo_knn_vectorizer.pickle", 'rb'))
    model = pickle.load(open("models/emo_KNN_Distant_model.pickle", 'rb'))

    stop_words = stopwords.words('english')
    # convert to lowercase
    data = request.get_json(force=True)
    text1 = data['comment'].lower()
    text_final = ''.join(c for c in text1 if not c.isdigit())
    # remove stop words
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    sa = SentimentIntensityAnalyzer()
    sentiment_dict = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + sentiment_dict['compound']) / 2, 2)

    text = text_final
    color='black'
    positive_score = sentiment_dict['pos']
    negative_score = sentiment_dict['neg']
    neutral_score = sentiment_dict['neu']

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        emo_status = model.predict(vectorizer.transform([processed_doc1]))[0]
        sentiment_status = "Positive"
        color = 'green'

    elif sentiment_dict['compound'] <= - 0.05:
        emo_status = model.predict(vectorizer.transform([processed_doc1]))[0]
        sentiment_status = "Negative"
        color = 'red'


    else:
        sentiment_status = "Neutral"
        emo_status = "Neutral"
        color = 'black'

    send_to_frontend = [text, positive_score, negative_score, neutral_score, sentiment_status, emo_status]

    # text = typed text by user
    # sentiment_status = overall sentiment (it will give 2 outputs Positive and negative)
    # emo_status = text emotion status , (it will give emotion result)

    final_output = "The comment you raised is considered as a " + sentiment_status + " content. You have commented it in a " + emo_status + " mood.You can  proceed or update the content. "

    return {
        'text': text,
        'output': final_output,
        'color': color
    }

if __name__ == "__main__":
    app.run(debug=False)
