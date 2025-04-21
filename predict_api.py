import joblib
from flask import Flask, request, jsonify

# Load the model and vectorizer
model = joblib.load('LogisticRegression_All_shots_data_model.pkl')
vectorizer = joblib.load('LogisticRegression_All_shots_data_vectorizer.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.json
        sentence = json_data['sentence']
        # Transform the sentence using the loaded vectorizer
        vect_sentence = vectorizer.transform([sentence])
        # Predict using the loaded model
        prediction = model.predict(vect_sentence)
        result = 'Tell' if prediction[0] == 0 else 'Show'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
