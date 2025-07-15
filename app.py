import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # ✅ Use POST
def predict():
    try:
        # ✅ Get inputs from form (not args)
        nitrogen = float(request.form.get('nitrogen'))
        phosphorus = float(request.form.get('phosphorus'))
        potassium = float(request.form.get('potassium'))
        temperature = float(request.form.get('temperature'))
        ph = float(request.form.get('ph'))
        humidity = float(request.form.get('humidity'))
        rainfall = float(request.form.get('rainfall'))

        # Prepare input
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, ph, humidity, rainfall]])

        # Predict
        prediction = model.predict(input_data)

        # ✅ Return result
        return render_template('index.html', prediction_text=prediction[0])
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
