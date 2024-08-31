from flask import Flask,render_template, request, jsonify
import pickle
import os
import numpy as np

path = os.path.join(os.path.dirname(__file__), 'house_tax_model.pkl')
scaler = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
with open(path, 'rb') as file:
    model = pickle.load(file)
with open(scaler, 'rb') as file:
    scale = pickle.load(file)
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/main')
def MainPage():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #perdicts tax per 10,000 dollars
    data = request.get_json(force=True)
    try:
        
        features = [float(data['crim']), float(data['zn']), float(data['indus']), float(data['chas']), 
                    float(data['nox']), float(data['rm']), float(data['age']), float(data['dis']), 
                    float(data['rad']), float(data['ptratio']), float(data['b']), 
                    float(data['lstat']),float(data['medv'])]
#         crim	zn	indus	chas	nox	rm	age	dis	rad	tax	ptratio	b	lstat	medv
# 0	0.00632	18.0	2.31	0	0.538	6.575	65.2	4.0900	1	296	15.3	396.90	4.98	24.0
        # Convert features to a numpy array and reshape it to an array with one row and as many columns as possible
        features = np.array(features).reshape(1, -1)
        scaled_features = scale.transform(features)
        prediction = model.predict(scaled_features)
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': e})
if __name__ == '__main__':
    app.run(debug=True)