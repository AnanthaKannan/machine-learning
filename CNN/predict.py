#!flask/bin/python
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras import backend as k



app = Flask(__name__)
CORS(app)
@app.route('/')
def index():
    return "Hello, World!"
    

    
@app.route('/predict', methods=['POST'])

def create_task():
#    image_path = 'img_dataset/prediction_img/pipeorhome17.jpg'
    k.clear_session()
    image_path =  request.files['file']
    Prediction_output = []
    test_image = image.load_img(image_path, 
                                target_size = (64, 64))
    damage_model = load_model('damageOrNot.h5')
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    damage_output = damage_model.predict(test_image)
    print(damage_output)

    
    if damage_output[0][0] == 1 :
        Prediction_output.append("Damage")
    elif damage_output[0][1] == 1:
         Prediction_output.append("Not Damage")
         
    return jsonify({"data" : {
                              "Product":Prediction_output[0]
                              }})

    
if __name__ == '__main__':
    app.run(debug=True, port=5090)
    