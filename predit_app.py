import numpy as np
from keras.preprocessing import image
import io
import base64
from flask import request
from flask import jsonify
from flask import Flask
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
graph = tf.get_default_graph()
sess = tf.Session()
app=Flask(__name__)
def get_model():
    global model
    with open('CNN2','rb') as f:
        model=pickle.load(f)
def preprocess_image(image1,targetsize):
    if image1.mode != 'RGB':
        image1=image1.convert("RGB")
    image1=image1.resize(targetsize)
    image1=image.img_to_array(image1)
    image1=np.expand_dims(image1,axis=0)
    return image1
print("loading the model")
set_session(sess)
get_model()
@app.route('/',methods=['POST'])
def predict():
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    image1=Image.open(io.BytesIO(decoded))
    processed_image=preprocess_image(image1,targetsize=(64,64))
    with graph.as_default():
        global result
        set_session(sess)
        result=model.predict(processed_image)
        print(result)
        
    x=''
    if result[0][0]==1:
        x='dog'
    else:
        x='cat'
    print(x)
    response={'prediction': x}
    return jsonify(response)
    





if __name__=='__main__':
    app.run(debug=True)
    

