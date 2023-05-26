from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import sys
import numpy as np
from keras.models import load_model

app = Flask(__name__)
CORS(app)
model = load_model('model.h5')


@app.route('/checkImg', methods=['POST'])
def mask_image():
    try:
        imgfile = request.files['image']
        img = Image.open(imgfile)
        img = img.convert('RGB')
        img_size = (255, 255)
        resize = img.resize(img_size)
        arr = np.array(resize, dtype=np.uint8)
        arr = np.asarray([arr], dtype=np.uint8)
        predict = model.predict(arr)
        nsfw = np.argmax(predict, axis=1)
        if nsfw == 1:
            return jsonify({'status': "Oke"})
        else:
            return jsonify({'status': "NotOke"})
    except Exception as err:
        print(err)
        return jsonify({'status': "NotOke"})


@app.route('/healthcheck', methods=['GET'])
def test():
    print("log: got at test", file=sys.stderr)
    return jsonify({'status': 'success'})


@app.route('/', methods=['GET'])
def index():
    print("Okeeeeeeeeeee")
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
