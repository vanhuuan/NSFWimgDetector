from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import os, io, sys
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)

@app.route('/maskImage', methods=['POST'])
def mask_image():
    # print(request.files , file=sys.stderr)
    file = request.files['image']
    img = Image.open(file)
    img.save("an.JPEG")
    return jsonify({'status': str(img)})


@app.route('/test', methods=['GET', 'POST'])
def test():
    print("log: got at test", file=sys.stderr)
    return jsonify({'status': 'succces'})


if __name__ == '__main__':
    app.run(debug=True)
