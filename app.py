from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import os, io, sys
import numpy as np
import cv2
import base64
from firebase_admin import credentials, initialize_app, storage
import jwt

app = Flask(__name__)
CORS(app)
credent = {
  "type": "service_account",
  "project_id": "bosha-4df95",
  "private_key_id": "fca74ee4109f4eb5299fdf425cd1692df5668000",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC62WPsEJS+ZWI9\nskgF8HH3kTZ7/XlD+yf6DQY5YCaUDqcWfg0RKvnUYj8nmvgFojsaJ0wHbfEmB7ms\nCfbhZod5dkfvlQEbJagPGCVcSh/+NuPsI0DXS3Qkqj7erlS9Ync+y78ZROwQ6DKq\nBVoa8P2TS4kG3hnhZ17g1PB3fP+HKMnh5ETe1t2b2IduL8rvFVw0qrd4vrB3HYc/\nSEJDeB+brQaXqecVNTiz3C08pWeZvyqsUWgnxWOESw0QdDtdrK6nUssn4kdVyHGY\n2Q4cuwWDmLCLrorujrrKZuUlNHsUmngNn7D5/IKhWFv++ss7jpHjJpXCjTgJPHlh\nGjqvZKenAgMBAAECggEAKxom0LfqeBlPFkhgEO6se52F4ghQxvvS9v22J9okTBct\nOCUPwpwVUYbwZqR6/mG0HI5gHX6hGFLD5AnQZ6KNMdm1MnE+9hIzuHerdqSqyKvE\n3B4twPxo6GRVaGzyo1VpiJTbmmqpwiSuO0ZMyNY6yJWuIGVvtu1vl5HHfujY0UaU\nj9dY1vCq7BEY6dSJLe0OKlhFg9VnjGg2x02PR/Xxjz2E+Z+rWFBbi9j8Aa3UhMtn\nm61empfusOAsKuVXgRPkoHfemqV4WspcPZDY3mm8D8Shkjx9cWJuySTq7a4X6Uu7\nizJtkejG4oGiH9SvrSSZv1SOQ8mqRfw7SVm34+gk4QKBgQDeq029y2wr9NaYBaLI\nXouNXS6xrp7VFbHcqueFZckcRrH1jb2E/bRWOsIpTY93+btMw0DALlL+Uee4Hb9V\nzXFbtH9gtiYhFqbCxueEdIk+6mC4/VrFqxDbBoTpsANAzguf1ewRxmVN5einDVoU\nrQsSbAEDYFynDMv70ewK/qmoeQKBgQDW0XaDCnNMaWc/luyID2ZZB2YUSGYKJ2kb\nMf0gEfy+5pcWeNOcLIGHzs9Os2o6OSmfIv6rBeaa1A3VhcyHcK2P5yIjYfxPbKL4\nJKl2KYWfSZX1ebOEi8y2n8Bz6YqtVPUbxC6dD+xEJ1yHB4ak2OF2qUPBC904BgyP\nwL6bkBUJHwKBgGJ58E/X/2cRSTbCh1OGPMGwTE5Yo/YpOhgVnX7pnR/cRCgNH5wJ\neBAKvIP3wVOWJre0Qpa75giDwK80uMIBsfiUwyDKbeKEWgwaW5nA0FH65fvqUJLL\nLREwuOeA6G9osMtnmJfj3jmtTE9VIFm0jgj/qxMLMu7OQgiEtpfGONJBAoGAY8X/\n3FczPA9MpC1OCNfWOo2ELPF/a0HowHfUUFvqXyQskoRmDIhjVReKE8yjDw3TthUD\n4Z0pAUpybnTpjRKZmlfI9FQjYyRbARPOHZKlfFkqLExtQ6fG2h+SvIOBJIZsHYxd\n9hcaLoF1qUWmlNQizAFL9dKz2bx9gxsLrKccDhMCgYEAgnelUCncyp2ph417+JDK\nMUyOC5812bQFQnxKIQHGxCdRMRf1ZTAGIgLOp0Ofkn1y3i2IBH5ySx9VizBujtOY\nA0U/Br+UY+iB1Vyt6TRNT29yn7z2b9Jp6Dv6j3AAAhHORXMYH/RrTD6GVNCzWMdj\nEJtvlWS3OTExkANtf9SdJjo=\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-9pn2w@bosha-4df95.iam.gserviceaccount.com",
  "client_id": "116342376852829120203",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-9pn2w%40bosha-4df95.iam.gserviceaccount.com"
}
cred = credentials.Certificate(credent)
firebase = initialize_app(cred, {'storageBucket': 'bosha-4df95.appspot.com'})
secret = "PDv7DrqznYL6nv7DrqzjnQYO9JxIsWdcjnQYL6nu0f"


def decode_user(token: str):
    decoded_data = jwt.decode(jwt=token, bkey=secret, algorithms=["HS256"], verify=False)
    return decoded_data


@app.route('/maskImage', methods=['POST'])
def mask_image():
    token = request.headers['Authorization']
    print(token)
    claims = decode_user(token)
    print(claims)
    file = request.files['image']
    bucket = storage.bucket()
    blob = bucket.blob("images/newimage.png")
    blob.upload_from_file(file)
    return jsonify({'status': "Okeeee"})


@app.route('/test', methods=['GET', 'POST'])
def test():
    print("log: got at test", file=sys.stderr)
    return jsonify({'status': 'succces'})


if __name__ == '__main__':
    app.run(debug=True)
