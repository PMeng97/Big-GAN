import uuid
import io
import flask
from flask import jsonify
import os
import sys
import json
from base64 import encodebytes
import torch
from predict import txt2img
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

@app.route('/')
def ping_server():
    return "Welcome to the world of vivaaaaaaabb."


@app.route("/txt2img/<prompt>", methods=['GET'])
def txt2img_generation(prompt):
    # Action needed to store records in MongoDB
    print('!!txt2img_generation: Request received')
    imgs = txt2img(prompt)
    if imgs == 'wrong class':
        return jsonify({'result': imgs})
    response_imgs = []
    for i in imgs:
        buf = io.BytesIO()
        i.save(buf, format='PNG')
        img = buf.getvalue()
        encoded_img = encodebytes(buf.getvalue()).decode('ascii')
        response_imgs.append(encoded_img)
    print('!!txt2img_generation: Generation finished')
#    return flask.Response(img, mimetype='image/png')
    return jsonify({'result': response_imgs})

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    # load_model()
    app.run(host='0.0.0.0')