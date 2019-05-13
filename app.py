#!/usr/bin/env python3
from flask import Flask, request,render_template
import json
import numpy as np
import urllib.request as url_read
import cv2

import Services

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/testAPI', methods=['POST'])
def result():
    e1 = cv2.getTickCount()
    try:
        img_url = request.form.get('url')
        img = url_read.urlopen(img_url)
        # img = np.asarray(bytearray(resp.read()), dtype="uint8")
        net, classes = Services.loadNetwork()
        results = Services.predict(img, net, classes )
        return json.dumps(results)
    except:
        return 'fail'


@app.route('/isAlive', methods=['POST', 'GET'])
def isAlive():
    return "Yes I am alive and Happy :)))";


if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0')