#!/usr/bin/env python3
from flask import Flask, render_template, request, send_from_directory
import json, os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    packet = {
        'example_service': "guojun.chen@yale.edu",
    }
    return json.dumps(packet)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=os.getenv("MAIN_PORT", 30000))