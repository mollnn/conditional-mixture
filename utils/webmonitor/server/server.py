from flask import Flask, render_template
from gevent import pywsgi

app = Flask(__name__)

@app.route('/monitor') 
def index():
    return render_template('index.html')

if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0',5000), app)
    server.serve_forever()