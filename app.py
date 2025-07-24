# app.py

from flask import Flask, render_template, jsonify, request
from main_code import *

app = Flask(__name__)

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/toggle_threads', methods=['POST'])
def toggle_threads():
    """Handles starting and stopping of the background threads."""
    data = request.get_json()
    if data.get('command') == 'start':
        start_threads()
        return jsonify(status='Threads are ON')
    elif data.get('command') == 'stop':
        stop_threads()
        return jsonify(status='Threads are OFF')
    return jsonify(status='Invalid command'), 400

@app.route('/status')
def status():
    """Returns the current status of the threads."""
    print(global_do_loop, global_id_info_loop, do_parse, abort)
    if global_do_loop and global_id_info_loop and do_parse and not abort:
        return jsonify(status='ON')
    return jsonify(status='OFF')


@app.route('/predict')
def predict():
    """Triggers the prediction and returns the results as JSON."""
    predictions = run_predictions()
    return jsonify(predictions)


if __name__ == '__main__':
    app.run(debug=True)