from flask import Flask, Response, stream_with_context
import json
from queue import Queue
import threading

app = Flask(__name__)
data_queue = Queue()

@app.route('/watch-data', methods=['POST'])
def receive_watch_data():
    """Endpoint for Apple Watch to send data"""
    data = request.json
    data_queue.put(data)
    return {"status": "ok"}

@app.route('/stream')
def stream():
    """SSE stream endpoint for Streamlit"""
    def generate():
        while True:
            data = data_queue.get()
            yield f"data: {json.dumps(data)}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8765)