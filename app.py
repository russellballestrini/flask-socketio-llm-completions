from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room

import eventlet

import openai

import markdown

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, async_mode='eventlet')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat/<room>')
def chat(room):
    return render_template('chat.html', room=room)

@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)
    emit('message', f"{data['username']} has joined the room.", room=room)

@socketio.on('message')
def handle_message(data):
    emit('message', f"{data['username']}: {data['message']}", room=data['room'])

    openai.api_key = "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"

    # Call the chat_gpt function without blocking using eventlet.spawn
    eventlet.spawn(chat_gpt, data['username'], data['room'], data['message'])


def chat_gpt(username, room, message):
    # Send user's message to ChatGPT API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )

    # Extract response from ChatGPT API
    response_text = response['choices'][0]['message']['content']

    # Convert response_text to Markdown
    response_md = markdown.markdown(response_text, extensions=['fenced_code'])

    # Emit the response to the room
    socketio.emit('message', f"{username} (GPT-3.5): {response_md}", room=room)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)

