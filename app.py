from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room

import eventlet

import markdown

import openai

import os


app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key"
socketio = SocketIO(app, async_mode="eventlet")


from flask_sqlalchemy import SQLAlchemy

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chat.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(128), nullable=False)
    content = db.Column(db.String(1024), nullable=False)
    room = db.Column(db.String(128), nullable=False)

    def __init__(self, username, content, room):
        self.username = username
        self.content = content
        self.room = room


# Create the database and tables
with app.app_context():
    db.create_all()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat/<room>")
def chat(room):
    return render_template("chat.html", room=room)


@socketio.on("join")
def on_join(data):
    room = data["room"]
    join_room(room)

    # Fetch previous messages from the database
    previous_messages = Message.query.filter_by(room=room).all()

    for message in previous_messages:
        emit(
            "previous_messages",
            {"username": message.username, "message": message.content},
            room=request.sid,
        )

    emit("message", f"{data['username']} has joined the room.", room=room)


@socketio.on("message")
def handle_message(data):
    # Save the message to the database
    new_message = Message(
        username=data["username"], content=data["message"], room=data["room"]
    )
    db.session.add(new_message)
    db.session.commit()

    emit("message", f"{data['username']}: {data['message']}", room=data["room"])

    # Emit a temporary message indicating that GPT is processing
    emit("message", f"<span id='processing'>Processing...</span>", room=data["room"])

    # Call the chat_gpt function without blocking using eventlet.spawn
    eventlet.spawn(chat_gpt, data["username"], data["room"], data["message"])


def chat_gpt(username, room, message):

    with app.app_context():
        last_messages = (
            Message.query.filter_by(room=room)
            .order_by(Message.id.desc())
            .limit(10)
            .all()
        )

    # Format these messages as a chat history, with each message being a dict with 'role' and 'content'.
    chat_history = [
        {"role": "system" if msg.username == "GPT-3.5" else "user", "content": msg.content}
        for msg in reversed(last_messages)
    ]

    # Append the new message
    chat_history.append({"role": "user", "content": message})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=chat_history
    )

    # Extract response from ChatGPT API
    response_text = response["choices"][0]["message"]["content"]

    # Convert response_text to Markdown
    response_md = markdown.markdown(response_text, extensions=["fenced_code"])

    # Save ChatGPT's response in the database
    with app.app_context():
        chatgpt_response_message = Message(
            username="GPT-3.5", content=response_md, room=room
        )
        db.session.add(chatgpt_response_message)
        db.session.commit()

    socketio.emit("delete_processing_message", "", room=room)

    # Emit the response to the room
    socketio.emit("message", f"{username} (GPT-3.5): {response_md}", room=room)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001)
