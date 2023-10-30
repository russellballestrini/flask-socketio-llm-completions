from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room

import eventlet

import openai

from openai.error import InvalidRequestError

import os

import time

import boto3
import json


app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key"
socketio = SocketIO(app, async_mode="eventlet")


from flask_sqlalchemy import SQLAlchemy

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chat.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# Create an argument parser for aws profile.
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--profile", help="AWS profile name")
args = parser.parse_args()
profile_name = args.profile


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

    # Send the history of messages only to the newly connected client.
    # The reason for using `request.sid` here is to target the specific session (or client) that
    # just connected, so only they receive the backlog of messages, rather than broadcasting
    # this information to all clients in the room.
    for message in previous_messages:
        emit(
            "previous_messages",
            {
                "id": message.id,
                "username": message.username,
                "message": message.content,
            },
            room=request.sid,
        )

    # Broadcast to all clients in the room that a new user has joined.
    # Here, `room=room` ensures the message is sent to everyone in that specific room.
    emit(
        "message",
        {"id": None, "content": f"{data['username']} has joined the room."},
        room=room,
    )


@socketio.on("message")
def handle_message(data):
    # Save the message to the database
    new_message = Message(
        username=data["username"], content=data["message"], room=data["room"]
    )
    db.session.add(new_message)
    db.session.commit()

    emit(
        "message",
        {"id": new_message.id, "content": f"{data['username']}: {data['message']}"},
        room=data["room"],
    )

    if "claude" in data["message"] or "gpt" in data["message"]:
        # Emit a temporary message indicating that llm is processing
        emit(
            "message",
            {"id": None, "content": f"<span id='processing'>Processing...</span>"},
            room=data["room"],
        )

        if "claude" in data["message"]:
            eventlet.spawn(chat_claude, data["username"], data["room"], data["message"])
        if "gpt" in data["message"]:
            eventlet.spawn(chat_gpt, data["username"], data["room"], data["message"])


@socketio.on("delete_message")
def handle_delete_message(data):
    msg_id = data["message_id"]
    # Delete the message from the database
    message = db.session.query(Message).filter(Message.id == msg_id).one_or_none()
    if message:
        db.session.delete(message)
        db.session.commit()

    # Notify all clients in the room to remove the message from their DOM
    emit("message_deleted", {"message_id": msg_id}, room=data["room"])


def chat_claude(username, room, message):
    with app.app_context():
        # claude has a 100,000 token context window for prompts.
        all_messages = (
            Message.query.filter_by(room=room).order_by(Message.id.desc()).all()
        )

    chat_history = ""

    for msg in reversed(all_messages):
        if msg.username in ["gpt-3.5-turbo", "anthropic.claude-v2"]:
            chat_history += f"Assistant: {msg.username}: {msg.content}\n\n"
        else:
            chat_history += f"Human: {msg.username}: {msg.content}\n\n"

    # append the new message.
    chat_history += f"Human: {username}: {message}\n\nAssistant:"

    # Initialize the Bedrock client using boto3 and profile name.
    if profile_name:
        session = boto3.Session(profile_name=profile_name)
        client = session.client("bedrock-runtime", region_name="us-east-1")
    else:
        client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Define the request parameters
    params = {
        "modelId": "anthropic.claude-v2",
        "contentType": "application/json",
        "accept": "*/*",
        "body": json.dumps(
            {
                "prompt": chat_history,
                "max_tokens_to_sample": 2048,
                "temperature": 0,
                "top_k": 250,
                "top_p": 0.999,
                "stop_sequences": ["\n\nHuman:"],
                "anthropic_version": "bedrock-2023-05-31",
            }
        ).encode(),
    }

    # Invoke the model with response stream
    response = client.invoke_model_with_response_stream(**params)

    # Process the event stream
    buffer = ""

    # save empty message, we need the ID when we chunk the response.
    with app.app_context():
        new_message = Message(username="anthropic.claude-v2", content=buffer, room=room)
        db.session.add(new_message)
        db.session.commit()
        msg_id = new_message.id

    first_chunk = True
    for event in response["body"]:
        content = ""

        if "chunk" in event:
            chunk_data = json.loads(event["chunk"]["bytes"].decode())
            content = chunk_data["completion"]

        if content:
            buffer += content  # Accumulate content

            if first_chunk:
                socketio.emit(
                    "message_chunk",
                    {
                        "id": msg_id,
                        "content": f"{username} (anthropic.claude-v2): {content}",
                    },
                    room=room,
                )
                first_chunk = False
            else:
                socketio.emit(
                    "message_chunk",
                    {"id": msg_id, "content": content},
                    room=room,
                )
            socketio.sleep(0)  # Force immediate handling

    # Save the entire completion to the database
    with app.app_context():
        new_message = (
            db.session.query(Message).filter(Message.id == msg_id).one_or_none()
        )
        if new_message:
            new_message.content = buffer
            db.session.add(new_message)
            db.session.commit()

    socketio.emit("delete_processing_message", msg_id, room=room)


def chat_gpt(username, room, message):
    with app.app_context():
        last_messages = (
            Message.query.filter_by(room=room)
            .order_by(Message.id.desc())
            .limit(10)
            .all()
        )

    chat_history = [
        {
            "role": "system"
            if (
                msg.username == "gpt-3.5-turbo" or msg.username == "anthropic.claude-v2"
            )
            else "user",
            "content": f"{msg.username}: {msg.content}",
        }
        for msg in reversed(last_messages)
    ]

    chat_history.append({"role": "user", "content": message})

    buffer = ""  # Content buffer for accumulating the chunks

    # save empty message, we need the ID when we chunk the response.
    with app.app_context():
        new_message = Message(username="gpt-3.5-turbo", content=buffer, room=room)
        db.session.add(new_message)
        db.session.commit()
        msg_id = new_message.id

    first_chunk = True
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0,
        stream=True,
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")

        if content:
            buffer += content  # Accumulate content

            if first_chunk:
                socketio.emit(
                    "message_chunk",
                    {
                        "id": msg_id,
                        "content": f"{username} (gpt-3.5-turbo): {content}",
                    },
                    room=room,
                )
                first_chunk = False
            else:
                socketio.emit(
                    "message_chunk",
                    {"id": msg_id, "content": content},
                    room=room,
                )
            socketio.sleep(0)  # Force immediate handling

    # Save the entire completion to the database
    with app.app_context():
        new_message = (
            db.session.query(Message).filter(Message.id == msg_id).one_or_none()
        )
        if new_message:
            new_message.content = buffer
            db.session.add(new_message)
            db.session.commit()

    socketio.emit("delete_processing_message", msg_id, room=room)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001)
