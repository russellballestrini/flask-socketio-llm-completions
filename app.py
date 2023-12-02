from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit, join_room

import eventlet

from openai import OpenAI

openai_client = OpenAI()

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
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--profile", help="AWS profile name")
# args = parser.parse_args()
# profile_name = args.profile
profile_name = None


class Room(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False, unique=True)
    title = db.Column(
        db.String(128), nullable=True
    )  # Initially, there might be no title


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(128), nullable=False)
    content = db.Column(db.String(1024), nullable=False)
    room_id = db.Column(db.Integer, db.ForeignKey("room.id"), nullable=False)

    def __init__(self, username, content, room_id):
        self.username = username
        self.content = content
        self.room_id = room_id


def get_room(room_name):
    """Utility function to get room from room name."""
    room = Room.query.filter_by(name=room_name).first()
    if room:
        return room
    else:
        # Create a new room since it doesn't exist
        new_room = Room(name=room_name)
        db.session.add(new_room)
        db.session.commit()
        return new_room


# Create the database and tables
# with app.app_context():
#    db.create_all()

from flask_migrate import Migrate

migrate = Migrate(app, db)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, "static"), "favicon.ico")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat/<room_name>")
def chat(room_name):
    # Query all rooms so that newest is first.
    rooms = Room.query.order_by(Room.id.desc()).all()

    # Get username from query parameters
    username = request.args.get("username", "guest")

    # Pass username and rooms into the template
    return render_template(
        "chat.html", room_name=room_name, rooms=rooms, username=username
    )


@socketio.on("join")
def on_join(data):
    room_name = data["room_name"]
    room = get_room(room_name)

    # this makes the client start listening for new events for this room.
    join_room(room_name)

    # update the title bar with the proper room title, if it exists for just this new client.
    if room.title:
        socketio.emit("update_room_title", {"title": room.title}, room=request.sid)

    # Fetch previous messages from the database
    previous_messages = Message.query.filter_by(room_id=room.id).all()

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

    message_count = len(previous_messages)
    if room.title is None and message_count >= 6:
        room.title = gpt_generate_room_title(previous_messages, "gpt-4-1106-preview")
        db.session.add(room)
        db.session.commit()
        socketio.emit("update_room_title", {"title": room.title}, room=room.name)

    # Broadcast to all clients in the room that a new user has joined.
    # Here, `room=room` ensures the message is sent to everyone in that specific room.
    emit(
        "message",
        {"id": None, "content": f"{data['username']} has joined the room."},
        room=room.name,
    )


def load_s3_file(room_name, s3_file_path, username):
    # Initialize the S3 client
    s3_client = boto3.client("s3")

    # Assuming the bucket name is set in an environment variable
    bucket_name = os.environ.get("S3_BUCKET_NAME")

    try:
        # Retrieve the file content from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_file_path)
        file_content = response["Body"].read().decode("utf-8")

        # Format the file content as a code block
        formatted_content = f"```\n{file_content}\n```"

        # Save the message to the database
        with app.app_context():
            room = get_room(room_name)
            new_message = Message(
                username=username,  # Use the username who issued the command
                content=formatted_content,
                room_id=room.id,
            )
            db.session.add(new_message)
            db.session.commit()

            # Emit the file content as a message to the chatroom with the message ID
            socketio.emit(
                "message",
                {
                    "id": new_message.id,
                    "content": formatted_content,
                },
                room=room_name,
            )

    except Exception as e:
        # Handle errors (e.g., file not found, access denied)
        error_message = f"Error loading file from S3: {e}"

        # Save the error message to the database
        with app.app_context():
            room = get_room(room_name)
            new_error_message = Message(
                username=username,
                content=error_message,
                room_id=room.id,
            )
            db.session.add(new_error_message)
            db.session.commit()

            # Emit the error message to the chatroom without a message ID
            socketio.emit(
                "message",
                {
                    "id": new_error_message.id,
                    "content": error_message,
                },
                room=room_name,
            )


@socketio.on("message")
def handle_message(data):
    room_name = data["room_name"]
    room = get_room(room_name)

    # Save the message to the database
    new_message = Message(
        username=data["username"],
        content=data["message"],
        room_id=room.id,
    )
    db.session.add(new_message)
    db.session.commit()

    emit(
        "message",
        {
            "id": new_message.id,
            "content": f"**{data['username']}:**\n\n{data['message']}",
        },
        room=room.name,
    )

    # detect and process special commands.
    commands = data["message"].splitlines()

    for command in commands:
        if command.startswith("/s3 load"):
            # Extract the S3 file path
            s3_file_path = command.split(" ", 2)[2]
            # Load the file from S3 and emit its content
            eventlet.spawn(load_s3_file, room_name, s3_file_path, data["username"])

    if (
        "claude-v1" in data["message"]
        or "claude-v2" in data["message"]
        or "gpt-3" in data["message"]
        or "gpt-4" in data["message"]
    ):
        # Emit a temporary message indicating that llm is processing
        emit(
            "message",
            {"id": None, "content": f"<span id='processing'>Processing...</span>"},
            room=room.name,
        )

        if "claude-v1" in data["message"]:
            eventlet.spawn(chat_claude, data["username"], room.name, data["message"])
        if "claude-v2" in data["message"]:
            eventlet.spawn(
                chat_claude,
                data["username"],
                room.name,
                data["message"],
                model_name="anthropic.claude-v2",
            )
        if "gpt-3" in data["message"]:
            eventlet.spawn(chat_gpt, data["username"], room.name, data["message"])
        if "gpt-4" in data["message"]:
            eventlet.spawn(
                chat_gpt,
                data["username"],
                room.name,
                data["message"],
                model_name="gpt-4-1106-preview",
            )


@socketio.on("delete_message")
def handle_delete_message(data):
    msg_id = data["message_id"]
    # Delete the message from the database
    message = db.session.query(Message).filter(Message.id == msg_id).one_or_none()
    if message:
        db.session.delete(message)
        db.session.commit()

    # Notify all clients in the room to remove the message from their DOM
    emit("message_deleted", {"message_id": msg_id}, room=data["room_name"])


def chat_claude(username, room_name, message, model_name="anthropic.claude-v1"):
    with app.app_context():
        room = get_room(room_name)
        # claude has a 100,000 token context window for prompts.
        all_messages = (
            Message.query.filter_by(room_id=room.id).order_by(Message.id.desc()).all()
        )

    chat_history = ""

    for msg in reversed(all_messages):
        if msg.username in [
            "gpt-3.5-turbo",
            "anthropic.claude-v1",
            "anthropic.claude-v2",
            "gpt-4",
            "gpt-4-1106-preview",
        ]:
            chat_history += f"Assistant: {msg.username}: {msg.content}\n\n"
        else:
            chat_history += f"Human: {msg.username}: {msg.content}\n\n"

    # append the new message.
    chat_history += f"Human: {username}: {message}\n\nAssistant: {model_name}: "

    # Initialize the Bedrock client using boto3 and profile name.
    if profile_name:
        session = boto3.Session(profile_name=profile_name)
        client = session.client("bedrock-runtime", region_name="us-east-1")
    else:
        client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Define the request parameters
    params = {
        "modelId": model_name,
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
        new_message = Message(username=model_name, content=buffer, room_id=room.id)
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
                        "content": f"**{username} ({model_name}):**\n\n{content}",
                    },
                    room=room.name,
                )
                first_chunk = False
            else:
                socketio.emit(
                    "message_chunk",
                    {"id": msg_id, "content": content},
                    room=room.name,
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

    socketio.emit("delete_processing_message", msg_id, room=room.name)


def chat_gpt(username, room_name, message, model_name="gpt-3.5-turbo"):
    limit = 15
    if model_name == "gpt-4-1106-preview":
        limit = 1000

    with app.app_context():
        room = get_room(room_name)
        last_messages = (
            Message.query.filter_by(room_id=room.id)
            .order_by(Message.id.desc())
            .limit(limit)
            .all()
        )
        if room.title is None and len(last_messages) >= 5:
            room.title = gpt_generate_room_title(last_messages, model_name)
            db.session.add(room)
            db.session.commit()
            socketio.emit("update_room_title", {"title": room.title}, room=room.name)

        chat_history = [
            {
                "role": "system"
                if (
                    msg.username == "gpt-3.5-turbo"
                    or msg.username == "anthropic.claude-v1"
                    or msg.username == "anthropic.claude-v2"
                    or msg.username == "gpt-4"
                    or msg.username == "gpt-4-1106-preview"
                )
                else "user",
                "content": f"{msg.username}: {msg.content}",
            }
            for msg in reversed(last_messages)
        ]

        chat_history.append({"role": "user", "content": f"{message}\n\n{model_name}: "})

    buffer = ""  # Content buffer for accumulating the chunks

    # save empty message, we need the ID when we chunk the response.
    with app.app_context():
        new_message = Message(username=model_name, content=buffer, room_id=room.id)
        db.session.add(new_message)
        db.session.commit()
        msg_id = new_message.id

    first_chunk = True
    for chunk in openai_client.chat.completions.create(
        model=model_name, messages=chat_history, temperature=0, stream=True
    ):
        content = chunk.choices[0].delta.content

        if content:
            buffer += content  # Accumulate content

            if first_chunk:
                socketio.emit(
                    "message_chunk",
                    {
                        "id": msg_id,
                        "content": f"**{username} ({model_name}):**\n\n{content}",
                    },
                    room=room.name,
                )
                first_chunk = False
            else:
                socketio.emit(
                    "message_chunk",
                    {"id": msg_id, "content": content},
                    room=room.name,
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

    socketio.emit("delete_processing_message", msg_id, room=room.name)


def gpt_generate_room_title(messages, model_name):
    """
    Generate a title for the room based on a list of messages.
    """

    chat_history = [
        {
            "role": "system"
            if (
                msg.username == "gpt-3.5-turbo"
                or msg.username == "anthropic.claude-v1"
                or msg.username == "anthropic.claude-v2"
                or msg.username == "gpt-4"
                or msg.username == "gpt-4-1106-preview"
            )
            else "user",
            "content": f"{msg.username}: {msg.content}",
        }
        for msg in reversed(messages)
    ]

    chat_history.append(
        {
            "role": "system",
            "content": "return a short title for the title bar of this conversation.",
        }
    )

    # Interaction with LLM to generate summary
    # For example, using OpenAI's GPT model
    response = openai_client.chat.completions.create(
        messages=chat_history,
        model=model_name,  # or any appropriate model
        max_tokens=20,
    )

    title = response.choices[0].message.content
    return title.replace('"', "")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001)
