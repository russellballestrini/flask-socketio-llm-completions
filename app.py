# import eventlet
# eventlet.monkey_patch()

import gevent
from gevent import monkey

monkey.patch_all()


import json
import yaml
import os

import random

import boto3
import tiktoken
import together
from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit, join_room
from flask_sqlalchemy import SQLAlchemy
from groq import Groq
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openai import OpenAI

app = Flask(__name__)

app.config["SECRET_KEY"] = "your_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chat.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# socketio = SocketIO(app, async_mode="eventlet")
socketio = SocketIO(app, async_mode="gevent")

# Global dictionary to keep track of cancellation requests
cancellation_requests = {}

system_users = [
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-2024-08-06",
    "gpt-4-1106-preview",
    "gpt-4-turbo-preview",
    "gpt-4-turbo",
    "mistral",
    "mistral-tiny",
    "mistral-small",
    "mistral-medium",
    "mistral-large-latest",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mixtral-8x7b-32768",
    "llama2-70b-4096",
    "llama3-70b-8192",
    "gemma-7b-it",
    "openchat/openchat-3.5-1210",
    "openchat/openchat-3.5-0106",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "mistral-7b-instruct-v0.2.Q3_K_L.gguf",
    "mistral-7b-instruct-v0.2-code-ft.Q3_K_L.gguf",
    "openhermes-2.5-mistral-7b.Q6_K.gguf",
    "System",
]

HELP_MESSAGE = """
**Available Commands:**
- `/activity [s3_file_path]`: Start an activity from the specified S3 file path.
- `/activity cancel`: Cancel the current activity.
- `/activity info`: Display information about the current activity.
- `/activity metadata`: Display metadata for the current activity.
- `/s3 ls [s3_file_path_pattern]`: List files in S3 matching the pattern.
- `/s3 load [s3_file_path]`: Load a file from S3.
- `/s3 save [s3_key_path]`: Save the most recent code block from the chatroom to S3.
- `/title new`: Generates a new title which reflects conversation content for the current chatroom using gpt-4.
- `/cancel`: Cancel the most recent chat completion from streaming into the chatroom.
- `/help`: Display this help message.

**Available Models:**
- `gpt-3`: Use for GPT-3 model.
- `gpt-4`: Use for GPT-4 model.
- `gpt-4o-2024-08-06`: Use for the cheapest version of GPT-4o.
- `gpt-mini`: Use for GPT-4o-mini model.
- `claude-haiku`: Use for Claude-haiku model.
- `claude-sonnet`: Use for Claude-sonnet model.
- `claude-opus`: Use for Claude-opus model.
- `mistral-tiny`: Use for Mistral-tiny model.
- `mistral-small`: Use for Mistral-small model.
- `mistral-medium`: Use for Mistral-medium model.
- `mistral-large`: Use for Mistral-large model.
- `together/openchat`: Use for Together OpenChat model.
- `together/mistral`: Use for Together Mistral model.
- `together/mixtral`: Use for Together Mixtral model.
- `together/solar`: Use for Together Solar model.
- `groq/mixtral`: Use for Groq Mixtral model.
- `groq/llama2`: Use for Groq Llama-2 model.
- `groq/llama3`: Use for Groq Llama-3 model.
- `groq/gemma`: Use for Groq Gemma model.
- `vllm/hermes-llama-3`: Use for vLLM Hermes model.
- `dall-e-3`: Use for DALL-E 3 model.

The system will process your message and provide a response from the selected language model.
"""


class Room(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False, unique=True)
    title = db.Column(db.String(128), nullable=True)


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(128), nullable=False)
    content = db.Column(db.String(1024), nullable=False)
    token_count = db.Column(db.Integer)
    room_id = db.Column(db.Integer, db.ForeignKey("room.id"), nullable=False)

    def __init__(self, username, content, room_id):
        self.username = username
        self.content = content
        self.room_id = room_id
        self.token_count = self.count_tokens()

    def count_tokens(self):
        # Replace 'gpt-3.5-turbo' with the model you are using.
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.token_count = len(encoding.encode(self.content))
        return self.token_count

    def is_base64_image(self):
        return self.content.startswith('<img src="data:image/jpeg;base64,')


class ActivityState(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    room_id = db.Column(db.Integer, db.ForeignKey("room.id"), nullable=False)
    section_id = db.Column(db.String(128), nullable=False)
    step_id = db.Column(db.String(128), nullable=False)
    attempts = db.Column(db.Integer, default=0)
    max_attempts = db.Column(db.Integer, default=3)
    s3_file_path = db.Column(db.String(256), nullable=False)
    json_metadata = db.Column(db.UnicodeText, default="{}")

    @property
    def dict_metadata(self):
        return json.loads(self.json_metadata) if self.json_metadata else {}

    @dict_metadata.setter
    def dict_metadata(self, value):
        self.json_metadata = json.dumps(value)

    def add_metadata(self, key, value):
        metadata = self.dict_metadata
        metadata[key] = value
        self.dict_metadata = metadata

    def remove_metadata(self, key):
        metadata = self.dict_metadata
        if key in metadata:
            del metadata[key]
        self.dict_metadata = metadata


def get_room(room_name):
    """Utility function to get room from room name."""
    room = Room.query.filter_by(name=room_name).first()
    if room:
        return room
    else:
        # Create a new room since it doesn't exist
        new_room = Room()
        new_room.name = room_name
        db.session.add(new_room)
        db.session.commit()
        return new_room


def get_s3_client():
    """Utility function to get the S3 client with the appropriate profile."""
    if app.config.get("PROFILE_NAME"):
        session = boto3.Session(profile_name=app.config["PROFILE_NAME"])
        s3_client = session.client("s3")
    else:
        s3_client = boto3.client("s3")
    return s3_client


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


@app.route("/search")
def search_page():
    # Query all rooms so that newest is first.
    rooms = Room.query.order_by(Room.id.desc()).all()

    keywords = request.args.get("keywords", "")
    username = request.args.get("username", "guest")
    if not keywords:
        return render_template(
            "search.html",
            rooms=rooms,
            keywords=keywords,
            results=[],
            username=username,
            error="Keywords are required",
        )

    # Call the function to search messages
    search_results = search_messages(keywords)

    return render_template(
        "search.html",
        rooms=rooms,
        keywords=keywords,
        results=search_results,
        username=username,
        error=None,
    )


def search_messages(keywords):
    search_results = {}

    # Split the keywords by spaces
    keyword_list = keywords.lower().split()

    # Search for messages containing any of the keywords
    messages = Message.query.filter(
        db.or_(*[Message.content.ilike(f"%{keyword}%") for keyword in keyword_list])
    ).all()

    for message in messages:
        room = Room.query.get(message.room_id)
        if room:
            # Calculate the score based on the number of occurrences of all keywords
            score = sum(
                message.content.lower().count(keyword) for keyword in keyword_list
            )

            if room.id not in search_results:
                search_results[room.id] = {
                    "room_id": room.id,
                    "room_name": room.name,
                    "room_title": room.title,
                    "score": 0,
                }

            search_results[room.id]["score"] += score

    # Convert the dictionary to a list and sort results by score in descending order
    search_results_list = list(search_results.values())
    search_results_list.sort(key=lambda x: x["score"], reverse=True)

    return search_results_list


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

    # count the number of tokens in this room.
    total_token_count = 0

    # Send the history of messages only to the newly connected client.
    # The reason for using `request.sid` here is to target the specific session (or client) that
    # just connected, so only they receive the backlog of messages, rather than broadcasting
    # this information to all clients in the room.
    for message in previous_messages:
        if not message.is_base64_image():
            total_token_count += message.token_count
        emit(
            "previous_messages",
            {
                "id": message.id,
                "username": message.username,
                "content": message.content,
            },
            room=request.sid,
        )

    message_count = len(previous_messages)
    if room.title is None and message_count >= 6:
        room.title = gpt_generate_room_title(previous_messages)
        db.session.add(room)
        db.session.commit()
        socketio.emit("update_room_title", {"title": room.title}, room=room.name)
        # Emit an event to update this rooms title in the sidebar for all users.
        updated_room_data = {"id": room.id, "name": room.name, "title": room.title}
        socketio.emit("update_room_list", updated_room_data, room=None)

    # Broadcast to all clients in the room that a new user has joined.
    # Here, `room=room` ensures the message is sent to everyone in that specific room.
    emit(
        "message",
        {"id": None, "content": f"{data['username']} has joined the room."},
        room=room.name,
    )
    emit(
        "message",
        {
            "id": None,
            "content": f"Estimated {total_token_count} total tokens in conversation.",
        },
        room=request.sid,
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
            "username": data["username"],
            "content": data["message"],
        },
        room=room.name,
    )

    # detect and process special commands.
    commands = data["message"].splitlines()

    for command in commands:
        if command.startswith("/help"):
            # Emit the help message
            socketio.emit(
                "message",
                {
                    "id": "tmp-1",
                    "username": "System",
                    "content": HELP_MESSAGE,
                },
                room=room_name,
            )
            return
        if command.startswith("/activity cancel"):
            gevent.spawn(cancel_activity, room_name, data["username"])
            # Exit early since we're canceling the activity
            return
        if command.startswith("/activity info"):
            gevent.spawn(display_activity_info, room_name, data["username"])
            # Exit early since we're displaying activity info
            return
        if command.startswith("/activity metadata"):
            gevent.spawn(display_activity_metadata, room_name, data["username"])
            # Exit early since we're displaying activity metadata
            return
        if command.startswith("/activity"):
            s3_file_path = command.split(" ", 1)[1].strip()
            gevent.spawn(start_activity, room_name, s3_file_path, data["username"])
            # Exit early since we're starting an activity
            return
        if command.startswith("/s3 ls"):
            # Extract the S3 file path pattern
            s3_file_path_pattern = command.split(" ", 2)[2].strip()
            # List files from S3 and emit their names
            gevent.spawn(
                list_s3_files, room.name, s3_file_path_pattern, data["username"]
            )
        if command.startswith("/s3 load"):
            # Extract the S3 file path
            s3_file_path = command.split(" ", 2)[2].strip()
            # Load the file from S3 and emit its content
            gevent.spawn(load_s3_file, room_name, s3_file_path, data["username"])
        if command.startswith("/s3 save"):
            # Extract the S3 key path
            s3_key_path = command.split(" ", 2)[2].strip()
            # Save the most recent code block to S3
            gevent.spawn(
                save_code_block_to_s3, room_name, s3_key_path, data["username"]
            )
        if command.startswith("/title new"):
            gevent.spawn(generate_new_title, room_name, data["username"])
        if command.startswith("/cancel"):
            # Cancel the most recent generation request
            gevent.spawn(cancel_generation, room_name)

    # Check if the user is in activity mode
    activity_state = ActivityState.query.filter_by(room_id=room.id).first()
    if activity_state:
        gevent.spawn(
            handle_activity_response, room_name, data["message"], data["username"]
        )

    if "dall-e-3" in data["message"]:
        # Use the entire message as the prompt for DALL-E 3
        # Generate the image and emit its URL
        gevent.spawn(
            generate_dalle_image, data["room_name"], data["message"], data["username"]
        )

    if (
        "claude-" in data["message"]
        or "gpt-" in data["message"]
        or "mistral-" in data["message"]
        or "together/" in data["message"]
        or "localhost/" in data["message"]
        or "vllm/" in data["message"]
        or "groq/" in data["message"]
    ):
        # Emit a temporary message indicating that the llm is processing
        emit(
            "message",
            {"id": None, "content": "<span id='processing'>Processing...</span>"},
            room=room.name,
        )

        if "claude-haiku" in data["message"]:
            gevent.spawn(
                chat_claude,
                data["username"],
                room.name,
                model_name="anthropic.claude-3-haiku-20240307-v1:0",
            )
        if "claude-sonnet" in data["message"]:
            gevent.spawn(chat_claude, data["username"], room.name)
        if "claude-opus" in data["message"]:
            gevent.spawn(
                chat_claude,
                data["username"],
                room.name,
                model_name="anthropic.claude-3-opus-20240229-v1:0",
            )
        if "gpt-3" in data["message"]:
            gevent.spawn(chat_gpt, data["username"], room.name)

        if "gpt-4o-2024-08-06" in data["message"]:
            gevent.spawn(
                chat_gpt,
                data["username"],
                room.name,
                model_name="gpt-4o-2024-08-06",
            )
        elif "gpt-4" in data["message"]:
            gevent.spawn(
                chat_gpt,
                data["username"],
                room.name,
                # model_name="gpt-4o",
                model_name="gpt-4o-2024-08-06",
            )

        if "gpt-mini" in data["message"]:
            gevent.spawn(
                chat_gpt,
                data["username"],
                room.name,
                model_name="gpt-4o-mini",
            )
        if "mistral-tiny" in data["message"]:
            gevent.spawn(
                chat_mistral,
                data["username"],
                room.name,
                model_name="mistral-tiny",
            )
        if "mistral-small" in data["message"]:
            gevent.spawn(
                chat_mistral,
                data["username"],
                room.name,
                model_name="mistral-small",
            )
        if "mistral-medium" in data["message"]:
            gevent.spawn(
                chat_mistral,
                data["username"],
                room.name,
                model_name="mistral-medium",
            )
        if "mistral-large" in data["message"]:
            gevent.spawn(
                chat_mistral,
                data["username"],
                room.name,
                model_name="mistral-large-latest",
            )
        if "together/openchat" in data["message"]:
            gevent.spawn(
                chat_together,
                data["username"],
                room.name,
                model_name="openchat/openchat-3.5-1210",
                stop=["<|end_of_turn|>", "</s>"],
            )
        if "together/mixtral" in data["message"]:
            gevent.spawn(
                chat_together,
                data["username"],
                room.name,
                model_name="mistralai/Mixtral-8x7B-v0.1",
            )
        if "together/mistral" in data["message"]:
            gevent.spawn(
                chat_together,
                data["username"],
                room.name,
                model_name="mistralai/Mistral-7B-Instruct-v0.1",
            )
        if "together/solar" in data["message"]:
            gevent.spawn(
                chat_together,
                data["username"],
                room.name,
                model_name="upstage/SOLAR-10.7B-Instruct-v1.0",
                stop=["###", "</s>"],
            )
        if "groq/mixtral" in data["message"]:
            gevent.spawn(
                chat_groq,
                data["username"],
                room.name,
                model_name="mixtral-8x7b-32768",
            )
        if "groq/llama2" in data["message"]:
            gevent.spawn(
                chat_groq,
                data["username"],
                room.name,
                model_name="llama2-70b-4096",
            )
        if "groq/llama3" in data["message"]:
            gevent.spawn(
                chat_groq,
                data["username"],
                room.name,
                model_name="llama3-70b-8192",
            )
        if "groq/gemma" in data["message"]:
            gevent.spawn(
                chat_groq,
                data["username"],
                room.name,
                model_name="gemma-7b-it",
            )
        if "vllm/openchat" in data["message"]:
            gevent.spawn(
                chat_gpt,
                data["username"],
                room.name,
                model_name="openchat/openchat-3.5-0106",
            )
        if "vllm/hermes-llama-3" in data["message"]:
            gevent.spawn(
                chat_gpt,
                data["username"],
                room.name,
                model_name="NousResearch/Hermes-2-Pro-Llama-3-8B",
            )
        if "localhost/mistral" in data["message"]:
            gevent.spawn(
                chat_llama,
                data["username"],
                room.name,
                model_name="mistral-7b-instruct-v0.2.Q3_K_L.gguf",
            )
        if "localhost/mistral-code" in data["message"]:
            gevent.spawn(
                chat_llama,
                data["username"],
                room.name,
                model_name="mistral-7b-instruct-v0.2-code-ft.Q3_K_L.gguf",
            )
        if "localhost/openhermes" in data["message"]:
            gevent.spawn(
                chat_llama,
                data["username"],
                room.name,
                model_name="openhermes-2.5-mistral-7b.Q6_K.gguf",
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


@socketio.on("update_message")
def handle_update_message(data):
    message_id = data["message_id"]
    new_content = data["content"]
    room_name = data["room_name"]

    # Find the message by ID
    message = Message.query.get(message_id)
    if message:
        # Update the message content
        message.content = new_content
        message.count_tokens()
        db.session.add(message)
        db.session.commit()

        # Emit an event to update the message on all clients
        emit(
            "message_updated",
            {
                "message_id": message_id,
                "content": new_content,
                "username": message.username,
            },
            room=room_name,
        )


def group_consecutive_roles(messages):
    if not messages:
        return []

    grouped_messages = []
    current_role = messages[0]["role"]
    current_content = []

    for message in messages:
        if message["role"] == current_role:
            current_content.append(message["content"])
        else:
            grouped_messages.append(
                {"role": current_role, "content": " ".join(current_content)}
            )
            current_role = message["role"]
            current_content = [message["content"]]

    # Append the last grouped message
    grouped_messages.append(
        {"role": current_role, "content": " ".join(current_content)}
    )

    return grouped_messages


def chat_claude(
    # username, room_name, model_name="anthropic.claude-3-5-sonnet-20240620-v1:0"
    username,
    room_name,
    model_name="anthropic.claude-3-sonnet-20240229-v1:0",
):
    with app.app_context():
        room = get_room(room_name)
        # claude has a 200,000 token context window for prompts.
        all_messages = (
            Message.query.filter_by(room_id=room.id).order_by(Message.id.desc()).all()
        )

    chat_history = []
    for msg in reversed(all_messages):
        if msg.is_base64_image():
            continue
        role = "assistant" if msg.username in system_users else "user"
        chat_history.append({"role": role, "content": msg.content})

    # only claude cares about this constrant.
    chat_history = group_consecutive_roles(chat_history)

    # Initialize the Bedrock client using boto3 and profile name.
    if app.config.get("PROFILE_NAME"):
        session = boto3.Session(profile_name=app.config["PROFILE_NAME"])
        client = session.client("bedrock-runtime", region_name="us-west-2")
    else:
        client = boto3.client("bedrock-runtime", region_name="us-west-2")

    # Define the request parameters
    params = {
        "modelId": model_name,
        "contentType": "application/json",
        "accept": "*/*",
        "body": json.dumps(
            {
                "messages": chat_history,
                "max_tokens": 4096,
                "temperature": 0,
                "top_k": 250,
                "top_p": 0.999,
                "stop_sequences": ["\n\nHuman:"],
                "anthropic_version": "bedrock-2023-05-31",
            }
        ).encode(),
    }

    # Process the event stream
    buffer = ""

    # save empty message, we need the ID when we chunk the response.
    with app.app_context():
        new_message = Message(username=model_name, content=buffer, room_id=room.id)
        db.session.add(new_message)
        db.session.commit()
        msg_id = new_message.id

    try:
        # Invoke the model with response stream
        response = client.invoke_model_with_response_stream(**params)["body"]

        first_chunk = True
        for event in response:
            content = ""

            # Check if there has been a cancellation request, break if there is.
            if cancellation_requests.get(msg_id):
                del cancellation_requests[msg_id]
                break

            if "chunk" in event:
                chunk_data = json.loads(event["chunk"]["bytes"].decode())

                if chunk_data["type"] == "content_block_delta":
                    if chunk_data["delta"]["type"] == "text_delta":
                        content = chunk_data["delta"]["text"]

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

    except Exception as e:
        with app.app_context():
            message_content = f"AWS Bedrock Error: {e}"
            new_message = (
                db.session.query(Message).filter(Message.id == msg_id).one_or_none()
            )
            if new_message:
                new_message.content = message_content
                new_message.count_tokens()
                db.session.add(new_message)
                db.session.commit()
        socketio.emit(
            "message",
            {
                "id": msg_id,
                "username": model_name,
                "content": message_content,
            },
            room=room_name,
        )
        socketio.emit("delete_processing_message", msg_id, room=room.name)
        # exit early to avoid clobbering the error message.
        return None

    # Save the entire completion to the database
    with app.app_context():
        new_message = (
            db.session.query(Message).filter(Message.id == msg_id).one_or_none()
        )
        if new_message:
            new_message.content = buffer
            new_message.count_tokens()
            db.session.add(new_message)
            db.session.commit()

    socketio.emit("delete_processing_message", msg_id, room=room.name)


def get_openai_client_and_model(model_name="gpt-4o-mini"):
    vllm_endpoint = os.environ.get("VLLM_ENDPOINT")
    vllm_api_key = os.environ.get("VLLM_API_KEY", "not-needed")

    if "gpt" not in model_name and vllm_endpoint:
        openai_client = OpenAI(base_url=vllm_endpoint, api_key=vllm_api_key)
    else:
        openai_client = OpenAI()

    return openai_client, model_name


def chat_gpt(username, room_name, model_name="gpt-4o-mini"):
    openai_client, model_name = get_openai_client_and_model(model_name)

    limit = 20
    if "gpt-4" in model_name:
        limit = 1000

    with app.app_context():
        room = get_room(room_name)
        last_messages = (
            Message.query.filter_by(room_id=room.id)
            .order_by(Message.id.desc())
            .limit(limit)
            .all()
        )

        chat_history = [
            {
                "role": "system" if msg.username in system_users else "user",
                # "content": f"{msg.username}: {msg.content}",
                "content": msg.content,
            }
            for msg in reversed(last_messages)
            if not msg.is_base64_image()
        ]

    buffer = ""  # Content buffer for accumulating the chunks

    # save empty message, we need the ID when we chunk the response.
    with app.app_context():
        new_message = Message(username=model_name, content=buffer, room_id=room.id)
        db.session.add(new_message)
        db.session.commit()
        msg_id = new_message.id

    first_chunk = True

    try:
        chunks = openai_client.chat.completions.create(
            model=model_name, messages=chat_history, temperature=0, stream=True
        )
    except Exception as e:
        with app.app_context():
            message_content = f"{model_name} Error: {e}"
            new_message = (
                db.session.query(Message).filter(Message.id == msg_id).one_or_none()
            )
            if new_message:
                new_message.content = message_content
                new_message.count_tokens()
                db.session.add(new_message)
                db.session.commit()
        socketio.emit(
            "message",
            {
                "id": msg_id,
                "username": model_name,
                "content": message_content,
            },
            room=room_name,
        )
        socketio.emit("delete_processing_message", msg_id, room=room.name)
        # exit early to avoid clobbering the error message.
        return None

    for chunk in chunks:
        # Check if there has been a cancellation request, break if there is.
        if cancellation_requests.get(msg_id):
            del cancellation_requests[msg_id]
            break

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
            new_message.count_tokens()
            db.session.add(new_message)
            db.session.commit()

    socketio.emit("delete_processing_message", msg_id, room=room.name)


def chat_mistral(username, room_name, model_name="mistral-tiny"):
    with app.app_context():
        room = get_room(room_name)
        last_messages = (
            Message.query.filter_by(room_id=room.id)
            .order_by(Message.id.desc())
            .limit(50)
            .all()
        )

        chat_history = []
        combined_content = ""
        last_role = None

        # Function to add a ChatMessage to the history
        def add_message(role, content):
            if content:
                chat_history.append(ChatMessage(role=role, content=content))

        # Iterate over messages to combine consecutive assistant messages
        for msg in reversed(last_messages):
            if msg.is_base64_image():
                continue
            current_role = "assistant" if msg.username in system_users else "user"
            formatted_content = f"{msg.username}: {msg.content}"

            if current_role == last_role and current_role == "assistant":
                # Combine messages if the current and last messages are from the assistant
                combined_content += "\n" + formatted_content
            else:
                # Add the previous combined message to chat history if roles switch
                add_message(last_role, combined_content)
                combined_content = formatted_content  # Start new combination
                last_role = current_role

        # Add the last combined message to the chat history
        add_message(last_role, combined_content)

        # Remove trailing assistant messages until a user message is found.
        while chat_history and chat_history[-1].role == "assistant":
            chat_history.pop()

    # Initialize the Mistral client
    mistral_client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])

    buffer = ""  # Content buffer for accumulating the chunks

    # Save an empty message to get an ID for the chunks
    with app.app_context():
        new_message = Message(username=model_name, content=buffer, room_id=room.id)
        db.session.add(new_message)
        db.session.commit()
        msg_id = new_message.id

    first_chunk = True

    try:
        # Use the Mistral client to stream the chat completion
        for chunk in mistral_client.chat_stream(
            model=model_name, messages=chat_history
        ):
            # Check if there has been a cancellation request, break if there is.
            if cancellation_requests.get(msg_id):
                del cancellation_requests[msg_id]
                break

            content_chunk = chunk.choices[0].delta.content

            if content_chunk:
                buffer += content_chunk  # Accumulate content

                if first_chunk:
                    socketio.emit(
                        "message_chunk",
                        {
                            "id": msg_id,
                            "content": f"**{username} ({model_name}):**\n\n{content_chunk}",
                        },
                        room=room.name,
                    )
                    first_chunk = False
                else:
                    socketio.emit(
                        "message_chunk",
                        {"id": msg_id, "content": content_chunk},
                        room=room.name,
                    )
                socketio.sleep(0)  # Force immediate handling

    except Exception as e:
        with app.app_context():
            message_content = f"Mistral Error: {e}"
            new_message = (
                db.session.query(Message).filter(Message.id == msg_id).one_or_none()
            )
            if new_message:
                new_message.content = message_content
                new_message.count_tokens()
                db.session.add(new_message)
                db.session.commit()
        socketio.emit(
            "message",
            {
                "id": msg_id,
                "username": model_name,
                "content": message_content,
            },
            room=room.name,
        )
        return None

    # Save the entire completion to the database
    with app.app_context():
        new_message = (
            db.session.query(Message).filter(Message.id == msg_id).one_or_none()
        )
        if new_message:
            new_message.content = buffer
            new_message.count_tokens()
            db.session.add(new_message)
            db.session.commit()

    socketio.emit("delete_processing_message", msg_id, room=room.name)


def chat_together(
    username,
    room_name,
    message,
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    stop=["[/INST]", "</s>"],
):
    together.api_key = os.environ["TOGETHER_API_KEY"]

    with app.app_context():
        room = get_room(room_name)
        last_messages = (
            Message.query.filter_by(room_id=room.id)
            .order_by(Message.id.desc())
            .limit(15)
            .all()
        )

        chat_history = [
            f"{msg.username}: {msg.content}"
            for msg in reversed(last_messages)
            if not msg.is_base64_image()
        ]
        if "mistralai" in model_name:
            chat_history_str = "\n\n".join(chat_history)
        elif "solar" in model_name:
            chat_history_str = "### \n\n".join(chat_history)
            chat_history_str += "### Assistant:"

        else:
            chat_history_str = "<|end_of_turn|>\n\n".join(chat_history)
            chat_history_str += "<|end_of_turn|>Math Correct Assistant:"

    buffer = ""  # Content buffer for accumulating the chunks

    # Save an empty message to get an ID for the chunks
    with app.app_context():
        new_message = Message(username=model_name, content=buffer, room_id=room.id)
        db.session.add(new_message)
        db.session.commit()
        msg_id = new_message.id

    first_chunk = True

    try:
        # Use the Together client to stream the chat completion
        prompt = f"{chat_history_str}"
        if "mistralai" in model_name:
            prompt = f"[INST] {chat_history_str} [/INST]"
        if "solar" in model_name:
            prompt = f"<s> {chat_history_str}"

        chunks = together.Complete.create_streaming(
            prompt,
            model=model_name,
            max_tokens=2048,
            stop=stop,
            repetition_penalty=1,
            top_p=0.7,
            top_k=50,
        )

        for chunk in chunks:
            # Check if there has been a cancellation request, break if there is.
            if cancellation_requests.get(msg_id):
                del cancellation_requests[msg_id]
                break

            buffer += chunk  # Accumulate content

            if first_chunk:
                socketio.emit(
                    "message_chunk",
                    {
                        "id": msg_id,
                        "content": f"**{username} ({model_name}):**\n\n{chunk}",
                    },
                    room=room.name,
                )
                first_chunk = False
            else:
                socketio.emit(
                    "message_chunk",
                    {"id": msg_id, "content": chunk},
                    room=room.name,
                )
            socketio.sleep(0)  # Force immediate handling

    except Exception as e:
        with app.app_context():
            message_content = f"Together Error: {e}"
            new_message = (
                db.session.query(Message).filter(Message.id == msg_id).one_or_none()
            )
            if new_message:
                new_message.content = message_content
                new_message.count_tokens()
                db.session.add(new_message)
                db.session.commit()
        socketio.emit(
            "message",
            {
                "id": msg_id,
                "username": model_name,
                "content": message_content,
            },
            room=room.name,
        )
        return None

    # Save the entire completion to the database
    with app.app_context():
        new_message = (
            db.session.query(Message).filter(Message.id == msg_id).one_or_none()
        )
        if new_message:
            new_message.content = buffer
            new_message.count_tokens()
            db.session.add(new_message)
            db.session.commit()

    socketio.emit("delete_processing_message", msg_id, room=room.name)


def chat_groq(username, room_name, model_name="mixtral-8x7b-32768"):
    # https://console.groq.com/docs/models
    _limit = 15
    if "mixtral" in model_name:
        _limit = 50

    with app.app_context():
        room = get_room(room_name)
        last_messages = (
            Message.query.filter_by(room_id=room.id)
            .order_by(Message.id.desc())
            .limit(_limit)
            .all()
        )

        chat_history = [
            {
                "role": "system" if msg.username in system_users else "user",
                "content": msg.content,
            }
            for msg in reversed(last_messages)
            if not msg.is_base64_image()
        ]

    # Initialize the Groq client
    client = Groq()

    buffer = ""  # Content buffer for accumulating the chunks

    # Save an empty message to get an ID for the chunks
    with app.app_context():
        new_message = Message(username=model_name, content=buffer, room_id=room.id)
        db.session.add(new_message)
        db.session.commit()
        msg_id = new_message.id

    first_chunk = True

    try:
        # Use the Groq client to stream the chat completion
        stream = client.chat.completions.create(
            messages=chat_history,
            model=model_name,
            stream=True,
        )

        for chunk in stream:
            content_chunk = chunk.choices[0].delta.content

            if content_chunk:
                buffer += content_chunk  # Accumulate content

                if first_chunk:
                    socketio.emit(
                        "message_chunk",
                        {
                            "id": msg_id,
                            "content": f"**{username} ({model_name}):**\n\n{content_chunk}",
                        },
                        room=room.name,
                    )
                    first_chunk = False
                else:
                    socketio.emit(
                        "message_chunk",
                        {"id": msg_id, "content": content_chunk},
                        room=room.name,
                    )
                socketio.sleep(0)  # Force immediate handling

    except Exception as e:
        with app.app_context():
            message_content = f"Groq Error: {e}"
            new_message = (
                db.session.query(Message).filter(Message.id == msg_id).one_or_none()
            )
            if new_message:
                new_message.content = message_content
                new_message.count_tokens()
                db.session.add(new_message)
                db.session.commit()
        socketio.emit(
            "message",
            {
                "id": msg_id,
                "username": model_name,
                "content": message_content,
            },
            room=room.name,
        )
        return None

    # Save the entire completion to the database
    with app.app_context():
        new_message = (
            db.session.query(Message).filter(Message.id == msg_id).one_or_none()
        )
        if new_message:
            new_message.content = buffer
            new_message.count_tokens()
            db.session.add(new_message)
            db.session.commit()

    socketio.emit("delete_processing_message", msg_id, room=room.name)


def chat_llama(username, room_name, model_name="mistral-7b-instruct-v0.2.Q3_K_L.gguf"):
    import llama_cpp

    # https://llama-cpp-python.readthedocs.io/en/latest/api-reference/
    model = llama_cpp.Llama(model_name, n_gpu_layers=-1, n_ctx=32000)

    limit = 15
    with app.app_context():
        room = get_room(room_name)
        last_messages = (
            Message.query.filter_by(room_id=room.id)
            .order_by(Message.id.desc())
            .limit(limit)
            .all()
        )

        chat_history = [
            {
                "role": "system" if msg.username in system_users else "user",
                "content": f"{msg.username}: {msg.content}",
            }
            for msg in reversed(last_messages)
            if not msg.is_base64_image()
        ]

    buffer = ""  # Content buffer for accumulating the chunks

    # save empty message, we need the ID when we chunk the response.
    with app.app_context():
        new_message = Message(username=model_name, content=buffer, room_id=room.id)
        db.session.add(new_message)
        db.session.commit()
        msg_id = new_message.id

    first_chunk = True

    try:
        chunks = model.create_chat_completion(
            messages=chat_history,
            stream=True,
        )
    except Exception as e:
        with app.app_context():
            message_content = f"LLama Error: {e}"
            new_message = (
                db.session.query(Message).filter(Message.id == msg_id).one_or_none()
            )
            if new_message:
                new_message.content = message_content
                new_message.count_tokens()
                db.session.add(new_message)
                db.session.commit()
        socketio.emit(
            "message",
            {
                "id": msg_id,
                "username": model_name,
                "content": message_content,
            },
            room=room_name,
        )
        socketio.emit("delete_processing_message", msg_id, room=room.name)
        # exit early to avoid clobbering the error message.
        return None

    for chunk in chunks:
        # Check if there has been a cancellation request, break if there is.
        if cancellation_requests.get(msg_id):
            del cancellation_requests[msg_id]
            break

        content = chunk["choices"][0]["delta"].get("content")

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
            new_message.count_tokens()
            db.session.add(new_message)
            db.session.commit()

    socketio.emit("delete_processing_message", msg_id, room=room.name)


def gpt_generate_room_title(messages):
    """
    Generate a title for the room based on a list of messages.
    """
    openai_client, model_name = get_openai_client_and_model()

    chat_history = [
        {
            "role": "system" if msg.username in system_users else "user",
            "content": f"{msg.username}: {msg.content}",
        }
        for msg in reversed(messages)
        if not msg.is_base64_image()
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


def generate_new_title(room_name, username):
    with app.app_context():
        room = get_room(room_name)
        # Get the last few messages to generate a title
        last_messages = (
            Message.query.filter_by(room_id=room.id)
            .order_by(Message.id.desc())
            .limit(1000)  # Adjust the limit as needed
            .all()
        )

        # Generate the title using the messages
        new_title = gpt_generate_room_title(last_messages)

        # Update the room title in the database
        room.title = new_title
        db.session.add(room)
        db.session.commit()

        # Emit the new title to the room.
        socketio.emit("update_room_title", {"title": new_title}, room=room_name)

        # Emit an event to update this rooms title in the sidebar for all users.
        updated_room_data = {"id": room.id, "name": room.name, "title": room.title}
        socketio.emit("update_room_list", updated_room_data, room=None)

        # Optionally, send a confirmation message to the room
        confirmation_message = f"New title created: {new_title}"
        new_message = Message(
            username=username, content=confirmation_message, room_id=room.id
        )
        db.session.add(new_message)
        db.session.commit()
        socketio.emit(
            "message",
            {
                "id": new_message.id,
                "username": username,
                "content": confirmation_message,
            },
            room=room_name,
        )


def generate_dalle_image(room_name, message, username):
    socketio.emit(
        "message",
        {"id": None, "content": "Processing..."},
        room=room_name,
    )

    openai_client = OpenAI()
    # Initialize the content variable to hold either the image tag or an error message
    content = ""

    try:
        # Call the DALL-E 3 API to generate an image in base64 format
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=message,
            n=1,
            size="1024x1024",
            response_format="b64_json",
        )

        # Access the base64-encoded image data
        image_data = response.data[0].b64_json
        revised_prompt = response.data[0].revised_prompt

        # Create an HTML img tag with the base64 data
        content = f'<img src="data:image/jpeg;base64,{image_data}" alt="{message}"><p>{revised_prompt}</p>'

    except Exception as e:
        # Set the content to an error message
        content = f"Error generating image: {e}"

    # Store the content in the database and emit to the frontend
    with app.app_context():
        room = get_room(room_name)
        new_message = Message(
            username=username,
            content=content,  # Store the img tag or error message as the content
            room_id=room.id,  # Make sure you have the room ID available
        )
        db.session.add(new_message)
        db.session.commit()

        # Emit the message with the content to the frontend
        socketio.emit(
            "message",
            {"id": new_message.id, "username": username, "content": content},
            room=room_name,
        )


def find_most_recent_code_block(room_name):
    with app.app_context():
        # Get the room object from the database
        room = get_room(room_name)
        if not room:
            return None  # Room not found

        # Get the most recent message for the room
        latest_message = (
            Message.query.filter_by(room_id=room.id)
            .order_by(Message.id.desc())
            .offset(1)
            .first()
        )

    if latest_message:
        # Split the message content into lines
        lines = latest_message.content.split("\n")
        # Initialize variables to store the code block
        code_block_lines = []
        code_block_started = False
        for line in lines:
            # Check if the line starts with a code block fence
            if line.startswith("```"):
                # If we've already started capturing, this fence ends the block
                if code_block_started:
                    break
                else:
                    # Start capturing from the next line
                    code_block_started = True
                    continue
            elif code_block_started:
                # If we're inside a code block, capture the line
                code_block_lines.append(line)

        # Join the captured lines to form the code block content
        code_block_content = "\n".join(code_block_lines)
        return code_block_content

    # No code block found in the latest message
    return None


def save_code_block_to_s3(room_name, s3_key_path, username):
    # Initialize the S3 client
    s3_client = get_s3_client()

    # Assuming the bucket name is set in an environment variable
    bucket_name = os.environ.get("S3_BUCKET_NAME")

    # Find the most recent code block
    code_block_content = find_most_recent_code_block(room_name)

    # Initialize a variable to hold the message content
    message_content = ""

    if code_block_content:
        try:
            # Save the code block content to S3
            s3_client.put_object(
                Bucket=bucket_name, Key=s3_key_path, Body=code_block_content
            )
            # Set the success message content
            message_content = f"Code block saved to S3 at {s3_key_path}"
        except Exception as e:
            # Set the error message content if S3 save fails
            message_content = f"Error saving file to S3: {e}"
    else:
        # Set the error message content if no code block is found
        message_content = "No code block found to save to S3."

    # Save the message to the database and emit to the frontend
    with app.app_context():
        # Get the room object from the database
        room = get_room(room_name)
        if room:
            # Create a new message object
            new_message = Message(
                username=username, content=message_content, room_id=room.id
            )
            # Add the new message to the session and commit
            db.session.add(new_message)
            db.session.commit()

            # Emit the message to the frontend with the new message ID
            socketio.emit(
                "message",
                {
                    "id": new_message.id,
                    "username": username,
                    "content": message_content,
                },
                room=room_name,
            )


def load_s3_file(room_name, s3_file_path, username):
    # Initialize the S3 client
    s3_client = get_s3_client()

    # Assuming the bucket name is set in an environment variable
    bucket_name = os.environ.get("S3_BUCKET_NAME")

    # Initialize message content variable
    message_content = ""

    try:
        # Retrieve the file content from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_file_path)
        file_content = response["Body"].read().decode("utf-8")

        # Format the file content as a code block
        message_content = f"```\n{file_content}\n```"

    except Exception as e:
        # Handle errors (e.g., file not found, access denied)
        message_content = f"Error loading file from S3: {e}"

    # Save the message to the database and emit to the chatroom
    with app.app_context():
        room = get_room(room_name)
        new_message = Message(
            username=username,
            content=message_content,
            room_id=room.id,
        )
        db.session.add(new_message)
        db.session.commit()

        # Emit the message to the chatroom with the message ID
        socketio.emit(
            "message",
            {
                "id": new_message.id,
                "username": username,
                "content": message_content,
            },
            room=room_name,
        )


def list_s3_files(room_name, s3_file_path_pattern, username):
    import fnmatch
    from datetime import timezone

    # Initialize the S3 client
    s3_client = get_s3_client()

    # Assuming the bucket name is set in an environment variable
    bucket_name = os.environ.get("S3_BUCKET_NAME")

    # Initialize the list to hold all file information
    files = []

    # Initialize the pagination token
    continuation_token = None

    # Loop to handle pagination
    while True:
        # List objects in the S3 bucket with pagination support
        list_kwargs = {
            "Bucket": bucket_name,
        }
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token

        response = s3_client.list_objects_v2(**list_kwargs)

        # Process the current page of results
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if s3_file_path_pattern == "*" or fnmatch.fnmatch(
                key, s3_file_path_pattern
            ):
                size = obj["Size"]
                last_modified = obj["LastModified"]
                # Convert last_modified to a timezone-aware datetime object
                last_modified = (
                    last_modified.replace(tzinfo=timezone.utc)
                    .astimezone(tz=None)
                    .strftime("%Y-%m-%d %H:%M:%S %Z")
                )
                files.append(
                    f"{key} (Size: {size} bytes, Last Modified: {last_modified})"
                )

        # Check if there are more pages
        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break  # No more pages

    # Format the message content with the list of files and metadata
    message_content = (
        "```\n" + "\n".join(files) + "\n```" if files else "No files found."
    )

    # Save the message to the database and emit to the chatroom
    with app.app_context():
        room = Room.query.filter_by(name=room_name).first()
        if room:
            new_message = Message(
                username=username,
                content=message_content,
                room_id=room.id,
            )
            db.session.add(new_message)
            db.session.commit()

            # Emit the message to the chatroom with the message ID
            socketio.emit(
                "message",
                {
                    "id": new_message.id,
                    "username": username,
                    "content": message_content,
                },
                room=room_name,
            )


def cancel_generation(room_name):
    with app.app_context():
        room = get_room(room_name)
        # Get the most recent message for the room that is being generated
        latest_message = (
            Message.query.filter_by(room_id=room.id)
            .order_by(Message.id.desc())
            .offset(1)
            .first()
        )

    if latest_message:
        # Set the cancellation request for the given message ID
        cancellation_requests[latest_message.id] = True
        # Optionally, inform the user that the generation has been canceled
        socketio.emit(
            "message",
            {
                "id": None,
                "username": "System",
                "content": f"Generation for message ID {latest_message.id} has been canceled.",
            },
            room=room_name,
        )


def get_activity_content(file_path):
    """
    Load the activity content from either S3 or the local filesystem based on the configuration.
    """
    if app.config["LOCAL_ACTIVITIES"]:
        # Load the activity YAML from a local file
        with open(file_path, "r") as file:
            activity_yaml = file.read()
    else:
        # Load the activity YAML from S3
        s3_client = get_s3_client()
        bucket_name = os.environ.get("S3_BUCKET_NAME")
        response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
        activity_yaml = response["Body"].read().decode("utf-8")

    return yaml.safe_load(activity_yaml)


def loop_through_steps_until_question(
    activity_content, activity_state, room_name, username
):
    room = get_room(room_name)

    current_section_id = activity_state.section_id
    current_step_id = activity_state.step_id

    # Get the user's language preference from metadata
    user_language = activity_state.dict_metadata.get("language", "English")

    while True:
        section = next(
            (
                s
                for s in activity_content["sections"]
                if s["section_id"] == current_section_id
            ),
            None,
        )
        if not section:
            break

        step = next(
            (s for s in section["steps"] if s["step_id"] == current_step_id), None
        )
        if not step:
            break

        # Emit the current step content blocks
        content = "\n\n".join(step["content_blocks"])
        translated_content = translate_text(content, user_language)
        new_message = Message(
            username="System", content=translated_content, room_id=room.id
        )
        db.session.add(new_message)
        db.session.commit()

        socketio.emit(
            "message",
            {
                "id": new_message.id,
                "username": "System",
                "content": translated_content,
            },
            room=room_name,
        )

        # Check if the current step has a question
        if "question" in step:
            question_content = f"Question: {step['question']}"
            translated_question_content = translate_text(
                question_content, user_language
            )
            new_message = Message(
                username="System", content=translated_question_content, room_id=room.id
            )
            db.session.add(new_message)
            db.session.commit()

            socketio.emit(
                "message",
                {
                    "id": new_message.id,
                    "username": "System",
                    "content": translated_question_content,
                },
                room=room_name,
            )
            break

        # Move to the next step
        next_section, next_step = get_next_step(
            activity_content, current_section_id, current_step_id
        )

        if next_step:
            activity_state.section_id = next_section["section_id"]
            activity_state.step_id = next_step["step_id"]
            activity_state.attempts = 0

            db.session.add(activity_state)
            db.session.commit()

            current_section_id = next_section["section_id"]
            current_step_id = next_step["step_id"]
        else:
            # Activity completed

            # Display activity info before completing
            display_activity_info(room_name, username)

            db.session.delete(activity_state)
            db.session.commit()
            socketio.emit(
                "message",
                {
                    "id": None,
                    "username": "System",
                    "content": "Activity completed!",
                },
                room=room_name,
            )
            break


def start_activity(room_name, s3_file_path, username):
    activity_content = get_activity_content(s3_file_path)

    with app.app_context():
        # Save the initial state to the database
        room = get_room(room_name)
        initial_section = activity_content["sections"][0]
        initial_step = initial_section["steps"][0]

        activity_state = ActivityState(
            room_id=room.id,
            section_id=initial_section["section_id"],
            step_id=initial_step["step_id"],
            max_attempts=activity_content.get("default_max_attempts_per_step", 3),
            s3_file_path=s3_file_path,  # Save the S3 file path
        )
        db.session.add(activity_state)
        db.session.commit()

        # Loop through steps until a question is found or the end is reached
        loop_through_steps_until_question(
            activity_content, activity_state, room_name, username
        )


def cancel_activity(room_name, username):
    with app.app_context():
        room = get_room(room_name)
        activity_state = ActivityState.query.filter_by(room_id=room.id).first()

        if not activity_state:
            socketio.emit(
                "message",
                {
                    "id": None,
                    "username": "System",
                    "content": "No active activity found to cancel.",
                },
                room=room_name,
            )
            return

        # Delete the activity state
        db.session.delete(activity_state)
        db.session.commit()

        # Emit a message indicating the activity has been canceled
        socketio.emit(
            "message",
            {
                "id": None,
                "username": "System",
                "content": "Activity has been canceled.",
            },
            room=room_name,
        )


def display_activity_metadata(room_name, username):
    with app.app_context():
        room = get_room(room_name)
        activity_state = ActivityState.query.filter_by(room_id=room.id).first()

        if not activity_state:
            socketio.emit(
                "message",
                {
                    "id": None,
                    "username": "System",
                    "content": "No active activity found.",
                },
                room=room_name,
            )
            return

        # Pretty print the metadata
        metadata_pretty = json.dumps(activity_state.dict_metadata, indent=2)

        # Store and emit the metadata
        metadata_message = f"```\n{metadata_pretty}\n```"
        new_message = Message(
            username="System", content=metadata_message, room_id=room.id
        )
        db.session.add(new_message)
        db.session.commit()

        socketio.emit(
            "message",
            {
                "id": new_message.id,
                "username": "System",
                "content": metadata_message,
            },
            room=room_name,
        )


def handle_activity_response(room_name, user_response, username):
    with app.app_context():
        room = get_room(room_name)
        activity_state = ActivityState.query.filter_by(room_id=room.id).first()

        if not activity_state:
            return

        # Load the activity content
        activity_content = get_activity_content(activity_state.s3_file_path)

        try:
            # Find the current section and step
            section = next(
                s
                for s in activity_content["sections"]
                if s["section_id"] == activity_state.section_id
            )
            step = next(
                s for s in section["steps"] if s["step_id"] == activity_state.step_id
            )

            # Check if the step has a question
            if "question" in step:
                # Categorize the user's response
                category = categorize_response(
                    step["question"],
                    user_response,
                    step["buckets"],
                    step["tokens_for_ai"],
                )

                transition = step["transitions"].get(category, None)

                if transition is None:
                    # Emit an error message and return early
                    socketio.emit(
                        "message",
                        {
                            "id": None,
                            "username": "System",
                            "content": f"Error: Unrecognized category '{category}'. Please try again.",
                        },
                        room=room_name,
                    )
                    return

                next_section_and_step = transition.get("next_section_and_step", None)
                counts_as_attempt = transition.get("counts_as_attempt", True)

                # Emit the category to the frontend
                socketio.emit(
                    "message",
                    {
                        "id": None,
                        "username": "System",
                        "content": f"Category: {category}",
                    },
                    room=room_name,
                )

                # Check metadata conditions for the current step
                if "metadata_conditions" in transition:
                    conditions_met = all(
                        activity_state.dict_metadata.get(key) == value
                        for key, value in transition["metadata_conditions"].items()
                    )
                    if not conditions_met:
                        # Emit a message indicating the conditions are not met
                        socketio.emit(
                            "message",
                            {
                                "id": None,
                                "username": "System",
                                "content": "You do not have the required items to proceed.",
                            },
                            room=room_name,
                        )
                        # Remind the user of what they can do in the room
                        content_blocks = step.get("content_blocks", [])
                        question = step.get("question", "")
                        options_message = (
                            "\n\n".join(content_blocks) + "\n\n" + question
                        )

                        new_message = Message(
                            username="System", content=options_message, room_id=room.id
                        )
                        db.session.add(new_message)
                        db.session.commit()

                        socketio.emit(
                            "message",
                            {
                                "id": new_message.id,
                                "username": "System",
                                "content": options_message,
                            },
                            room=room_name,
                        )
                        return

                # this gives the llm context on what changed.
                new_metadata = {}

                # Update metadata based on user actions
                if "metadata_add" in transition:
                    for key, value in transition["metadata_add"].items():
                        if value == "the-users-response":
                            value = user_response
                        new_metadata[key] = value
                        activity_state.add_metadata(key, value)

                if "metadata_remove" in transition:
                    for key in transition["metadata_remove"]:
                        activity_state.remove_metadata(key)

                # Handle metadata_random
                if "metadata_random" in transition:
                    random_key = random.choice(
                        list(transition["metadata_random"].keys())
                    )
                    random_value = transition["metadata_random"][random_key]
                    new_metadata[random_key] = random_value
                    activity_state.add_metadata(random_key, random_value)

                # Commit the changes after the loop
                db.session.add(activity_state)
                db.session.commit()

                user_language = activity_state.dict_metadata.get("language", "English")

                # Provide feedback based on the category
                feedback = provide_feedback(
                    transition,
                    category,
                    step["question"],
                    # step["tokens_for_ai"],
                    "",
                    user_response,
                    user_language,
                    username,
                    activity_state.json_metadata,
                    json.dumps(new_metadata),
                )

                # Store and emit the feedback
                if feedback:
                    # feedback is metadata language aware, doesn't need to be translated.
                    new_message = Message(
                        username="System", content=feedback, room_id=room.id
                    )
                    db.session.add(new_message)
                    db.session.commit()

                    socketio.emit(
                        "message",
                        {
                            "id": new_message.id,
                            "username": "System",
                            "content": feedback,
                        },
                        room=room_name,
                    )

                # Emit the transition content blocks if they exist
                if "content_blocks" in transition:
                    transition_content = "\n\n".join(transition["content_blocks"])
                    translated_transition_content = translate_text(
                        transition_content, user_language
                    )
                    new_message = Message(
                        username="System",
                        content=translated_transition_content,
                        room_id=room.id,
                    )
                    db.session.add(new_message)
                    db.session.commit()

                    socketio.emit(
                        "message",
                        {
                            "id": new_message.id,
                            "username": "System",
                            "content": translated_transition_content,
                        },
                        room=room_name,
                    )

                # if "correct" or max_attempts reached.
                if (
                    category
                    not in [
                        "off_topic",
                        "asking_clarifying_questions",
                        "partial_understanding",
                    ]
                    or activity_state.attempts >= activity_state.max_attempts
                ):
                    if next_section_and_step:
                        (
                            current_section_id,
                            current_step_id,
                        ) = next_section_and_step.split(":")
                        next_section = next(
                            s
                            for s in activity_content["sections"]
                            if s["section_id"] == current_section_id
                        )
                        next_step = next(
                            s
                            for s in next_section["steps"]
                            if s["step_id"] == current_step_id
                        )
                    else:
                        # Move to the next step or section
                        next_section, next_step = get_next_step(
                            activity_content, section["section_id"], step["step_id"]
                        )

                    if next_step:
                        activity_state.section_id = next_section["section_id"]
                        activity_state.step_id = next_step["step_id"]
                        activity_state.attempts = 0

                        db.session.add(activity_state)
                        db.session.commit()

                        # Loop through steps until a question is found or the end is reached
                        loop_through_steps_until_question(
                            activity_content, activity_state, room_name, username
                        )
                else:
                    # the user response is any bucket other than correct.
                    if counts_as_attempt:
                        activity_state.attempts += 1
                        db.session.add(activity_state)
                        db.session.commit()

                    # Emit the question again
                    question_content = f"Question: {step['question']}"
                    translated_question_content = translate_text(
                        question_content, user_language
                    )
                    new_message = Message(
                        username="System",
                        content=translated_question_content,
                        room_id=room.id,
                    )
                    db.session.add(new_message)
                    db.session.commit()

                    socketio.emit(
                        "message",
                        {
                            "id": new_message.id,
                            "username": "System",
                            "content": translated_question_content,
                        },
                        room=room_name,
                    )
            else:
                # Handle steps without a question
                loop_through_steps_until_question(
                    activity_content, activity_state, room_name, username
                )

        except Exception as e:
            import traceback

            msg = traceback.format_exc()
            socketio.emit(
                "message",
                {
                    "id": None,
                    "username": "System",
                    "content": f"Error processing activity response: {e}\n\n{msg}",
                },
                room=room_name,
            )


def display_activity_info(room_name, username):
    with app.app_context():
        room = get_room(room_name)
        activity_state = ActivityState.query.filter_by(room_id=room.id).first()

        if not activity_state:
            socketio.emit(
                "message",
                {
                    "id": None,
                    "username": "System",
                    "content": "No active activity found.",
                },
                room=room_name,
            )
            return

        # Load the activity content
        activity_content = get_activity_content(activity_state.s3_file_path)

        try:
            # Fetch the entire room history
            all_messages = (
                Message.query.filter_by(room_id=room.id)
                .order_by(Message.id.asc())
                .all()
            )
            chat_history = [
                {
                    "role": "system" if msg.username in system_users else "user",
                    "username": msg.username,
                    "content": msg.content,
                }
                for msg in all_messages
            ]

            # Prepare the rubric for grading
            rubric = activity_content.get(
                "tokens_for_ai_rubric",
                """
                Grade the responses of all users based on the following criteria:
                - Accuracy: How correct is the response?
                - Completeness: Does the response fully address the question?
                - Clarity: Is the response clear and easy to understand?
                - Engagement: Is the response engaging and interesting?
                Provide a score out of 10 for each criterion and an overall grade for each user.
                Finally order each user by who is winning. Number of correct answers and accuracy & include an enumeration of the feats!
                Take into account how many attempts the user took to get a passing answer when ranking.
                Don't just try to give the user a "B" or 35/40, really figure out a good placement considering some people don't know how to type.
            """,
            )

            # Generate the grading using the AI
            grading_message = generate_grading(chat_history, rubric)

            # Store and emit the activity info
            info_message = f"Activity Info:\nCurrent Section: {activity_state.section_id}\nCurrent Step: {activity_state.step_id}\nAttempts: {activity_state.attempts}\n\n{grading_message}"
            new_message = Message(
                username="System", content=info_message, room_id=room.id
            )
            db.session.add(new_message)
            db.session.commit()

            socketio.emit(
                "message",
                {
                    "id": new_message.id,
                    "username": "System",
                    "content": info_message,
                },
                room=room_name,
            )

        except Exception as e:
            socketio.emit(
                "message",
                {
                    "id": None,
                    "username": "System",
                    "content": f"Error displaying activity info: {e}",
                },
                room=room_name,
            )
            # Debugging: Log exception
            print(f"Exception: {e}")


def generate_grading(chat_history, rubric):
    openai_client, model_name = get_openai_client_and_model()
    messages = [
        {
            "role": "system",
            "content": f"Using the following rubric, grade the responses in the chat history:\n\n{rubric}",
        },
        {
            "role": "user",
            "content": f"Chat History:\n\n{json.dumps(chat_history, indent=2)}",
        },
    ]

    try:
        completion = openai_client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=1000, temperature=0.7
        )
        grading = completion.choices[0].message.content.strip()
        return grading
    except Exception as e:
        return f"Error generating grading: {e}"


def get_next_step(activity_content, current_section_id, current_step_id):
    for section in activity_content["sections"]:
        if section["section_id"] == current_section_id:
            for i, step in enumerate(section["steps"]):
                if step["step_id"] == current_step_id:
                    if i + 1 < len(section["steps"]):
                        return section, section["steps"][i + 1]
                    else:
                        # Move to the next section
                        next_section_index = (
                            activity_content["sections"].index(section) + 1
                        )
                        if next_section_index < len(activity_content["sections"]):
                            next_section = activity_content["sections"][
                                next_section_index
                            ]
                            return next_section, next_section["steps"][0]
    return None, None


# Categorize the user's response.
def categorize_response(question, response, buckets, tokens_for_ai):
    openai_client, model_name = get_openai_client_and_model()
    bucket_list = ", ".join(buckets)
    messages = [
        {
            "role": "system",
            "content": f"{tokens_for_ai} Categorize the following response into one of the following buckets: {bucket_list}. Return ONLY a bucket label.",
        },
        {
            "role": "user",
            "content": f"Question: {question}\nResponse: {response}\n\nCategory:",
        },
    ]

    try:
        completion = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=10,
            temperature=0,
        )
        category = (
            completion.choices[0].message.content.strip().lower().replace(" ", "_")
        )
        return category
    except Exception as e:
        return f"Error: {e}"


# Generate AI feedback
def generate_ai_feedback(
    category,
    question,
    user_response,
    tokens_for_ai,
    username,
    json_metadata,
    json_new_metadata,
):
    openai_client, model_name = get_openai_client_and_model()
    messages = [
        {
            "role": "system",
            "content": f"{tokens_for_ai} Generate a human-readable feedback message based on the following:",
        },
        {
            "role": "user",
            "content": f"Username: {username}\nQuestion: {question}\nResponse: {user_response}\nCategory: {category}\nMetadata: {json_metadata}\n New Metadata: {json_new_metadata}",
        },
    ]

    try:
        completion = openai_client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=1000, temperature=0.7
        )
        feedback = completion.choices[0].message.content.strip()
        return feedback
    except Exception as e:
        return f"Error: {e}"


def provide_feedback(
    transition,
    category,
    question,
    user_response,
    user_language,
    tokens_for_ai,
    username,
    json_metadata,
    json_new_metadata,
):
    feedback = ""
    if "ai_feedback" in transition:
        tokens_for_ai += f" You must provide the feedback in the user's language: {user_language}. {transition['ai_feedback'].get('tokens_for_ai', '')}."
        ai_feedback = generate_ai_feedback(
            category,
            question,
            user_response,
            tokens_for_ai,
            username,
            json_metadata,
            json_new_metadata,
        )
        feedback += f"\n\nAI Feedback: {ai_feedback}"

    return feedback


def translate_text(text, target_language):
    # Guard clause for default language
    if target_language.lower() == "english":
        return text

    openai_client, model_name = get_openai_client_and_model()
    messages = [
        {
            "role": "system",
            "content": f"Translate the following text to {target_language}.",
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    try:
        completion = openai_client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=2000, temperature=0.7
        )
        translation = completion.choices[0].message.content.strip()
        return translation
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", help="AWS profile name", default=None)
    parser.add_argument(
        "--local-activities",
        action="store_true",
        help="Use local activity files instead of S3",
    )
    args = parser.parse_args()

    # Set profile_name as a global attribute of the app object
    app.config["PROFILE_NAME"] = args.profile
    app.config["LOCAL_ACTIVITIES"] = args.local_activities

    socketio.run(app, host="0.0.0.0", port=5001, use_reloader=True)
