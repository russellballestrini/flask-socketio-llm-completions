from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit, join_room

import eventlet

from openai import OpenAI

import tiktoken

import os
import time

import boto3
import json

from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config["SECRET_KEY"] = "your_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chat.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

socketio = SocketIO(app, async_mode="eventlet")

# Global dictionary to keep track of cancellation requests
cancellation_requests = {}


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
        if command.startswith("/s3 ls"):
            # Extract the S3 file path pattern
            s3_file_path_pattern = command.split(" ", 2)[2]
            # List files from S3 and emit their names
            eventlet.spawn(
                list_s3_files, room.name, s3_file_path_pattern, data["username"]
            )
        if command.startswith("/s3 load"):
            # Extract the S3 file path
            s3_file_path = command.split(" ", 2)[2]
            # Load the file from S3 and emit its content
            eventlet.spawn(load_s3_file, room_name, s3_file_path, data["username"])
        if command.startswith("/s3 save"):
            # Extract the S3 key path
            s3_key_path = command.split(" ", 2)[2]
            # Save the most recent code block to S3
            eventlet.spawn(
                save_code_block_to_s3, room_name, s3_key_path, data["username"]
            )
        if command.startswith("/title new"):
            eventlet.spawn(generate_new_title, room_name, data["username"])
        if command.startswith("/cancel"):
            # Cancel the most recent generation request
            eventlet.spawn(cancel_generation, room_name, data["username"])

    if "dall-e-3" in data["message"]:
        # Use the entire message as the prompt for DALL-E 3
        # Generate the image and emit its URL
        eventlet.spawn(
            generate_dalle_image, data["room_name"], data["message"], data["username"]
        )

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
        if msg.is_base64_image():
            continue
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
    if app.config.get("PROFILE_NAME"):
        session = boto3.Session(profile_name=app.config["PROFILE_NAME"])
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


def chat_gpt(username, room_name, message, model_name="gpt-3.5-turbo"):
    openai_client = OpenAI()
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
            # Emit an event to update this rooms title in the sidebar for all users.
            updated_room_data = {"id": room.id, "name": room.name, "title": room.title}
            socketio.emit("update_room_list", updated_room_data, room=None)

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
            if not msg.is_base64_image()
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

    try:
        chunks = openai_client.chat.completions.create(
            model=model_name, messages=chat_history, temperature=0, stream=True
        )
    except Exception as e:
        with app.app_context():
            message_content = f"OpenAi Error: {e}"
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


def gpt_generate_room_title(messages, model_name):
    """
    Generate a title for the room based on a list of messages.
    """
    openai_client = OpenAI()

    def is_base64_image(content):
        return '<img src="data:image/jpeg;base64,' in content

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
        if not is_base64_image(msg.content)
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
            .limit(100)  # Adjust the limit as needed
            .all()
        )

        # Generate the title using the messages
        new_title = gpt_generate_room_title(last_messages, "gpt-4-1106-preview")

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
    s3_client = boto3.client("s3")

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
    s3_client = boto3.client("s3")

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
    s3_client = boto3.client("s3")

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


def cancel_generation(room_name, username):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", help="AWS profile name", default=None)
    args = parser.parse_args()

    # Set profile_name as a global attribute of the app object
    app.config["PROFILE_NAME"] = args.profile

    socketio.run(app, host="0.0.0.0", port=5001)
