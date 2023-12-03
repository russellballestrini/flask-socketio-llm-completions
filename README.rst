flask-socketio-llm-completions
========================================

This project is a chatroom application that allows users to join different chat rooms, send messages, and interact with multiple language models in real-time. The backend is built with Flask and Flask-SocketIO for real-time web communication, while the frontend uses HTML, CSS, and JavaScript to provide an interactive user interface.

Features
--------

- Real-time messaging between users in a chatroom.
- Ability to join different chatrooms with unique URLs.
- Integration with OpenAI's language models for generating room titles and processing messages.
- Syntax highlighting for code blocks within messages.
- Markdown rendering for messages.
- Commands to load and save code blocks to AWS S3.
- Database storage for messages and chatrooms using SQLAlchemy.
- Migration support with Flask-Migrate.

Requirements
------------

- Python 3.6+
- Flask
- Flask-SocketIO
- Flask-SQLAlchemy
- Flask-Migrate
- eventlet or gevent
- boto3 (for interacting with AWS Bedrock currently Claude, and S3 access)
- openai (for interacting with OpenAI's language models)

Installation
------------

To set up the project, follow these steps:

1. Clone this repository::

    git clone https://github.com/your-username/flask-socketio-llm-completions.git
    cd flask-socketio-llm-completions

2. Create a virtual environment and activate it::

    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required dependencies::

    pip install -r requirements.txt

4. Set up environment variables for your AWS credentials and OpenAI API key::

    export AWS_ACCESS_KEY_ID="your_access_key"
    export AWS_SECRET_ACCESS_KEY="your_secret_key"
    export S3_BUCKET_NAME="your_s3_bucket_name"
    export OPENAI_API_KEY="your_openai_api_key"

5. Initialize the database:

   Before running the application for the first time, you need to create the database and tables, and then stamp the Alembic migrations to mark them as up to date. Follow these steps::

        python init_db.py
        flask db stamp head

Usage
-----

To start the application with socket.io run::

    python app.py

The application will be available at ``http://127.0.0.1:5001`` by default.

Interacting with Language Models
--------------------------------

To interact with the various language models, you can use the following commands within the chat:

- For GPT-3, send a message with ``gpt-3`` including your prompt.
- For GPT-4, send a message with ``gpt-4`` including your prompt.
- For Claude-v1, send a message with ``claude-v1`` including your prompt.
- For Claude-v2, send a message with ``claude-v2`` including your prompt.

The system will process your message and provide a response from the selected language model.

Structure
---------

- ``app.py``: The main Flask application file containing the backend logic.
- ``chat.html``: The HTML template for the chatroom interface.
- ``static/``: Directory for static files like CSS, JavaScript, and images.
- ``templates/``: Directory for HTML templates.

Commands
--------

The application supports special commands for interacting with AWS S3:

- ``/s3 load <file_path>``: Loads a file from S3 and displays its content in the chatroom.
- ``/s3 save <file_path>``: Saves the most recent code block from the chatroom to S3.

Contributing
------------

Contributions to this project are welcome. Please follow the standard fork and pull request workflow.

License
-------

This project is public domain. It is free for use and distribution without any restrictions.
