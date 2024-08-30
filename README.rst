flask-socketio-llm-completions
========================================

This project is a chatroom application that allows users to join different chat rooms, send messages, and interact with multiple language models in real-time. The backend is built with Flask and Flask-SocketIO for real-time web communication, while the frontend uses HTML, CSS, and JavaScript to provide an interactive user interface.

To view a short video of the chat in action click this screenshot:

.. image:: flask-socketio-llm-completions-2.png
    :alt: youtube video link image
    :target: https://www.youtube.com/watch?v=pd3shNtSojY
    :align: center

Features
--------

- Real-time messaging between users in a chatroom.
- Ability to join different chatrooms with unique URLs.
- Integration with language models for generating room titles and processing messages.
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
- mistralai (for interacting with MistralAI's language models)
- together (for interacting with together.ai language models)
- groq (for interacting with very fast groq language models)

Installation
------------

To set up the project, follow these steps:

1. Clone this repository::

    git clone https://github.com/russellballestrini/flask-socketio-llm-completions.git
    cd flask-socketio-llm-completions

2. Create a virtual environment and activate it::

    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`

3. Install the required dependencies::

    pip install -r requirements.txt

4. Initialize the database:

   Before running the application for the first time, you need to create the database and tables, and then stamp the Alembic migrations to mark them as up to date. Follow these steps::

        python init_db.py
        flask db stamp head

Usage
-----

Set up optional environment variables for your AWS, OpenAI, MistralAI, or together.ai API keys::

    export AWS_ACCESS_KEY_ID="your_access_key"
    export AWS_SECRET_ACCESS_KEY="your_secret_key"
    export S3_BUCKET_NAME="your_s3_bucket_name"
    export OPENAI_API_KEY="your_openai_api_key"
    export MISTRAL_API_KEY="your_mistralai_api_key"
    export TOGETHER_API_KEY="your_togetherai_api_key"
    export GROQ_API_KEY="your_groq_api_key"
    export VLLM_API_KEY="not-needed"
    export VLLM_ENDPOINT="http://localhost:18888/v1"

To start the application with socket.io run::

    python app.py

Optionally flags ``python app.py --local-activities --profile <aws-profile-name>``::

    usage: app.py [-h] [--profile PROFILE] [--local-activities]
    
    options:
      -h, --help          show this help message and exit
      --profile PROFILE   AWS profile name
      --local-activities  Use local activity files instead of S3


The application will be available at ``http://127.0.0.1:5001`` by default.


Interacting with Language Models
--------------------------------

To interact with the various language models, you can use the following commands within the chat:

- For GPT-3, send a message with ``gpt-3`` and include your prompt.
- For GPT-4o, send a message with ``gpt-4`` and include your prompt.
- For GPT-4o cheapest version, send a message with ``gpt-4o-2024-08-06`` and include your prompt.
- For GPT-4o-mini, send a message with ``gpt-mini`` and include your prompt.
- For Claude-haiku, send a message with ``claude-haiku`` and include your prompt.
- For Claude-sonnet, send a message with ``claude-sonnet`` and include your prompt.
- For Claude-opus, send a message with ``claude-opus`` and include your prompt.
- For Mistral-tiny, send a message with ``mistral-tiny`` and include your prompt.
- For Mistral-small, send a message with ``mistral-small`` and include your prompt.
- For Mistral-medium, send a message with ``mistral-medium`` and include your prompt.
- For Mistral-medium, send a message with ``mistral-large`` and include your prompt.
- For Together OpenChat, send a message with ``together/openchat`` and include your prompt.
- For Together Mistral, send a message with ``together/mistral`` and include your prompt.
- For Together Mixtral, send a message with ``together/mixtral`` and include your prompt.
- For Together Solar, send a message with ``together/solar`` and include your prompt.
- For Groq Mixtral, send a message with ``groq/mixtral`` and include your prompt.
- For Groq Llama-2, send a message with ``groq/llama2`` and include your prompt.
- For Groq Llama-3, send a message with ``groq/llama3`` and include your prompt.
- For Groq Gemma, send a message with ``groq/gemma`` and include your prompt.
- For vLLM Hermes, send a message with ``vllm/hermes-llama-3`` and include your prompt.
- For Dall-e-3, send a message with ``dall-e-3`` and include your prompt.

The system will process your message and provide a response from the selected language model.

Commands
--------

The application supports special commands for interacting with the chatroom:

- ``/s3 load <file_path>``: Loads a file from S3 and displays its content in the chatroom.
- ``/s3 save <file_path>``: Saves the most recent code block from the chatroom to S3.
- ``/s3 ls <file_s3_path_pattern>``: Lists files from S3 that match the given pattern. Use ``*`` to list all files.
- ``/title new``: Generates a new title which reflects conversation content for the current chatroom using gpt-4.
- ``/cancel``: Cancel the most recent chat completion from streaming into the chatroom.
- ``/python``: Executes the most recent Python code block sent in the chatroom and returns the output or any errors.
- ``/help``: Displays the list of commands and models to choose from.

The ``/s3 ls`` command can be used to list files in the connected S3 bucket. You can specify a pattern to filter the files listed. For example:

- ``/s3 ls *`` will list all files in the bucket.
- ``/s3 ls *.py`` will list all Python files.
- ``/s3 ls README.*`` will list files starting with "README." and any extension.

The command will return the file name, size in bytes, and the last modified timestamp for each file that matches the pattern.

Structure
---------

- ``app.py``: The main Flask application file containing the backend logic.
- ``chat.html``: The HTML template for the chatroom interface.
- ``static/``: Directory for static files like CSS, JavaScript, and images.
- ``templates/``: Directory for HTML templates.
- ``research/``: Guarded AI activities or processes. Example YAMLs.


Activity Mode
--------------

Activity mode is an interactive experience where users can engage with a guided AI to learn and answer questions.

The AI provides feedback based on the user's responses and guides them through different sections and steps of an activity.

This mode is designed to be on the "rails", educational, & engaging.

The server expects to load the YAML file out of the S3 bucket you specify in your environment variables.

1. **Start an Activity**: Use the ``/activity`` command followed by the object path to the activity YAML file to start a new activity.

    ``/activity path-to-activity.yaml``

2. **Display Activity Info**: Use the ``/activity info`` command to display AI information about the current activity, including grading and user performance.

    ``/activity info``

3. **Display Activity Metadata**: Use the ``/activity metadata`` command to display metadata information collected about the activity.

    ``/activity metadata``

4. **Cancel an Activity**: Use the ``/activity cancel`` command to display cancel the current activity running in the room.

    ``/activity cancel``


5. **Battleship example**:

    ``/activity research/activity29-battleship.yaml``

    .. image:: flask-socketio-llm-completions-battleship.png
        :align: center


Contributing
------------

Contributions to this project are welcome. Please follow the standard fork and pull request workflow.

License
-------

This project is public domain. It is free for use and distribution without any restrictions.
