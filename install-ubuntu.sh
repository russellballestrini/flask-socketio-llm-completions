#!/bin/bash

# Exit on any error
set -e

# Update the package list
sudo apt update

# Install git, python3-pip, and python3-venv if they are not already installed
sudo apt install -y git python3-pip python3.10-venv sqlite3

# Clone the repository
if [ ! -d "/opt/flask-socketio-llm-completions" ]; then
    sudo git clone https://github.com/russellballestrini/flask-socketio-llm-completions.git /opt/flask-socketio-llm-completions
    touch /opt/flask-socketio-llm-completions/.flaskenv
else
    echo "The directory /opt/flask-socketio-llm-completions already exists."
fi

# Navigate to the repository directory
cd /opt/flask-socketio-llm-completions

# Create a Python virtual environment
if [ ! -d "env" ]; then
    python3 -m venv env
else
    echo "The virtual environment already exists."
fi

# Activate the virtual environment
source env/bin/activate

# Install the Python dependencies from requirements.txt
pip install -r requirements.txt

# Define the service name
SERVICE_NAME="flask-socketio-llm-completions"

# Create a systemd service file
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"

# Check if the service file already exists
if [ ! -f "$SERVICE_FILE" ]; then
    # Create the service file with the following content
    sudo bash -c "cat > $SERVICE_FILE" << EOF
[Unit]
Description=Flask SocketIO LLM Completions Service
After=network.target

[Service]
User=$(whoami)
Group=$(whoami)
WorkingDirectory=/opt/flask-socketio-llm-completions
Environment="PATH=/opt/flask-socketio-llm-completions/env/bin"
ExecStart=/opt/flask-socketio-llm-completions/env/bin/python app.py
EnvironmentFile=/opt/flask-socketio-llm-completions/.flaskenv

Restart=always

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd to read the new service file
    sudo systemctl daemon-reload

    # Enable the service to start on boot
    sudo systemctl enable $SERVICE_NAME

    # Start the service
    sudo systemctl start $SERVICE_NAME

    echo "Service $SERVICE_NAME has been installed and started."
else
    echo "Service file $SERVICE_FILE already exists."
fi

# Initialize the database
python init_db.py

# Stamp the database with the Alembic head revision
FLASK_APP=app.py flask db stamp head

# Deactivate the virtual environment
deactivate

echo "Setup and service configuration completed successfully."
