#!/bin/bash

# Exit on any error
set -e

# Application user, group, and installation path
APP_USER="flaskapp"
APP_GROUP="flaskapp"
INSTALL_PATH="/opt/flask-socketio-llm-completions"

# Update the package list
apt update

# Install git, python3-pip, and python3-venv if they are not already installed
apt install -y git python3-pip python3.10-venv sqlite3

# Create a user and group for the application if they don't exist
if ! id "$APP_USER" &>/dev/null; then
    useradd --system --create-home --shell /bin/false $APP_USER
fi

# Clone the repository
if [ ! -d "$INSTALL_PATH" ]; then
    git clone https://github.com/russellballestrini/flask-socketio-llm-completions.git $INSTALL_PATH
    touch $INSTALL_PATH/.flaskenv
else
    echo "The directory $INSTALL_PATH already exists."
fi

# Change ownership of the directory to the application user
chown -R $APP_USER:$APP_GROUP $INSTALL_PATH

# Navigate to the repository directory
cd $INSTALL_PATH

# Create a Python virtual environment
if [ ! -d "env" ]; then
    python3 -m venv env
    # Change ownership of the virtual environment to the application user
    chown -R $APP_USER:$APP_GROUP env
fi

# Activate the virtual environment and install the Python dependencies as the application user
sudo -u $APP_USER /bin/bash -c "source $INSTALL_PATH/env/bin/activate && pip install -r $INSTALL_PATH/requirements.txt"

# Define the service name
SERVICE_NAME="flask-socketio-llm-completions"

# Create a systemd service file
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"

# Check if the service file already exists
if [ ! -f "$SERVICE_FILE" ]; then
    # Create the service file with the following content
    cat > $SERVICE_FILE << EOF
[Unit]
Description=Flask SocketIO LLM Completions Service
After=network.target

[Service]
User=$APP_USER
Group=$APP_GROUP
WorkingDirectory=$INSTALL_PATH
Environment="PATH=$INSTALL_PATH/env/bin"
ExecStart=$INSTALL_PATH/env/bin/python app.py
EnvironmentFile=$INSTALL_PATH/.flaskenv

Restart=always

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd to read the new service file
    systemctl daemon-reload

    # Enable the service to start on boot
    systemctl enable $SERVICE_NAME

    # Start the service
    systemctl start $SERVICE_NAME

    echo "Service $SERVICE_NAME has been installed and started."
else
    echo "Service file $SERVICE_FILE already exists."
fi

# Initialize the database as the application user within the virtual environment
sudo -u $APP_USER /bin/bash -c "source $INSTALL_PATH/env/bin/activate && python $INSTALL_PATH/init_db.py"

# Stamp the database with the Alembic head revision as the application user within the virtual environment
sudo -u $APP_USER /bin/bash -c "source $INSTALL_PATH/env/bin/activate && FLASK_APP=$INSTALL_PATH/app.py flask db stamp head"

echo "Setup and service configuration completed successfully."
