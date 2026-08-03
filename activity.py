import json
import yaml
import os
import random
import gevent

from flask import request
from sqlalchemy.exc import InvalidRequestError

# Import app, socketio, and db from the main module
# These will be set via import when this module is imported by app.py
app = None
socketio = None
db = None

# Import models
from models import Room, Message, ActivityState

# Import helper functions from app.py
# These will be imported when this module is loaded
get_room = None
get_s3_client = None
get_openai_client_and_model = None

# Import SYSTEM_USERS from app.py
SYSTEM_USERS = None

# Import activity utilities for v2.0 features
from activity_utils import (
    render_template,
    evaluate_condition,
    check_conditions,
    filter_content_blocks,
    resolve_conditional_navigation,
    select_weighted_random,
    get_progressive_hint,
    create_template_context,
    create_completion_skip_thinking,
    strip_reasoning,
)


def handle_get_activity_status(data):
    """Get the current activity status for a room."""
    room_name = data["room_name"]
    room = get_room(room_name)

    if room:
        activity_state = ActivityState.query.filter_by(room_id=room.id).first()

        if activity_state:
            socketio.emit(
                "activity_status",
                {
                    "active": True,
                    "activity_name": activity_state.s3_file_path,
                    "section_id": activity_state.section_id,
                    "step_id": activity_state.step_id,
                },
                room=request.sid,
            )
        else:
            socketio.emit("activity_status", {"active": False}, room=request.sid)


def get_activity_content(file_path):
    """
    Load the activity content from either S3 or the local filesystem based on the configuration.
    """
    if app.config["LOCAL_ACTIVITIES"]:
        # Load the activity YAML from a local file with path traversal protection
        import os.path

        # Normalize the path and ensure it's within the research directory
        normalized_path = os.path.normpath(file_path)

        # Ensure path doesn't contain dangerous patterns
        if ".." in normalized_path or normalized_path.startswith("/"):
            raise ValueError(f"Invalid file path: {file_path}")

        # Ensure file is within research directory and has .yaml extension
        if not normalized_path.startswith("research/") or not normalized_path.endswith(
            ".yaml"
        ):
            raise ValueError(
                f"File must be in research/ directory and end with .yaml: {file_path}"
            )

        # Additional safety check - ensure resolved path is still in research dir
        full_path = os.path.abspath(normalized_path)
        research_dir = os.path.abspath("research/")
        if not full_path.startswith(research_dir):
            raise ValueError(f"Path traversal attempt detected: {file_path}")

        with open(normalized_path, "r") as file:
            activity_yaml = file.read()
    else:
        # Load the activity YAML from S3
        s3_client = get_s3_client()
        bucket_name = os.environ.get("S3_BUCKET_NAME")
        response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
        activity_yaml = response["Body"].read().decode("utf-8")

    return yaml.safe_load(activity_yaml)


def loop_through_steps_until_question(
    activity_content,
    activity_state,
    room_name,
    username,
    classifier_model="MODEL_0",
    feedback_model="MODEL_0",
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
        if "content_blocks" in step:
            # Create template context
            context = create_template_context(
                metadata=activity_state.dict_metadata,
                current_attempt=activity_state.attempts,
                max_attempts=activity_state.max_attempts,
                current_section=current_section_id,
                current_step=current_step_id,
                username=username,
            )

            # Filter and render content blocks (supports conditional blocks and templates)
            filtered_blocks = filter_content_blocks(
                step["content_blocks"], activity_state.dict_metadata, context
            )

            if filtered_blocks:
                content = "\n\n".join(filtered_blocks)
                translated_content = translate_text(
                    content, user_language, feedback_model
                )
                new_message = Message(
                    username="System", content=translated_content, room_id=room.id
                )
                db.session.add(new_message)
                db.session.commit()

                socketio.emit(
                    "chat_message",
                    {
                        "id": new_message.id,
                        "username": "System",
                        "content": translated_content,
                    },
                    room=room_name,
                )
                socketio.sleep(0.1)

        # Check if the current step has a question
        if "question" in step:
            # Create template context
            context = create_template_context(
                metadata=activity_state.dict_metadata,
                current_attempt=activity_state.attempts,
                max_attempts=activity_state.max_attempts,
                current_section=current_section_id,
                current_step=current_step_id,
                username=username,
            )

            # Render template variables in question
            question_content = render_template(step["question"], context)
            translated_question_content = translate_text(
                question_content, user_language, feedback_model
            )
            new_message = Message(
                username="System (Question)",
                content=translated_question_content,
                room_id=room.id,
            )
            db.session.add(new_message)
            db.session.commit()

            socketio.emit(
                "chat_message",
                {
                    "id": new_message.id,
                    "username": "System",
                    "content": translated_question_content,
                },
                room=room_name,
            )
            socketio.sleep(0.1)
            break

        # Move to the next step
        next_section, next_step = get_next_step(
            activity_content, current_section_id, current_step_id
        )

        if next_step:
            activity_state.attempts = 0
            activity_state.section_id = next_section["section_id"]
            activity_state.step_id = next_step["step_id"]

            db.session.add(activity_state)
            db.session.commit()

            current_section_id = next_section["section_id"]
            current_step_id = next_step["step_id"]
        else:
            # Activity completed

            # Display activity info before completing
            display_activity_info(room_name, username, feedback_model)

            db.session.delete(activity_state)
            db.session.commit()
            socketio.emit(
                "chat_message",
                {
                    "id": None,
                    "username": "System",
                    "content": "Activity completed!",
                },
                room=room_name,
            )
            # Return to activity chooser
            socketio.emit("activity_status", {"active": False}, room=room_name)
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

        # Get model configuration from activity content if specified
        # Default to MODEL_0 (Hermes) for both - fast, accurate, and always available
        classifier_model = activity_content.get("classifier_model", "MODEL_0")
        feedback_model = activity_content.get("feedback_model", "MODEL_0")

        # Loop through steps until a question is found or the end is reached
        loop_through_steps_until_question(
            activity_content,
            activity_state,
            room_name,
            username,
            classifier_model=classifier_model,
            feedback_model=feedback_model,
        )

        # Emit activity status update
        socketio.emit(
            "activity_status",
            {
                "active": True,
                "activity_name": s3_file_path,
                "section_id": initial_section["section_id"],
                "step_id": initial_step["step_id"],
            },
            room=room_name,
        )


def cancel_activity(room_name, username):
    with app.app_context():
        room = get_room(room_name)
        activity_state = ActivityState.query.filter_by(room_id=room.id).first()

        if not activity_state:
            socketio.emit(
                "chat_message",
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
            "chat_message",
            {
                "id": None,
                "username": "System",
                "content": "Activity has been canceled.",
            },
            room=room_name,
        )

        # Emit activity status update
        socketio.emit("activity_status", {"active": False}, room=room_name)


def display_activity_metadata(room_name, username):
    with app.app_context():
        room = get_room(room_name)
        activity_state = ActivityState.query.filter_by(room_id=room.id).first()

        if not activity_state:
            socketio.emit(
                "chat_message",
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
            "chat_message",
            {
                "id": new_message.id,
                "username": "System",
                "content": metadata_message,
            },
            room=room_name,
        )


def execute_processing_script(metadata, script):
    # Prepare the environment for the script
    # Use the same dict for both globals and locals to support comprehensions
    script_env = {
        "__builtins__": __builtins__,
        "metadata": metadata,
        "script_result": None,
    }

    # Execute the script
    exec(script, script_env, script_env)

    # Return the result from the script
    return script_env["script_result"]


def handle_activity_response(room_name, user_response, username, model="MODEL_0"):
    with app.app_context():
        room = get_room(room_name)
        activity_state = ActivityState.query.filter_by(room_id=room.id).first()

        if not activity_state:
            return

        # Load the activity content
        activity_content = get_activity_content(activity_state.s3_file_path)

        # Get activity-level model defaults
        # Default to MODEL_0 (Hermes) for both - fast, accurate, and always available
        default_classifier_model = activity_content.get("classifier_model", "MODEL_0")
        default_feedback_model = activity_content.get("feedback_model", "MODEL_0")

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

            # Get step-level model overrides (if specified), otherwise use activity defaults
            classifier_model = step.get("classifier_model", default_classifier_model)
            feedback_model = step.get("feedback_model", default_feedback_model)

            feedback_tokens_for_ai = step.get("feedback_tokens_for_ai", "")

            # Check if the step has a question
            if "question" in step:
                # Execute pre-script if it exists (runs before categorization, with user_response available)
                if "pre_script" in step:
                    print(f"DEBUG: Executing pre-script")
                    # Add user_response to a temporary copy of metadata for pre_script
                    temp_metadata = activity_state.dict_metadata.copy()
                    temp_metadata["user_response"] = user_response
                    pre_result = (
                        execute_processing_script(temp_metadata, step["pre_script"])
                        or {}
                    )
                    # Update metadata with pre-script results
                    for key, value in pre_result.get("metadata", {}).items():
                        activity_state.add_metadata(key, value)
                    print(f"DEBUG: Pre-script completed, updated metadata")

                # Roll for random buckets BEFORE categorization
                triggered_random_buckets = []
                if "random_buckets" in step:
                    for bucket_name, config in step["random_buckets"].items():
                        probability = config.get("probability", 0)
                        roll = random.random()
                        if roll < probability:
                            triggered_random_buckets.append(bucket_name)
                            socketio.emit(
                                "chat_message",
                                {
                                    "id": None,
                                    "username": "System",
                                    "content": f"🎲 [RANDOM EVENT] '{bucket_name}' triggered!",
                                },
                                room=room_name,
                            )
                            socketio.sleep(0.05)

                # Categorize the user's response
                category = categorize_response(
                    step["question"],
                    user_response,
                    step["buckets"],
                    step.get("tokens_for_ai", ""),
                    classifier_model,
                )

                # Emit the category to the frontend
                socketio.emit(
                    "chat_message",
                    {
                        "id": None,
                        "username": "System",
                        "content": f"Category: {category}",
                    },
                    room=room_name,
                )
                socketio.sleep(0.1)

                # Combine user's category with triggered random buckets
                # User's response is processed FIRST, then random events
                all_active_buckets = [category] + triggered_random_buckets

                # Find transitions for all active buckets
                active_transitions = []
                for bucket in all_active_buckets:
                    transition = None
                    if bucket in step["transitions"]:
                        transition = step["transitions"][bucket]
                    elif str(bucket).isdigit() and int(bucket) in step["transitions"]:
                        transition = step["transitions"][int(bucket)]
                    else:
                        # Try boolean conversion
                        if str(bucket).lower() in ["yes", "true"]:
                            bucket = True
                        elif str(bucket).lower() in ["no", "false"]:
                            bucket = False
                        if bucket in step["transitions"]:
                            transition = step["transitions"][bucket]

                    if transition:
                        active_transitions.append((bucket, transition))

                # Error only if NO transitions found at all
                if not active_transitions:
                    socketio.emit(
                        "chat_message",
                        {
                            "id": None,
                            "username": "System",
                            "content": f"Error: Unrecognized category '{category}'. Please try again.",
                        },
                        room=room_name,
                    )
                    return

                # Track temporary metadata keys across all transitions
                metadata_tmp_keys = []

                # Track the final navigation target (use LAST transition's next_section_and_step)
                final_next_section_and_step = None

                # Track counts_as_attempt (if ANY transition counts, it counts)
                any_counts_as_attempt = False

                # Process ALL active transitions in order
                for bucket_name, transition in active_transitions:
                    # Emit separator between buckets (but not for the first one)
                    if bucket_name != all_active_buckets[0]:
                        socketio.emit(
                            "chat_message",
                            {
                                "id": None,
                                "username": "System",
                                "content": f"\n{'='*60}\nProcessing transition for bucket: '{bucket_name}'\n{'='*60}",
                            },
                            room=room_name,
                        )
                        socketio.sleep(0.05)

                    # Check metadata conditions for the current step (v2.0 advanced conditions)
                    if "metadata_conditions" in transition:
                        conditions_met = check_conditions(
                            activity_state.dict_metadata,
                            transition["metadata_conditions"],
                        )
                        if not conditions_met:
                            # Skip this transition if conditions not met
                            socketio.emit(
                                "chat_message",
                                {
                                    "id": None,
                                    "username": "System",
                                    "content": f"Skipping '{bucket_name}' - metadata conditions not met",
                                },
                                room=room_name,
                            )
                            socketio.sleep(0.05)
                            continue

                    # this gives the llm context on what changed.
                    new_metadata = {}

                    # Update metadata based on user actions
                    if "metadata_add" in transition:
                        for key, value in transition["metadata_add"].items():
                            if value == "the-users-response":
                                value = user_response
                            elif value == "the-llms-response":
                                continue
                            elif isinstance(value, str):
                                if value.startswith("n+random(") and value.endswith(
                                    ")"
                                ):
                                    # Extract the range and apply the random increment
                                    range_values = value[9:-1].split(",")
                                    if len(range_values) == 2:
                                        x, y = map(int, range_values)
                                        value = activity_state.dict_metadata.get(
                                            key, 0
                                        ) + random.randint(x, y)
                                elif value.startswith("n+") or value.startswith("n-"):
                                    # Check if this is string concatenation (n+,value) or numeric operation (n+5)
                                    if value.startswith("n+,") or value.startswith(
                                        "n-,"
                                    ):
                                        # String concatenation: append/remove from existing value
                                        operation = value[:2]  # "n+" or "n-"
                                        suffix = value[
                                            3:
                                        ]  # Everything after "n+," or "n-,"
                                        existing_value = (
                                            activity_state.dict_metadata.get(key, "")
                                        )
                                        if operation == "n+":
                                            # Append with comma separator if existing value is non-empty
                                            if existing_value:
                                                value = f"{existing_value},{suffix}"
                                            else:
                                                value = suffix
                                        elif operation == "n-":
                                            # Remove suffix from existing value
                                            if existing_value:
                                                parts = existing_value.split(",")
                                                parts = [
                                                    p for p in parts if p != suffix
                                                ]
                                                value = ",".join(parts)
                                            else:
                                                value = existing_value
                                    else:
                                        # Numeric operation: extract the numeric part c and apply the operation +/-
                                        try:
                                            c = int(value[2:])
                                            if value.startswith("n+"):
                                                value = (
                                                    activity_state.dict_metadata.get(
                                                        key, 0
                                                    )
                                                    + c
                                                )
                                            elif value.startswith("n-"):
                                                value = (
                                                    activity_state.dict_metadata.get(
                                                        key, 0
                                                    )
                                                    - c
                                                )
                                        except ValueError:
                                            print(
                                                f"Warning: Invalid numeric operation '{value}' for key '{key}'"
                                            )
                            new_metadata[key] = value
                            activity_state.add_metadata(key, value)

                    # Update metadata based on user actions
                    if "metadata_tmp_add" in transition:
                        for key, value in transition["metadata_tmp_add"].items():
                            if value == "the-users-response":
                                value = user_response
                            elif value == "the-llms-response":
                                continue
                            elif isinstance(value, str):
                                if value.startswith("n+random(") and value.endswith(
                                    ")"
                                ):
                                    # Extract the range and apply the random increment
                                    range_values = value[9:-1].split(",")
                                    if len(range_values) == 2:
                                        x, y = map(int, range_values)
                                        value = activity_state.dict_metadata.get(
                                            key, 0
                                        ) + random.randint(x, y)
                                elif value.startswith("n+") or value.startswith("n-"):
                                    # Check if this is string concatenation (n+,value) or numeric operation (n+5)
                                    if value.startswith("n+,") or value.startswith(
                                        "n-,"
                                    ):
                                        # String concatenation: append/remove from existing value
                                        operation = value[:2]  # "n+" or "n-"
                                        suffix = value[
                                            3:
                                        ]  # Everything after "n+," or "n-,"
                                        existing_value = (
                                            activity_state.dict_metadata.get(key, "")
                                        )
                                        if operation == "n+":
                                            # Append with comma separator if existing value is non-empty
                                            if existing_value:
                                                value = f"{existing_value},{suffix}"
                                            else:
                                                value = suffix
                                        elif operation == "n-":
                                            # Remove suffix from existing value
                                            if existing_value:
                                                parts = existing_value.split(",")
                                                parts = [
                                                    p for p in parts if p != suffix
                                                ]
                                                value = ",".join(parts)
                                            else:
                                                value = existing_value
                                    else:
                                        # Numeric operation: extract the numeric part c and apply the operation +/-
                                        try:
                                            c = int(value[2:])
                                            if value.startswith("n+"):
                                                value = (
                                                    activity_state.dict_metadata.get(
                                                        key, 0
                                                    )
                                                    + c
                                                )
                                            elif value.startswith("n-"):
                                                value = (
                                                    activity_state.dict_metadata.get(
                                                        key, 0
                                                    )
                                                    - c
                                                )
                                        except ValueError:
                                            print(
                                                f"Warning: Invalid numeric operation '{value}' for key '{key}'"
                                            )
                            new_metadata[key] = value
                            metadata_tmp_keys.append(key)
                            activity_state.add_metadata(key, value)

                    # Update metadata by appending values to lists
                    if "metadata_append" in transition:
                        for key, value in transition["metadata_append"].items():
                            # Determine the value to append
                            if value == "the-users-response":
                                value_to_append = user_response
                            elif value == "the-llms-response":
                                continue  # Handle this after feedback
                            else:
                                value_to_append = value

                            # Ensure the key exists and is a list
                            current_value = activity_state.dict_metadata.get(key, [])
                            if not isinstance(current_value, list):
                                current_value = [current_value]

                            # Append the value to the list
                            if isinstance(value_to_append, list):
                                current_value.extend(value_to_append)
                            else:
                                current_value.append(value_to_append)

                            # Update the metadata
                            activity_state.add_metadata(key, current_value)

                    # Update temporary metadata by appending values to lists
                    if "metadata_tmp_append" in transition:
                        for key, value in transition["metadata_tmp_append"].items():
                            # Determine the value to append
                            if value == "the-users-response":
                                value_to_append = user_response
                            elif value == "the-llms-response":
                                continue  # Handle this after feedback
                            else:
                                value_to_append = value

                            # Ensure the key exists and is a list
                            current_value = activity_state.dict_metadata.get(key, [])
                            if not isinstance(current_value, list):
                                current_value = [current_value]

                            # Append the value to the list
                            if isinstance(value_to_append, list):
                                current_value.extend(value_to_append)
                            else:
                                current_value.append(value_to_append)

                            # Update the metadata
                            activity_state.add_metadata(key, current_value)

                            # Track temporary metadata keys
                            metadata_tmp_keys.append(key)

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

                    if "metadata_tmp_random" in transition:
                        random_key = random.choice(
                            list(transition["metadata_tmp_random"].keys())
                        )
                        random_value = transition["metadata_tmp_random"][random_key]
                        new_metadata[random_key] = random_value
                        metadata_tmp_keys.append(random_key)
                        activity_state.add_metadata(random_key, random_value)

                    # Handle metadata_weighted_random (v2.0)
                    if "metadata_weighted_random" in transition:
                        for key, weighted_options in transition[
                            "metadata_weighted_random"
                        ].items():
                            selected_value = select_weighted_random(weighted_options)
                            new_metadata[key] = selected_value
                            activity_state.add_metadata(key, selected_value)

                    # Handle metadata_tmp_weighted_random (v2.0)
                    if "metadata_tmp_weighted_random" in transition:
                        for key, weighted_options in transition[
                            "metadata_tmp_weighted_random"
                        ].items():
                            selected_value = select_weighted_random(weighted_options)
                            new_metadata[key] = selected_value
                            metadata_tmp_keys.append(key)
                            activity_state.add_metadata(key, selected_value)

                    # Execute the post-script if it exists (supports both old and new naming)
                    post_script = step.get("post_script") or step.get(
                        "processing_script"
                    )
                    if post_script and (
                        transition.get("run_post_script", False)
                        or transition.get("run_processing_script", False)
                    ):
                        print(f"DEBUG: Executing post-script")
                        result = (
                            execute_processing_script(
                                activity_state.dict_metadata, post_script
                            )
                            or {}
                        )

                        plot_image_base64 = result.pop("plot_image", None)

                        # Add the result to the temporary metadata for use in AI feedback
                        metadata_tmp_keys.append("processing_script_result")
                        activity_state.add_metadata("processing_script_result", result)

                        # Update metadata with results from the processing script
                        for key, value in result.get("metadata", {}).items():
                            activity_state.add_metadata(key, value)

                        # Check if processing script wants to override the transition
                        if "next_section_and_step" in result:
                            final_next_section_and_step = result[
                                "next_section_and_step"
                            ]
                            print(
                                f"DEBUG: Processing script overriding transition to: {final_next_section_and_step}"
                            )

                        # Check if the result contains a plot image
                        if plot_image_base64:
                            plot_image_html = f'<img alt="Plot Image" src="data:image/png;base64,{plot_image_base64}">'

                            if result.get("set_background", False):
                                socketio.emit(
                                    "set_background",
                                    {"image_data": plot_image_base64},
                                    room=room_name,
                                )
                                socketio.sleep(0.1)
                            else:
                                # Save the plot image to the database
                                new_message = Message(
                                    username=username,
                                    content=plot_image_html,
                                    room_id=room.id,
                                )
                                db.session.add(new_message)
                                db.session.commit()

                                # Emit the plot image to the frontend
                                socketio.emit(
                                    "chat_message",
                                    {
                                        "id": new_message.id,
                                        "username": username,
                                        "content": plot_image_html,
                                    },
                                    room=room_name,
                                )
                                socketio.sleep(0.1)

                    if (
                        "metadata_clear" in transition
                        and transition["metadata_clear"] == True
                    ):
                        activity_state.clear_metadata()

                    print(activity_state.dict_metadata)

                    # Commit the changes after processing this transition
                    db.session.add(activity_state)
                    db.session.commit()

                    user_language = activity_state.dict_metadata.get(
                        "language", "English"
                    )

                    # Emit the transition content blocks if they exist (v2.0 with templates & conditions)
                    if "content_blocks" in transition:
                        # Create template context
                        context = create_template_context(
                            metadata=activity_state.dict_metadata,
                            current_attempt=activity_state.attempts,
                            max_attempts=activity_state.max_attempts,
                            current_section=activity_state.section_id,
                            current_step=activity_state.step_id,
                            username=username,
                        )

                        # Filter and render content blocks (supports conditional blocks and templates)
                        filtered_blocks = filter_content_blocks(
                            transition["content_blocks"],
                            activity_state.dict_metadata,
                            context,
                        )

                        if filtered_blocks:
                            transition_content = "\n\n".join(filtered_blocks)
                            translated_transition_content = translate_text(
                                transition_content, user_language, feedback_model
                            )
                            new_message = Message(
                                username="System",
                                content=translated_transition_content,
                                room_id=room.id,
                            )
                            db.session.add(new_message)
                            db.session.commit()

                            socketio.emit(
                                "chat_message",
                                {
                                    "id": new_message.id,
                                    "username": "System",
                                    "content": translated_transition_content,
                                },
                                room=room_name,
                            )
                            socketio.sleep(0.1)

                    # if "correct" or max_attempts reached.
                    # Provide feedback based on the category

                    # Handle feedback systems
                    feedback_messages = []

                    if "feedback_prompts" in step:
                        # New multi-prompt system - pass full metadata, let each prompt filter
                        multi_feedback_messages = provide_feedback_prompts(
                            transition,
                            bucket_name,  # Use bucket_name instead of category
                            step["question"],
                            step["feedback_prompts"],
                            user_response,
                            user_language,
                            username,
                            json.dumps(
                                activity_state.dict_metadata
                            ),  # Pass full metadata
                            json.dumps(new_metadata),
                            feedback_tokens_for_ai,  # Pass legacy tokens to be combined
                            feedback_model,
                        )
                        feedback_messages.extend(multi_feedback_messages)
                    elif feedback_tokens_for_ai:
                        # Legacy single feedback system - use transition-level filtering
                        feedback_metadata = activity_state.dict_metadata
                        if "metadata_feedback_filter" in transition:
                            filter_keys = transition["metadata_feedback_filter"]
                            feedback_metadata = {
                                k: v
                                for k, v in activity_state.dict_metadata.items()
                                if k in filter_keys
                            }

                        feedback = provide_feedback(
                            transition,
                            bucket_name,  # Use bucket_name instead of category
                            step["question"],
                            feedback_tokens_for_ai,
                            user_response,
                            user_language,
                            username,
                            json.dumps(feedback_metadata),
                            json.dumps(new_metadata),
                            feedback_model,
                        )
                        if feedback and feedback.strip():
                            feedback_messages.append(
                                {"name": "Feedback", "content": feedback}
                            )

                    # Store and emit all feedback messages
                    for feedback_msg in feedback_messages:
                        new_message = Message(
                            username=f"System ({feedback_msg['name'].title()})",
                            content=feedback_msg["content"],
                            room_id=room.id,
                        )
                        db.session.add(new_message)
                        db.session.commit()

                        socketio.emit(
                            "chat_message",
                            {
                                "id": new_message.id,
                                "username": f"System ({feedback_msg['name'].title()})",
                                "content": feedback_msg["content"],
                            },
                            room=room_name,
                        )
                        socketio.sleep(0.1)

                        # Add or append the LLM's response to the metadata
                        for key, value in transition.get("metadata_add", {}).items():
                            if value == "the-llms-response":
                                activity_state.add_metadata(key, feedback)

                        for key, value in transition.get("metadata_append", {}).items():
                            if value == "the-llms-response":
                                # Ensure the key exists and is a list
                                current_value = activity_state.dict_metadata.get(
                                    key, []
                                )
                                if not isinstance(current_value, list):
                                    current_value = [current_value]

                                # Append the feedback to the list
                                current_value.append(feedback)
                                activity_state.add_metadata(key, current_value)

                    # Track navigation (LAST transition's next_section_and_step wins)
                    if "next_section_and_step" in transition:
                        final_next_section_and_step = transition[
                            "next_section_and_step"
                        ]

                    # Track counts_as_attempt (if ANY transition counts, it counts)
                    if transition.get("counts_as_attempt", True):
                        any_counts_as_attempt = True

                # End of multi-bucket processing loop

                # Check for progressive hints (v2.0)
                if "hints" in step:
                    context = create_template_context(
                        metadata=activity_state.dict_metadata,
                        current_attempt=activity_state.attempts + 1,  # Next attempt
                        max_attempts=activity_state.max_attempts,
                        current_section=activity_state.section_id,
                        current_step=activity_state.step_id,
                        username=username,
                    )
                    hint = get_progressive_hint(
                        step["hints"], activity_state.attempts + 1, context
                    )
                    if hint:
                        # Display hint
                        translated_hint = translate_text(
                            hint["text"], user_language, feedback_model
                        )
                        new_message = Message(
                            username="System (Hint)",
                            content=translated_hint,
                            room_id=room.id,
                        )
                        db.session.add(new_message)
                        db.session.commit()

                        socketio.emit(
                            "chat_message",
                            {
                                "id": new_message.id,
                                "username": "System (Hint)",
                                "content": translated_hint,
                            },
                            room=room_name,
                        )
                        socketio.sleep(0.1)

                        # If hint doesn't count as attempt, don't increment
                        if not hint["counts_as_attempt"]:
                            any_counts_as_attempt = False

                if (
                    category
                    not in [
                        "partial_understanding",
                        "limited_effort",
                        "asking_clarifying_questions",
                        "set_language",
                        "off_topic",
                        "incorrect",  # Incorrect answers should stay on step and increment attempts
                    ]
                    or activity_state.attempts >= activity_state.max_attempts
                    or final_next_section_and_step  # Use final navigation from last transition
                ):
                    if final_next_section_and_step:
                        # Resolve conditional navigation (v2.0)
                        resolved_navigation = resolve_conditional_navigation(
                            final_next_section_and_step, activity_state.dict_metadata
                        )

                        if resolved_navigation:
                            (
                                current_section_id,
                                current_step_id,
                            ) = resolved_navigation.split(":")
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
                            # No navigation resolved, move to next step
                            next_section, next_step = get_next_step(
                                activity_content, section["section_id"], step["step_id"]
                            )
                    else:
                        # Move to the next step or section
                        next_section, next_step = get_next_step(
                            activity_content, section["section_id"], step["step_id"]
                        )

                    if next_step:
                        activity_state.attempts = 0
                        activity_state.section_id = next_section["section_id"]
                        activity_state.step_id = next_step["step_id"]

                        db.session.add(activity_state)
                        db.session.commit()

                        # Loop through steps until a question is found or the end is reached
                        loop_through_steps_until_question(
                            activity_content,
                            activity_state,
                            room_name,
                            username,
                            classifier_model=classifier_model,
                            feedback_model=feedback_model,
                        )
                else:
                    # the user response is any bucket other than correct.
                    # Count attempt if ANY transition counted
                    if any_counts_as_attempt:
                        activity_state.attempts += 1
                        db.session.add(activity_state)
                        db.session.commit()

                    # Emit the question again (v2.0 with templates)
                    context = create_template_context(
                        metadata=activity_state.dict_metadata,
                        current_attempt=activity_state.attempts,
                        max_attempts=activity_state.max_attempts,
                        current_section=activity_state.section_id,
                        current_step=activity_state.step_id,
                        username=username,
                    )
                    question_content = render_template(step["question"], context)
                    translated_question_content = translate_text(
                        question_content, user_language, feedback_model
                    )
                    new_message = Message(
                        username="System (Question)",
                        content=translated_question_content,
                        room_id=room.id,
                    )
                    db.session.add(new_message)
                    db.session.commit()

                    socketio.emit(
                        "chat_message",
                        {
                            "id": new_message.id,
                            "username": "System",
                            "content": translated_question_content,
                        },
                        room=room_name,
                    )
                    socketio.sleep(0.1)

                # Check if the activity state still exists before removing temporary metadata
                try:
                    # Remove temporary metadata at the end of the turn
                    for key in metadata_tmp_keys:
                        activity_state.remove_metadata(key)

                    # Commit the changes after removing temporary metadata
                    db.session.add(activity_state)
                    db.session.commit()

                except InvalidRequestError:
                    # Handle the case where the activity state was deleted
                    # print("Activity state was deleted before commit.")
                    db.session.rollback()

            else:
                # Handle steps without a question
                loop_through_steps_until_question(
                    activity_content,
                    activity_state,
                    room_name,
                    username,
                    classifier_model=classifier_model,
                    feedback_model=feedback_model,
                )

        except Exception as e:
            import traceback

            msg = traceback.format_exc()
            socketio.emit(
                "chat_message",
                {
                    "id": None,
                    "username": "System",
                    "content": f"Error processing activity response: {e}\n\n{msg}",
                },
                room=room_name,
            )


def display_activity_info(room_name, username, model="MODEL_0"):
    with app.app_context():
        room = get_room(room_name)
        activity_state = ActivityState.query.filter_by(room_id=room.id).first()

        if not activity_state:
            socketio.emit(
                "chat_message",
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
                    "role": "system" if msg.username in SYSTEM_USERS else "user",
                    "username": msg.username,
                    "content": msg.content,
                }
                for msg in all_messages
                if not msg.is_base64_image()
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
            grading_message = generate_grading(chat_history, rubric, model)

            # Store and emit the activity info
            info_message = f"Activity Info:\nCurrent Section: {activity_state.section_id}\nCurrent Step: {activity_state.step_id}\nAttempts: {activity_state.attempts}\n\n{grading_message}"
            new_message = Message(
                username="System", content=info_message, room_id=room.id
            )
            db.session.add(new_message)
            db.session.commit()

            socketio.emit(
                "chat_message",
                {
                    "id": new_message.id,
                    "username": "System",
                    "content": info_message,
                },
                room=room_name,
            )

        except Exception as e:
            socketio.emit(
                "chat_message",
                {
                    "id": None,
                    "username": "System",
                    "content": f"Error displaying activity info: {e}",
                },
                room=room_name,
            )
            # Debugging: Log exception
            print(f"Exception: {e}")


def generate_grading(chat_history, rubric, model="MODEL_0"):
    # Use provided model or fall back to default
    if model and model != "None":
        openai_client, model_name = get_openai_client_and_model(model)
    else:
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
        completion = create_completion_skip_thinking(
            openai_client,
            model=model_name,
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            n=1,
        )
        grading = strip_reasoning(completion.choices[0].message.content.strip())
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
def categorize_response(question, response, buckets, tokens_for_ai, model="MODEL_0"):
    # Use provided model or fall back to default
    if model and model != "None":
        openai_client, model_name = get_openai_client_and_model(model)
    else:
        openai_client, model_name = get_openai_client_and_model()
    bucket_list = ", ".join([str(bucket) for bucket in buckets])
    # Check if tokens_for_ai already includes format instructions (ANALYSIS/BUCKET format)
    if "ANALYSIS:" in tokens_for_ai and "BUCKET:" in tokens_for_ai:
        # YAML already specifies output format, don't override
        system_content = f"{tokens_for_ai}"
        user_content = f"Question: {question}\nResponse: {response}"
    else:
        # Use old simple format for backwards compatibility
        system_content = f"{tokens_for_ai} Categorize the following response into one of the following buckets: {bucket_list}. Return ONLY a bucket label."
        user_content = f"Question: {question}\nResponse: {response}\n\nCategory:"

    messages = [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    try:
        completion = create_completion_skip_thinking(
            openai_client,
            model=model_name,
            messages=messages,
            n=1,
            max_tokens=150,  # Increased for ANALYSIS + BUCKET format
            temperature=0,
        )
        full_response = strip_reasoning(completion.choices[0].message.content.strip())
        print(f"DEBUG BUCKET CATEGORIZATION: Full Hermes response: {full_response}")

        # Handle both ANALYSIS/BUCKET format and simple bucket response
        if "BUCKET:" in full_response:
            # New ANALYSIS/BUCKET format
            bucket_lines = [
                line for line in full_response.split("\n") if "BUCKET:" in line
            ]
            if bucket_lines:
                category = (
                    bucket_lines[0]
                    .split("BUCKET:")[1]
                    .strip()
                    .lower()
                    .replace(" ", "_")
                )
            else:
                category = full_response.lower().replace(" ", "_")
        elif "ANALYSIS:" in full_response:
            # Has analysis but no explicit BUCKET: line, try to extract from end
            lines = [line.strip() for line in full_response.split("\n") if line.strip()]
            if lines:
                category = lines[-1].lower().replace(" ", "_")
            else:
                category = full_response.lower().replace(" ", "_")
        else:
            # Simple bucket response (old format)
            category = full_response.lower().replace(" ", "_")

        print(f"DEBUG BUCKET CATEGORIZATION: Extracted category: {category}")
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
    model="MODEL_0",
):
    # Use provided model or fall back to default
    if model and model != "None":
        openai_client, model_name = get_openai_client_and_model(model)
    else:
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
        completion = create_completion_skip_thinking(
            openai_client,
            model=model_name,
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            n=1,
        )
        feedback = strip_reasoning(completion.choices[0].message.content.strip())
        return feedback
    except Exception as e:
        return f"Error: {e}"


def provide_feedback(
    transition,
    category,
    question,
    tokens_for_ai,
    user_response,
    user_language,
    username,
    json_metadata,
    json_new_metadata,
    model="MODEL_0",
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
            model,
        )
        feedback += f"\n\n{ai_feedback}"

    return feedback


def provide_feedback_prompts(
    transition,
    category,
    question,
    feedback_prompts,
    user_response,
    user_language,
    username,
    json_metadata,
    json_new_metadata,
    legacy_tokens_for_ai="",
    model="MODEL_0",
):
    """Generate feedback from multiple prompts"""
    feedback_messages = []

    # Parse full metadata once for filtering
    full_metadata = json.loads(json_metadata)

    # Add user_response to metadata for filtering purposes
    full_metadata["user_response"] = user_response

    for prompt in feedback_prompts:
        prompt_name = prompt.get("name", "unnamed")
        tokens_for_ai = prompt.get("tokens_for_ai", "")

        # Apply per-prompt metadata filtering if specified
        prompt_metadata = full_metadata
        if "metadata_filter" in prompt:
            filter_keys = prompt["metadata_filter"]
            prompt_metadata = {
                k: v for k, v in full_metadata.items() if k in filter_keys
            }

            # Check skip condition if specified
            skip_condition = prompt.get("skip_condition")
            if skip_condition:
                should_skip = False
                values = list(prompt_metadata.values())

                if skip_condition == "all_null":
                    should_skip = all(
                        value is None or value == "" or value == "None"
                        for value in values
                    )
                elif skip_condition == "all_false":
                    should_skip = all(
                        value is False or value == "False" for value in values
                    )
                elif skip_condition == "all_true":
                    should_skip = all(
                        value is True or value == "True" for value in values
                    )

                if should_skip:
                    print(
                        f"DEBUG: Skipping prompt '{prompt_name}' - skip_condition '{skip_condition}' met"
                    )
                    continue

            # Special debug for Ship Status and Game Over
            if prompt_name == "Ship Status":
                print(f"DEBUG SHIP STATUS - filter_keys: {filter_keys}")
                print(f"DEBUG SHIP STATUS - filtered metadata: {prompt_metadata}")
                print(
                    f"DEBUG SHIP STATUS - user_sunk_ship_this_round = '{prompt_metadata.get('user_sunk_ship_this_round')}'"
                )
                print(
                    f"DEBUG SHIP STATUS - ai_sunk_ship_this_round = '{prompt_metadata.get('ai_sunk_ship_this_round')}'"
                )
            elif prompt_name == "Game Over":
                print(f"DEBUG GAME OVER - filter_keys: {filter_keys}")
                print(f"DEBUG GAME OVER - filtered metadata: {prompt_metadata}")
                print(
                    f"DEBUG GAME OVER - game_over = '{prompt_metadata.get('game_over')}'"
                )
                print(
                    f"DEBUG GAME OVER - user_wins = '{prompt_metadata.get('user_wins')}'"
                )
                print(f"DEBUG GAME OVER - ai_wins = '{prompt_metadata.get('ai_wins')}'")
        else:
            if prompt_name == "Ship Status":
                print(
                    f"DEBUG SHIP STATUS - NO metadata_filter, full metadata: {prompt_metadata}"
                )
            elif prompt_name == "Game Over":
                print(
                    f"DEBUG GAME OVER - NO metadata_filter, full metadata: {prompt_metadata}"
                )

        # Combine legacy tokens with prompt-specific tokens
        if legacy_tokens_for_ai:
            tokens_for_ai = legacy_tokens_for_ai + " " + tokens_for_ai

        # Add language instruction
        tokens_for_ai += (
            f" You must provide the feedback in the user's language: {user_language}."
        )

        # Add transition-specific AI feedback if present
        if "ai_feedback" in transition:
            tokens_for_ai += f" {transition['ai_feedback'].get('tokens_for_ai', '')}"

        # Determine user_response for this prompt based on metadata filtering
        filtered_user_response = user_response
        if (
            "metadata_filter" in prompt
            and "user_response" not in prompt["metadata_filter"]
        ):
            filtered_user_response = ""  # Remove user response if not in filter

        ai_feedback = generate_ai_feedback(
            category,
            question,
            filtered_user_response,
            tokens_for_ai,
            username,
            json.dumps(prompt_metadata),  # Use filtered metadata for this prompt
            json_new_metadata,
            model,
        )

        # Only add feedback if it has content
        if ai_feedback and ai_feedback.strip():
            feedback_messages.append(
                {"name": prompt_name, "content": ai_feedback.strip()}
            )

    return feedback_messages


def translate_text(text, target_language, model="MODEL_0"):
    # Guard clause for default language
    target_language = target_language.lower().split()

    if "english" in target_language:
        return text

    # Use provided model or fall back to default
    if model and model != "None":
        openai_client, model_name = get_openai_client_and_model(model)
    else:
        openai_client, model_name = get_openai_client_and_model()
    messages = [
        {
            "role": "system",
            "content": f"Translate the following text to {target_language}. DO NOT add anything else extra to your translation. It should be as close to word for word the dame but translated. Don't start with 'Set_language:' DO NOT try to solve math questions, translate the text around it and use mathmatical notation like normal.",
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    try:
        completion = create_completion_skip_thinking(
            openai_client,
            model=model_name,
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
            n=1,
        )
        translation = strip_reasoning(completion.choices[0].message.content.strip())
        return translation
    except Exception as e:
        return f"Error: {e}"


def init_activity_module(
    app_instance, socketio_instance, db_instance, helper_functions
):
    """Initialize the activity module with dependencies from app.py"""
    global app, socketio, db
    global get_room, get_s3_client, get_openai_client_and_model
    global SYSTEM_USERS

    app = app_instance
    socketio = socketio_instance
    db = db_instance

    get_room = helper_functions["get_room"]
    get_s3_client = helper_functions["get_s3_client"]
    get_openai_client_and_model = helper_functions["get_openai_client_and_model"]
    SYSTEM_USERS = helper_functions["SYSTEM_USERS"]
