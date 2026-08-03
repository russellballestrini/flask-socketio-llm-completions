import argparse
import yaml
import json
import random
import os
import sys
from openai import OpenAI

# Add parent directory to path to import activity_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import activity utilities for v2.0 features
from activity_utils import (
    render_template,
    check_conditions,
    filter_content_blocks,
    resolve_conditional_navigation,
    select_weighted_random,
    get_progressive_hint,
    create_template_context,
    create_completion_skip_thinking,
    strip_reasoning,
)

# Global model-client mapping
MODEL_CLIENT_MAP = {}


def get_client_for_endpoint(endpoint, api_key):
    """Create OpenAI client for any endpoint"""
    return OpenAI(api_key=api_key, base_url=endpoint)


def initialize_model_map():
    """Initialize the model-client mapping from environment variables"""
    # Load endpoints from environment variables
    for i in range(1000):  # Support up to 1000 endpoints
        endpoint_key = f"MODEL_ENDPOINT_{i}"
        api_key_key = f"MODEL_API_KEY_{i}"

        endpoint = os.getenv(endpoint_key)
        api_key = os.getenv(api_key_key)

        if endpoint and api_key:
            try:
                client = get_client_for_endpoint(endpoint, api_key)
                # Query endpoint for available models
                try:
                    response = client.models.list()
                    model_list = response.data
                    print(
                        f"[DEBUG] {endpoint} returned models: {[m.id for m in model_list]}"
                    )
                    for m in model_list:
                        model_id = m.id
                        if model_id and model_id not in MODEL_CLIENT_MAP:
                            MODEL_CLIENT_MAP[model_id] = (client, endpoint)
                except Exception as e:
                    print(
                        f"Warning: Could not list models for endpoint '{endpoint}': {e}"
                    )
            except Exception as e:
                print(f"Warning: Failed to initialize endpoint {endpoint}: {e}")


def get_openai_client_and_model(model_name=None):
    """Get OpenAI client and model name

    Supports both direct model names and MODEL_X environment variable references.
    If model_name is MODEL_1, MODEL_2, etc., looks up from environment.
    """
    # Handle MODEL_X references
    if model_name and model_name.startswith("MODEL_"):
        # Extract the number from MODEL_X
        try:
            model_num = model_name.split("_")[1]
            endpoint_key = f"MODEL_ENDPOINT_{model_num}"
            api_key_key = f"MODEL_API_KEY_{model_num}"

            endpoint = os.getenv(endpoint_key)
            api_key = os.getenv(api_key_key)

            if endpoint and api_key:
                client = get_client_for_endpoint(endpoint, api_key)

                # Explicit MODEL_NAME_X wins: endpoints like Gemini list dozens
                # of models (some retired) and "first listed" picks wrong.
                explicit_model = os.getenv(f"MODEL_NAME_{model_num}")
                if explicit_model:
                    return client, explicit_model

                # Look up actual model name from MODEL_CLIENT_MAP for this endpoint
                actual_model = None
                for model_id, (registered_client, base_url) in MODEL_CLIENT_MAP.items():
                    if base_url == endpoint:
                        actual_model = model_id
                        break

                if actual_model:
                    return client, actual_model
                else:
                    # Fallback: query endpoint for models if not in map yet
                    try:
                        response = client.models.list()
                        if response.data:
                            actual_model = response.data[0].id
                            print(
                                f"[DEBUG] Using first model from {endpoint}: {actual_model}"
                            )
                            return client, actual_model
                    except Exception as e:
                        print(f"Warning: Could not query models from {endpoint}: {e}")

                    # Final fallback
                    print(
                        f"Warning: No models found for {endpoint}, using 'model' as fallback"
                    )
                    return client, "model"
        except Exception as e:
            print(f"Warning: Failed to load {model_name}: {e}, falling back to default")

    # Default to MODEL_1 (Hermes)
    if not model_name:
        return get_openai_client_and_model("MODEL_1")

    # Try to find client for specific model name
    for stored_model, (client, base_url) in MODEL_CLIENT_MAP.items():
        if model_name in stored_model or stored_model == model_name:
            return client, model_name

    # Fallback to first available client
    if MODEL_CLIENT_MAP:
        client, _ = next(iter(MODEL_CLIENT_MAP.values()))
        return client, model_name

    # Final fallback to environment or default OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "dummy-key")
    endpoint = os.getenv("MODEL_ENDPOINT_0", "https://api.openai.com/v1")

    client = get_client_for_endpoint(endpoint, api_key)
    return client, model_name


# Initialize the model mapping on startup
initialize_model_map()


# Load the YAML activity file
def load_yaml_activity(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


# Categorize the user's response
def categorize_response(question, response, buckets, tokens_for_ai, model="MODEL_1"):
    bucket_list = ", ".join([str(bucket) for bucket in buckets])
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
        client, model_name = get_openai_client_and_model(model)
        completion = create_completion_skip_thinking(
            client,
            model=model_name,
            messages=messages,
            max_tokens=5,
            temperature=0,
        )
        category = (
            strip_reasoning(completion.choices[0].message.content.strip())
            .lower()
            .replace(" ", "_")
        )
        return category
    except Exception as e:
        return f"Error: {e}"


# Generate AI feedback
def generate_ai_feedback(
    category, question, user_response, tokens_for_ai, metadata, model="MODEL_1"
):
    messages = [
        {
            "role": "system",
            "content": f"{tokens_for_ai} Generate a human-readable feedback message based on the following:",
        },
        {
            "role": "user",
            "content": f"Question: {question}\nResponse: {user_response}\nCategory: {category},\nMetadata: {metadata}",
        },
    ]

    try:
        client, model_name = get_openai_client_and_model(model)
        completion = create_completion_skip_thinking(
            client, model=model_name, messages=messages, max_tokens=250, temperature=0.7
        )
        feedback = strip_reasoning(completion.choices[0].message.content.strip())
        return feedback
    except Exception as e:
        return f"Error: {e}"


# Provide feedback based on the category (legacy single feedback system)
def provide_feedback(
    transition,
    category,
    question,
    user_response,
    user_language,
    tokens_for_ai,
    metadata,
    model="MODEL_1",
):
    feedback = ""
    if "ai_feedback" in transition:
        tokens_for_ai += f" Provide the feedback in {user_language}. {transition['ai_feedback'].get('tokens_for_ai', '')}."

        # Filter metadata for feedback if metadata_feedback_filter is specified
        feedback_metadata = metadata
        if "metadata_feedback_filter" in transition:
            filter_keys = transition["metadata_feedback_filter"]
            feedback_metadata = {k: v for k, v in metadata.items() if k in filter_keys}

        ai_feedback = generate_ai_feedback(
            category, question, user_response, tokens_for_ai, feedback_metadata, model
        )
        feedback += f"\n\nAI Feedback: {ai_feedback}"

    return feedback


# Provide feedback using multiple prompts (new system)
def provide_feedback_prompts(
    transition,
    category,
    question,
    feedback_prompts,
    user_response,
    user_language,
    metadata,
    legacy_tokens_for_ai="",
    model="MODEL_1",
):
    """Generate feedback from multiple prompts"""
    feedback_messages = []

    # Add user_response to metadata for filtering purposes
    full_metadata = metadata.copy()
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

        # Combine legacy tokens with prompt-specific tokens
        if legacy_tokens_for_ai:
            tokens_for_ai = legacy_tokens_for_ai + " " + tokens_for_ai

        # Add language instruction
        tokens_for_ai += f" Provide the feedback in {user_language}."

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
            prompt_metadata,
            model,
        )

        # Only add feedback if it has content and isn't exactly the STFU token
        if ai_feedback and ai_feedback.strip() and ai_feedback.strip() != "STFU":
            feedback_messages.append(
                {"name": prompt_name, "content": ai_feedback.strip()}
            )

    return feedback_messages


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


def get_next_section_and_step(activity_content, current_section_id, current_step_id):
    for section in activity_content["sections"]:
        if section["section_id"] == current_section_id:
            for i, step in enumerate(section["steps"]):
                if step["step_id"] == current_step_id:
                    if i + 1 < len(section["steps"]):
                        return section["section_id"], section["steps"][i + 1]["step_id"]
                    else:
                        # Move to the next section
                        next_section_index = (
                            activity_content["sections"].index(section) + 1
                        )
                        if next_section_index < len(activity_content["sections"]):
                            next_section = activity_content["sections"][
                                next_section_index
                            ]
                            return (
                                next_section["section_id"],
                                next_section["steps"][0]["step_id"],
                            )
    return None, None


def translate_text(text, target_language, model="MODEL_1"):
    # Guard clause for default language
    if target_language.lower() == "english":
        return text

    messages = [
        {
            "role": "system",
            "content": f"Translate the following text to {target_language}:",
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    try:
        client, model_name = get_openai_client_and_model(model)
        completion = create_completion_skip_thinking(
            client, model=model_name, messages=messages, max_tokens=500, temperature=0.7
        )
        translation = strip_reasoning(completion.choices[0].message.content.strip())
        return translation
    except Exception as e:
        return f"Error: {e}"


def simulate_activity(yaml_file_path):
    yaml_content = load_yaml_activity(yaml_file_path)
    max_attempts = yaml_content.get("default_max_attempts_per_step", 3)

    # Get activity-level model defaults (default to MODEL_1 - Hermes)
    default_classifier_model = yaml_content.get("classifier_model", "MODEL_1")
    default_feedback_model = yaml_content.get("feedback_model", "MODEL_1")

    current_section_id = yaml_content["sections"][0]["section_id"]
    current_step_id = yaml_content["sections"][0]["steps"][0]["step_id"]

    metadata = {"language": "English"}  # Default language

    while current_section_id and current_step_id:
        print(
            f"\n\nCurrent section: {current_section_id}, Current step: {current_step_id}\n\n"
        )
        section = next(
            (
                s
                for s in yaml_content["sections"]
                if s["section_id"] == current_section_id
            ),
            None,
        )

        step = next(
            (s for s in section["steps"] if s["step_id"] == current_step_id), None
        )

        # Get step-level model overrides (if specified), otherwise use activity defaults
        classifier_model = step.get("classifier_model", default_classifier_model)
        feedback_model = step.get("feedback_model", default_feedback_model)

        # Get the user's language preference from metadata
        user_language = metadata.get("language", "English")

        # Initialize attempts and max_attempts for this step
        attempts = 0
        step_max_attempts = step.get("max_attempts_per_step", max_attempts)

        # Create template context for rendering
        context = create_template_context(
            metadata=metadata,
            current_attempt=attempts,
            max_attempts=step_max_attempts,
            current_section=current_section_id,
            current_step=current_step_id,
            username="User",
        )

        # Translate and print all content blocks once per step (v2.0 with templates & conditionals)
        if "content_blocks" in step:
            # Filter and render content blocks
            filtered_blocks = filter_content_blocks(
                step["content_blocks"], metadata, context
            )
            if filtered_blocks:
                content = "\n\n".join(filtered_blocks)
                translated_content = translate_text(
                    content, user_language, feedback_model
                )
                print(translated_content)

        # Skip classification and feedback if there's no question
        if "question" not in step:
            current_section_id, current_step_id = get_next_section_and_step(
                yaml_content, current_section_id, current_step_id
            )
            continue

        # Render template variables in question (v2.0)
        question = render_template(step["question"], context)
        translated_question = translate_text(question, user_language, feedback_model)
        print(f"\nQuestion: {translated_question}")

        while attempts < step_max_attempts:
            # Update context with current attempt
            context = create_template_context(
                metadata=metadata,
                current_attempt=attempts + 1,  # 1-indexed for display
                max_attempts=step_max_attempts,
                current_section=current_section_id,
                current_step=current_step_id,
                username="User",
            )

            user_response = input("\nYour Response: ")

            # Roll for random buckets BEFORE categorization
            triggered_random_buckets = []
            if "random_buckets" in step:
                for bucket_name, config in step["random_buckets"].items():
                    probability = config.get("probability", 0)
                    roll = random.random()
                    if roll < probability:
                        triggered_random_buckets.append(bucket_name)
                        print(
                            f"🎲 [RANDOM EVENT] '{bucket_name}' triggered! (rolled {roll:.3f} < {probability})"
                        )
                    else:
                        print(
                            f"🎲 [RANDOM CHECK] '{bucket_name}' not triggered (rolled {roll:.3f} >= {probability})"
                        )

            # Execute pre-script if it exists (runs before categorization, with user_response available)
            if "pre_script" in step:
                print(f"DEBUG: Executing pre-script")
                # Add user_response to a temporary copy of metadata for pre_script
                temp_metadata = metadata.copy()
                temp_metadata["user_response"] = user_response
                pre_result = execute_processing_script(
                    temp_metadata, step["pre_script"]
                )

                # Update metadata with pre-script results
                for key, value in pre_result.get("metadata", {}).items():
                    metadata[key] = value
                print(f"DEBUG: Pre-script completed, updated metadata")

            category = categorize_response(
                question,
                user_response,
                step["buckets"],
                step["tokens_for_ai"],
                classifier_model,
            )
            print(f"\nCategory: {category}")

            # Combine user's category with triggered random buckets
            # User's response is processed FIRST, then random events
            all_active_buckets = [category] + triggered_random_buckets
            print(f"📋 Processing buckets in order: {all_active_buckets}")

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
                else:
                    print(f"⚠️  Warning: No transition found for bucket '{bucket}'")

            # If no valid transitions found at all (not even for user's category), error
            if not active_transitions:
                print(
                    f"\nError: No valid transition found for category '{category}'. Please try again."
                )
                continue

            print(f"✓ Found {len(active_transitions)} transition(s) to process")

            # Track temporary metadata keys across all transitions
            metadata_tmp_keys = []

            # Track the final navigation target (use LAST transition's next_section_and_step)
            final_next_section_and_step = None

            # Track counts_as_attempt (if ANY transition counts, then it counts)
            any_counts_as_attempt = False

            # Process ALL active transitions in order
            for bucket_name, transition in active_transitions:
                print(f"\n{'='*60}")
                print(f"Processing transition for bucket: '{bucket_name}'")
                print(f"{'='*60}")

                # Check metadata conditions (v2.0 advanced conditions)
                if "metadata_conditions" in transition:
                    conditions_met = check_conditions(
                        metadata, transition["metadata_conditions"]
                    )
                    if not conditions_met:
                        print(
                            f"⚠️  Skipping '{bucket_name}' - metadata conditions not met"
                        )
                        print(f"Current Metadata: {json.dumps(metadata, indent=2)}")
                        continue

                # Print transition content blocks if they exist (v2.0 with templates & conditionals)
                if "content_blocks" in transition:
                    # Create template context
                    context = create_template_context(
                        metadata=metadata,
                        current_attempt=attempts,
                        max_attempts=max_attempts,
                        current_section=current_section_id,
                        current_step=current_step_id,
                        username="User",
                    )

                    # Filter and render content blocks (supports conditional blocks and templates)
                    filtered_blocks = filter_content_blocks(
                        transition["content_blocks"], metadata, context
                    )

                    if filtered_blocks:
                        transition_content = "\n\n".join(filtered_blocks)
                        translated_transition_content = translate_text(
                            transition_content, user_language, feedback_model
                        )
                        print(translated_transition_content)

                # Update metadata based on user actions
                if "metadata_add" in transition:
                    for key, value in transition["metadata_add"].items():
                        if value == "the-users-response":
                            value = user_response
                        elif isinstance(value, str):
                            if value.startswith("n+random(") and value.endswith(")"):
                                # Extract the range and apply the random increment
                                range_values = value[9:-1].split(",")
                                if len(range_values) == 2:
                                    x, y = map(int, range_values)
                                    value = metadata.get(key, 0) + random.randint(x, y)
                            elif value.startswith("n+") or value.startswith("n-"):
                                # Check if this is string concatenation (n+,value) or numeric operation (n+5)
                                if value.startswith("n+,") or value.startswith("n-,"):
                                    # String concatenation: append/remove from existing value
                                    operation = value[:2]  # "n+" or "n-"
                                    suffix = value[
                                        3:
                                    ]  # Everything after "n+," or "n-,"
                                    existing_value = metadata.get(key, "")
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
                                            parts = [p for p in parts if p != suffix]
                                            value = ",".join(parts)
                                        else:
                                            value = existing_value
                                else:
                                    # Numeric operation: extract the numeric part c and apply the operation +/-
                                    try:
                                        c = int(value[2:])
                                        if value.startswith("n+"):
                                            value = metadata.get(key, 0) + c
                                        elif value.startswith("n-"):
                                            value = metadata.get(key, 0) - c
                                    except ValueError:
                                        print(
                                            f"Warning: Invalid numeric operation '{value}' for key '{key}'"
                                        )
                                        # Leave value as-is if parsing fails
                        metadata[key] = value

                if "metadata_tmp_add" in transition:
                    for key, value in transition["metadata_tmp_add"].items():
                        if value == "the-users-response":
                            value = user_response
                        elif isinstance(value, str):
                            if value.startswith("n+random(") and value.endswith(")"):
                                # Extract the range and apply the random increment
                                range_values = value[9:-1].split(",")
                                if len(range_values) == 2:
                                    x, y = map(int, range_values)
                                    value = random.randint(x, y)
                            elif value.startswith("n+") or value.startswith("n-"):
                                # Check if this is string concatenation (n+,value) or numeric operation (n+5)
                                if value.startswith("n+,") or value.startswith("n-,"):
                                    # String concatenation: append/remove from existing value
                                    operation = value[:2]  # "n+" or "n-"
                                    suffix = value[
                                        3:
                                    ]  # Everything after "n+," or "n-,"
                                    existing_value = metadata.get(key, "")
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
                                            parts = [p for p in parts if p != suffix]
                                            value = ",".join(parts)
                                        else:
                                            value = existing_value
                                else:
                                    # Numeric operation: extract the numeric part c and apply the operation +/-
                                    try:
                                        c = int(value[2:])
                                        if value.startswith("n+"):
                                            value = metadata.get(key, 0) + c
                                        elif value.startswith("n-"):
                                            value = metadata.get(key, 0) - c
                                    except ValueError:
                                        print(
                                            f"Warning: Invalid numeric operation '{value}' for key '{key}'"
                                        )
                                        # Leave value as-is if parsing fails
                        metadata[key] = value
                        metadata_tmp_keys.append(key)  # Track temporary keys

                if "metadata_remove" in transition:
                    for key in transition["metadata_remove"]:
                        if key in metadata:
                            del metadata[key]

                # Handle metadata_clear - clear all metadata if set to True
                if (
                    "metadata_clear" in transition
                    and transition["metadata_clear"] == True
                ):
                    metadata.clear()

                # Handle metadata_random
                if "metadata_random" in transition:
                    random_key = random.choice(
                        list(transition["metadata_random"].keys())
                    )
                    random_value = transition["metadata_random"][random_key]
                    metadata[random_key] = random_value

                if "metadata_tmp_random" in transition:
                    random_key = random.choice(
                        list(transition["metadata_tmp_random"].keys())
                    )
                    random_value = random.choice(
                        transition["metadata_tmp_random"][random_key]
                    )
                    metadata[random_key] = random_value
                    metadata_tmp_keys.append(random_key)  # Track temporary keys

                # Handle metadata_weighted_random (v2.0)
                if "metadata_weighted_random" in transition:
                    for key, weighted_options in transition[
                        "metadata_weighted_random"
                    ].items():
                        selected_value = select_weighted_random(weighted_options)
                        metadata[key] = selected_value

                # Handle metadata_tmp_weighted_random (v2.0)
                if "metadata_tmp_weighted_random" in transition:
                    for key, weighted_options in transition[
                        "metadata_tmp_weighted_random"
                    ].items():
                        selected_value = select_weighted_random(weighted_options)
                        metadata[key] = selected_value
                        metadata_tmp_keys.append(key)

                # Execute the processing script if it exists
                if "processing_script" in step and transition.get(
                    "run_processing_script", False
                ):
                    # Add user_response to metadata temporarily for processing script
                    temp_metadata = metadata.copy()
                    temp_metadata["user_response"] = user_response

                    result = execute_processing_script(
                        temp_metadata, step["processing_script"]
                    )

                    # Copy any changes back to main metadata (except user_response)
                    for key, value in temp_metadata.items():
                        if key != "user_response":
                            metadata[key] = value
                    metadata["processing_script_result"] = result
                    metadata_tmp_keys.append("processing_script_result")

                    # Update metadata with results from the processing script
                    for key, value in result.get("metadata", {}).items():
                        metadata[key] = value

                print(
                    f"\n[Metadata after '{bucket_name}']: {json.dumps(metadata, indent=2)}"
                )

                # Provide feedback for THIS bucket
                if "feedback_prompts" in step:
                    # New multi-prompt system - legacy tokens get combined with each prompt
                    multi_feedback_messages = provide_feedback_prompts(
                        transition,
                        bucket_name,  # Use bucket_name instead of category
                        question,
                        step["feedback_prompts"],
                        user_response,
                        user_language,
                        metadata,
                        step.get(
                            "feedback_tokens_for_ai", ""
                        ),  # Pass legacy tokens to be combined
                        feedback_model,
                    )
                    # Display feedback immediately for this bucket
                    for feedback_msg in multi_feedback_messages:
                        print(f"\n{feedback_msg['name']}: {feedback_msg['content']}")
                elif step.get("feedback_tokens_for_ai"):
                    # Legacy single feedback system - only if no feedback_prompts
                    feedback = provide_feedback(
                        transition,
                        bucket_name,  # Use bucket_name instead of category
                        question,
                        user_response,
                        user_language,
                        step.get("feedback_tokens_for_ai", ""),
                        metadata,
                        feedback_model,
                    )
                    if feedback and feedback.strip():
                        print(f"\nFeedback: {feedback}")

                # Track navigation (LAST transition's next_section_and_step wins)
                if "next_section_and_step" in transition:
                    final_next_section_and_step = transition["next_section_and_step"]
                    print(f"🎯 Navigation target set to: {final_next_section_and_step}")

                # Track counts_as_attempt (if ANY transition counts, it counts)
                if transition.get("counts_as_attempt", True):
                    any_counts_as_attempt = True

            # End of multi-bucket processing loop

            # Check for progressive hints (v2.0)
            if "hints" in step:
                hint_context = create_template_context(
                    metadata=metadata,
                    current_attempt=attempts + 1,  # Next attempt
                    max_attempts=step_max_attempts,
                    current_section=current_section_id,
                    current_step=current_step_id,
                    username="User",
                )
                hint = get_progressive_hint(step["hints"], attempts + 1, hint_context)
                if hint:
                    translated_hint = translate_text(
                        hint["text"], user_language, feedback_model
                    )
                    print(f"\n💡 Hint: {translated_hint}")
                    # If hint doesn't count as attempt, adjust counting
                    if not hint["counts_as_attempt"]:
                        any_counts_as_attempt = False

            # Check if we should break or continue attempting
            if category not in [
                "partial_understanding",
                "limited_effort",
                "asking_clarifying_questions",
                "set_language",
                "off_topic",
            ]:
                break

            # Increment attempts if ANY transition counted
            if any_counts_as_attempt:
                attempts += 1

        if attempts == step_max_attempts:
            print("\nMaximum attempts reached. Moving to the next step.")

        # Remove temporary metadata at the end of the step
        for key in metadata_tmp_keys:
            if key in metadata:
                del metadata[key]

        # Use the final navigation target (from LAST processed transition)
        # v2.0: Resolve conditional navigation
        if final_next_section_and_step:
            resolved_navigation = resolve_conditional_navigation(
                final_next_section_and_step, metadata
            )
            if resolved_navigation:
                current_section_id, current_step_id = resolved_navigation.split(":")
        else:
            # No navigation specified, move to next step automatically
            current_section_id, current_step_id = get_next_section_and_step(
                yaml_content, current_section_id, current_step_id
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate an activity.")
    parser.add_argument(
        "yaml_file_path",
        type=str,
        help="Path to the activity YAML file",
        default="activity0.yaml",
    )
    args = parser.parse_args()
    simulate_activity(args.yaml_file_path)
