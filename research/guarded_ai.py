import argparse
import yaml
import json
import random
from openai import OpenAI

client = OpenAI()


# Load the YAML activity file
def load_yaml_activity(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


# Categorize the user's response using gpt-4o-mini
def categorize_response(question, response, buckets, tokens_for_ai):
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
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=5,
            temperature=0,
        )
        category = (
            completion.choices[0].message.content.strip().lower().replace(" ", "_")
        )
        return category
    except Exception as e:
        return f"Error: {e}"


# Generate AI feedback using gpt-4o-mini
def generate_ai_feedback(category, question, user_response, tokens_for_ai):
    messages = [
        {
            "role": "system",
            "content": f"{tokens_for_ai} Generate a human-readable feedback message based on the following:",
        },
        {
            "role": "user",
            "content": f"Question: {question}\nResponse: {user_response}\nCategory: {category}",
        },
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=250, temperature=0.7
        )
        feedback = completion.choices[0].message.content.strip()
        return feedback
    except Exception as e:
        return f"Error: {e}"


# Provide feedback based on the category
def provide_feedback(
    transition, category, question, user_response, user_language, tokens_for_ai
):
    feedback = ""
    if "ai_feedback" in transition:
        tokens_for_ai += f" Provide the feedback in {user_language}. {transition['ai_feedback'].get('tokens_for_ai', '')}."
        ai_feedback = generate_ai_feedback(
            category, question, user_response, tokens_for_ai
        )
        feedback += f"\n\nAI Feedback: {ai_feedback}"

    return feedback


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


def translate_text(text, target_language):
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
        completion = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=500, temperature=0.7
        )
        translation = completion.choices[0].message.content.strip()
        return translation
    except Exception as e:
        return f"Error: {e}"


def simulate_activity(yaml_file_path):
    yaml_content = load_yaml_activity(yaml_file_path)
    max_attempts = yaml_content.get("default_max_attempts_per_step", 3)

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

        # Get the user's language preference from metadata
        user_language = metadata.get("language", "English")

        # Translate and print all content blocks once per step
        if "content_blocks" in step:
            content = "\n\n".join(step["content_blocks"])
            translated_content = translate_text(content, user_language)
            print(translated_content)

        # Skip classification and feedback if there's no question
        if "question" not in step:
            current_section_id, current_step_id = get_next_section_and_step(
                yaml_content, current_section_id, current_step_id
            )
            continue

        question = step["question"]
        translated_question = translate_text(question, user_language)
        print(f"\nQuestion: {translated_question}")

        attempts = 0
        while attempts < max_attempts:
            user_response = input("\nYour Response: ")

            category = categorize_response(
                question, user_response, step["buckets"], step["tokens_for_ai"]
            )
            print(f"\nCategory: {category}")

            transition = step["transitions"].get(category, None)
            if not transition:
                print("\nError: No valid transition found. Please try again.")
                continue

            # Check metadata conditions
            if "metadata_conditions" in transition:
                conditions_met = all(
                    metadata.get(key) == value
                    for key, value in transition["metadata_conditions"].items()
                )
                if not conditions_met:
                    print("\nYou do not meet the required conditions to proceed.")
                    print(f"Current Metadata: {json.dumps(metadata, indent=2)}")
                    continue

            # Print transition content blocks if they exist
            if "content_blocks" in transition:
                transition_content = "\n\n".join(transition["content_blocks"])
                translated_transition_content = translate_text(
                    transition_content, user_language
                )
                print(translated_transition_content)

            feedback = provide_feedback(
                transition,
                category,
                question,
                user_response,
                user_language,
                step["tokens_for_ai"],
            )
            print(f"\nFeedback: {feedback}")

            # Track temporary metadata keys
            metadata_tmp_keys = []

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
                            # Extract the numeric part c and apply the operation +/-
                            c = int(value[1:])
                            if value.startswith("n+"):
                                value = metadata.get(key, 0) + c
                            elif value.startswith("n-"):
                                value = metadata.get(key, 0) - c
                    metadata[key] = value

            if "metadata_tmp_add" in transition:
                for key, value in transition["metadata_tmp_add"].items():
                    if value == "the-users-response":
                        value = user_response
                    metadata[key] = value
                    metadata_tmp_keys.append(key)  # Track temporary keys

            if "metadata_remove" in transition:
                for key in transition["metadata_remove"]:
                    if key in metadata:
                        del metadata[key]

            # Handle metadata_random
            if "metadata_random" in transition:
                random_key = random.choice(list(transition["metadata_random"].keys()))
                random_value = transition["metadata_random"][random_key]
                metadata[random_key] = random_value

            if "metadata_tmp_random" in transition:
                random_key = random.choice(
                    list(transition["metadata_tmp_random"].keys())
                )
                random_value = transition["metadata_tmp_random"][random_key]
                metadata[random_key] = random_value
                metadata_tmp_keys.append(random_key)  # Track temporary keys

            print(f"\nMetadata: {json.dumps(metadata, indent=2)}")

            if category not in [
                "partial_understanding",
                "asking_clarifying_questions",
                "set_language",
                "off_topic",
            ]:
                break

            # Access counts_as_attempt directly from the transition
            counts_as_attempt = transition.get("counts_as_attempt", True)
            if counts_as_attempt:
                attempts += 1

        if attempts == max_attempts:
            print("\nMaximum attempts reached. Moving to the next step.")

        # Remove temporary metadata at the end of the step
        for key in metadata_tmp_keys:
            if key in metadata:
                del metadata[key]

        # Access next_section_and_step directly from the transition
        next_section_and_step = transition.get("next_section_and_step", None)
        if next_section_and_step:
            current_section_id, current_step_id = next_section_and_step.split(":")
        else:
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
