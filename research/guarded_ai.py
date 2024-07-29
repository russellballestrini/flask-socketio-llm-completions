import yaml
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
            "content": "{tokens_for_ai} Generate a human-readable feedback message based on the following:",
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
    yaml_content, section_id, step_id, category, question, user_response
):
    section = next(
        (s for s in yaml_content["sections"] if s["section_id"] == section_id), None
    )
    if not section:
        return "Section not found."

    step = next((s for s in section["steps"] if s["step_id"] == step_id), None)
    if not step:
        return "Step not found."

    transition = step["transitions"].get(category, None)
    if not transition:
        return "Category not found."

    feedback = ""
    if "ai_feedback" in transition:
        tokens_for_ai = (
            step["tokens_for_ai"] + " " + transition["ai_feedback"]["tokens_for_ai"]
        )
        ai_feedback = generate_ai_feedback(
            category, question, user_response, tokens_for_ai
        )
        feedback += f"\n\nAI Feedback: {ai_feedback}"

    next_section_and_step = transition.get("next_section_and_step", None)
    return feedback, next_section_and_step


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


# Simulate the activity
def simulate_activity(yaml_file_path):
    yaml_content = load_yaml_activity(yaml_file_path)
    max_attempts = yaml_content.get("default_max_attempts_per_step", 3)

    current_section_id = yaml_content["sections"][0]["section_id"]
    current_step_id = yaml_content["sections"][0]["steps"][0]["step_id"]

    while current_section_id and current_step_id:
        print(f"\n\nCurrent section: {current_section_id}, Current step: {current_step_id}\n\n")
        section = next(
            (
                s
                for s in yaml_content["sections"]
                if s["section_id"] == current_section_id
            ),
            None,
        )
        if not section:
            print("Section not found.")
            break

        step = next(
            (s for s in section["steps"] if s["step_id"] == current_step_id), None
        )
        if not step:
            print("Step not found.")
            break

        # Print all content blocks once per step
        if "content_blocks" in step:
            print("\n\n".join(step["content_blocks"]))

        # Skip classification and feedback if there's no question
        if "question" not in step:
            current_section_id, current_step_id = get_next_step(
                yaml_content, current_section_id, current_step_id
            )
            continue

        question = step["question"]

        attempts = 0
        while attempts < max_attempts:
            print(f"\nQuestion: {question}")

            user_response = input("\nYour Response: ")

            category = categorize_response(
                question, user_response, step["buckets"], step["tokens_for_ai"]
            )
            print(f"\nCategory: {category}")

            feedback, next_section_and_step = provide_feedback(
                yaml_content,
                section["section_id"],
                step["step_id"],
                category,
                question,
                user_response,
            )
            print(f"\nFeedback: {feedback}")

            if category not in [
                "off_topic",
                "asking_clarifying_questions",
                "partial_understanding",
            ]:
                break

            attempts += 1

        if attempts == max_attempts:
            print("\nMaximum attempts reached. Moving to the next step.")

        if next_section_and_step:
            current_section_id, current_step_id = next_section_and_step.split(":")
        else:
            current_section_id, current_step_id = get_next_section_and_step(
                yaml_content, current_section_id, current_step_id
            )


if __name__ == "__main__":
    # simulate_activity("activity13-choose-adventure.yaml")
    simulate_activity("activity0.yaml")
