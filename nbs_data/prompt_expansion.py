import json
import random
from openai import OpenAI

client = OpenAI()

def prompt_expansion(base_question, n=20):
    prompt = f"""
    Generate {n} diverse paraphrases of the following question.
    Keep the meaning identical (it should still ask for the description of the fMRI image).
    Avoid introducing any new information or changing the intent.
    Format the output as a list, with each paraphrase on a new line.

    Question: "{base_question}"
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or whichever LLM you use
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9
    )
    
    text = response.choices[0].message.content.strip()
    
    # Split lines into paraphrases (depending on how the LLM formats output)
    paraphrases = [line.strip("-•0123456789. ").strip() for line in text.split("\n") if line.strip()]
    
    return paraphrases

def answer_expansion(base_question, example_answers, n=20):
    prompt = f"""
    Generate {n} diverse answers to the following question based on the example answers provided.
    Ensure the answers are varied in phrasing but consistent in meaning. You can switch between short and long answers, and use capitalization or different wording
    Format the output as a list, with each answer on a new line.

    Question: "{base_question}"
    Example Answers:
    {', '.join(example_answers)}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or whichever LLM you use
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9
    )
    
    text = response.choices[0].message.content.strip()
    
    # Split lines into answers (depending on how the LLM formats output)
    answers = [line.strip("-•0123456789. ").strip() for line in text.split("\n") if line.strip()]
    
    return answers

if __name__ == "__main__":
    base_q = "Describe the brain activity from the fRMI scan"

    # repeat for 10 times
    all_paraphrases = []
    for i in range(10):
        paraphrases = prompt_expansion(base_q, n=20)        
        all_paraphrases.extend(paraphrases)

    # Deduplicate
    paraphrases = list(set(all_paraphrases))

    # Save to JSON
    with open("data/text_prompts/prompts/prompts_imaging.json", "w") as f:
        json.dump({"base_question": base_q, "paraphrases": paraphrases}, f, indent=2)

    print(f"Saved {len(paraphrases)} paraphrases.")

    # base_q = "From the fMRI scan, does the subject have autism?"
    # example_answers = ["Autism", "Yes, this subject has autism."]
    # all_answers = []
    # for i in range(3):
    #     answers = answer_expansion(base_q, example_answers=example_answers, n=10)
    #     all_answers.extend(answers)
    # all_answers.extend(example_answers)
    # answers = list(set(all_answers))

    # with open("data/answers_ASD_1.json", "w") as f:
    #     json.dump({"base_answer": example_answers[0], "answers": answers}, f, indent=2)
    # print(f"Saved {len(answers)} answers.")
