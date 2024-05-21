import pandas as pd
import json
from utils import *
import logging

def solve_task(task):
    prompt_content = json.loads(task["Full_prompt"])[-1]["content"]
    statements = prompt_content.split("\n\n")[1:]
    test_idx = statements.index("Test:")
    train_objects = statements[:test_idx]
    test_objects = statements[test_idx + 1 : -1]
    train_objects = list(map(extract_object, train_objects))
    test_objects = list(map(extract_object, test_objects))
    train_pairs = []
    for i in range(0, len(train_objects), 2):
        train_pairs.append(
            {"input": train_objects[i], "output": train_objects[i + 1]}
        )

    messages = [
        {
            "role": "system",
            "content": "Assistant is a chatbot that is capable of doing human-level reasoning and inference.",
        },
        {
            "role": "user",
            "content": f'Let\'s play some puzzles that focus on reasoning and logic. In each puzzle, you will be provided a few demonstrations of how an "input image" gets transformed into a corresponding "output image". Write a python function called "transform" that can produce the correct output given an input image. Your function will be tested on a new image. Here is the first puzzle:\n\n{train_pairs}\n\nThis is very important for my career. Take you time and come up with an answer that works for all examples provided.',
        },
    ]

    for attempt in range(4):
        response = gpt_req(messages, temp=0 if attempt in [0, 1] else 1)
        messages.append(response.choices[0].message)
        res_content = response.choices[0].message.content

        try:
            code = find_between_substrings(res_content, "```python", "```")
            if not code:
                code = find_between_substrings(res_content, "```Python", "```")
            logging.debug(code)
            feedback = try_code(code, train_pairs)
        except Exception as e:
            logging.warning(e)
            feedback = {
                "role": "user",
                "content": f"You were supposed to provide a python function called transform but did not. Try again",
            }
        if feedback is None:
            logging.info(f"got right answer on training samples on attempt {attempt}")
            break
        else:
            messages.append(feedback)

    c = {}
    exec(code, c)
    ans = get_grid_repr(c['transform'](test_objects[0]))
    final_response = get_final_repr(ans)

    passed = task["True_answer"] == final_response

    return final_response, passed, messages
