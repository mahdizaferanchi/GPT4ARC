from dotenv import load_dotenv
load_dotenv()

import sys
import pandas as pd
import numpy as np
import time
import json
import re
import logging

from openai import OpenAI
client = OpenAI()


def find_between_substrings(input_string, start_substring, end_substring):
    pattern = re.compile(
        f"{re.escape(start_substring)}(.*?){re.escape(end_substring)}", re.DOTALL
    )
    match = re.search(pattern, input_string)

    if match:
        return match.group(1)
    else:
        return None


def find_after_substring(input_string, substring):
    pattern = re.compile(f"{re.escape(substring)}(.*)")
    match = re.search(pattern, input_string)

    if match:
        return match.group(1)
    else:
        return None


def extract_object(res):
    return {
        "image_size": eval(find_between_substrings(res, "size:", "\nObjects")),
        "objects": json.loads(find_after_substring(res, "\nObjects:\n")),
    }


def transform(param):
    raise Exception("not supposed to be called")


def gpt_req(messages, temp=0):
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=temp,
                seed=42,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response
        except Exception as e:
            logging.warning(e)
            time.sleep(60)
    raise Exception('no success after retrying request 3 times')


def get_grid_repr(object):
    grid_repr = np.zeros(object["image_size"])
    for item in object["objects"]:
        for i, coord in enumerate(item["coordinates"]):
            try:
                if isinstance(item["color"], int):
                    grid_repr[*coord] = item["color"]
                elif isinstance(item["color"], list):
                    grid_repr[*coord] = item["color"][i]
                else:
                    raise Exception('color was not list or int')
            except Exception as e:
                logging.critical(f'color was unexpected and led to this: {e}')
                raise e
    return grid_repr


def get_final_repr(ans):
    final_repr = ""
    for row in ans:
        for col in row:
            final_repr += f"{int(col)}"
        final_repr += "\r\n"
    return final_repr

def try_code(code, samples):
    try:
        c = {}
        exec(code, c)

        logging.info(f"trying: {id(c['transform'])}")
        for ex_number, pair in enumerate(samples, 1):
            predicted = get_grid_repr(c['transform'](pair["input"]))
            actual = get_grid_repr(pair["output"])
            predicted_pair = {
                "input": pair["input"],
                "predicted_output": c['transform'](pair["input"]),
            }
            if not (predicted == actual).all():
                return {
                    "role": "user",
                    "content": f"For example number {ex_number}, your code leads to this input-output pair:\n\n{predicted_pair}.\n\nBut this is not correct. Try again and remember to take your time and come up with a function that works for all examples provided.",
                }
    except Exception as e:
        logging.warning(e)
        return {
            "role": "user",
            "content": f"You were supposed to provide a python function called transform but after trying to execute your code, I got this error:\n\n{e}. Try again",
        }
