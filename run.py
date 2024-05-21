import pandas as pd
from utils import *
from solve_task import solve_task
import logging



if __name__ == "__main__":
    logging.basicConfig(
        filename="run.log",
        encoding="utf-8",
        format="[%(asctime)s] %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logging.info('--------------started a new run--------------')

    experiments = pd.read_csv("results_gpt4_normal.csv")

    for idx, experiment in experiments.iterrows():
        if experiment['messages'] != 'empty':
            continue
        try:
            logging.info(f'starting index {idx}')
            final_response, passed, messages = solve_task(experiment)
            experiments.at[idx, "passed"] = passed
            experiments.at[idx, "messages"] = messages
            experiments.at[idx, "new_final_response"] = final_response
            experiments.to_csv("results_gpt4_normal.csv", index=False)

        except Exception as e:
            logging.warning(f"skipped iteration {idx} due to {e}")
