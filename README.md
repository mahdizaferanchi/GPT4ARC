### Improvement on the paper: LLMs and the Abstraction and Reasoning Corpus: Successes, Failures, and the Importance of Object-based Representations \[4\]
**Related: Graphs, constraints, and search for the abstraction and reasoning corpus \[5\]**

Identifying weaknesses in LLMs for reasoning tasks and providing external tools that can help them make up for those weaknesses is a fascinating research topic. These papers address these problems with the help of the Abstraction and Reasoning Corpus (ARC) \[6\].  
  
One potential approach for improving performance is to provide external feedback to the model, which seems to be crucial for LLMs to be able to correct their responses \[7\]. To achieve this, I suggest asking for code instead of the final answer. Then, the provided code can be applied to the training samples of the task. If the correct output is achieved, the same function can be applied to the test samples to get the final answer. Otherwise, we can provide feedback to the model in the form of the output its code generated.  
  
This process can be repeated for a limited number of iterations if the model's code does not produce the right answer for the training examples. This can be considered a kind of CoT \[8\] prompting technique with the added help of a code interpreter. The python-style pseudo-code below shows an overview of the algorithm:

```python
def try_code(code, samples):
	exec(code) # defines a function called `transform`
	for sample in samples: # iterate training examples
		if transform(sample['input']) != sample['output']:
			# return the wrong pair
			# that results from the model's code
			return {
				'input': sample['input'],
				'output': transform(sample['output'])
			} 
	return 'success' # return 'success' if code passes all training samples
	
for train_pairs, test_pair in tasks: # iterate tasks
	# we assume the data has already been converted
	# to the ARGA object-based representation 
	prompt = f"You will be given a logic and reasoning puzzle... {train_pairs}... Provide a python function named 'transform' that can transform the input to the correct corresponding output."
	response = gpt_request(prompt)
	# get initial response from LLM
	code = extract_code_from_response(response)
	result = try_code(code, train_samples) # first try
	# provide feedback `num_tries` times if code does not work
	for _ in range(num_tries):
		if result != 'success':
			feedback_prompt = f"Your code produced the wrong output: {result}. Try again."
			response = gpt_request(feedback_prompt)
			code = extract_code_from_response(response)
			result = try_code(code, train_samples)
	test_result = transform(test_pair['input']) == test_pair['output']
	print(f"Produced right answer to test input? -> {test_result}")
```
This can also be seen as a form of program synthesis that has shown great results for the ARC dataset when used with DSLs.

This repository implements this approach with GPT-4 and the available ARGA representations obtained from [here](https://github.com/khalil-research/LLM4ARC/blob/main/output-logs/object-based/ARC-subset/object_based_few_shot_object_json_4.csv). The results, along with GPT logs, are available in the `csv` file, which is the same as the one referenced but with 3 new columns (`new_final_response`: final result of transforming the test sample, `messages`: GPT-4 conversation history, and `passed`: True if `new_final_response` is correct). The code is also available. An additional `example_convo.txt` file shows a sample conversation, since they are somewhat hard to parse from the `csv`. Here is the performance comparison:

|             | Few-shot | In-context few-shot with CoT | Asking for code (new/ours) |
| ----------- | -------- | ---------------------------- | -------------------------- |
| Object JSON | 21       | 23                           | **24**                           |

The experiment was done with `num_tries` set to 4. Of the **24** solved tasks **19** of them were solved on the first try, **4** on the second try, **0** on third, and only **1** on the fourth try.

### References
4. [Xu, Yudong, et al. "LLMs and the Abstraction and Reasoning Corpus: Successes, Failures, and the Importance of Object-based Representations." _arXiv preprint arXiv:2305.18354_ (2023).](https://arxiv.org/abs/2305.18354)
5. [Xu, Yudong, Elias B. Khalil, and Scott Sanner. "Graphs, constraints, and search for the abstraction and reasoning corpus." _Proceedings of the AAAI Conference on Artificial Intelligence_. Vol. 37. No. 4. 2023.](https://ojs.aaai.org/index.php/AAAI/article/view/25527)
6. [Chollet, François. "On the measure of intelligence." _arXiv preprint arXiv:1911.01547_ (2019).](https://arxiv.org/abs/1911.01547)
7. [Huang, Jie, et al. "Large language models cannot self-correct reasoning yet." _arXiv preprint arXiv:2310.01798_ (2023).](https://arxiv.org/abs/2310.01798)
8. [Wei, Jason, et al. "Chain-of-thought prompting elicits reasoning in large language models." _Advances in Neural Information Processing Systems_ 35 (2022): 24824-24837.](https://arxiv.org/abs/2201.11903)