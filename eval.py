"""Per-model-tag task evaluation."""

from __future__ import annotations

import json
import re
import sys
import time
import multiprocessing
from fractions import Fraction
from math import isclose
from pathlib import Path
from typing import Union

import jsonlines


MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"


def is_digit(s):
    """
    Check whether a value can be parsed as a numeric literal.

    Parameters:
        s: Value to test (may be str, int, or float).

    Returns:
        True if `s` can be cast to float after stripping commas, else False.
    """
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False


def math_equal(prediction: Union[bool, float, str],
               reference: Union[float, str],
               include_percentage: bool = True,
               is_close: bool = True,
               timeout: bool = False,
               ) -> bool:
    """
    Check whether a prediction matches a reference numerically or symbolically.

    Parameters:
        prediction: Predicted answer (bool, float, or str).
        reference: Ground-truth answer (float or str).
        include_percentage: If True, also accept percent-shifted variants of reference.
        is_close: If True, use math.isclose for numerical comparison.
        timeout: If True, run symbolic comparison in a subprocess with a timeout.

    Returns:
        True if prediction equals reference under numerical or symbolic comparison, else False.
    """
    try:
        if is_digit(prediction) and is_digit(reference):
            prediction = float(str(prediction).replace(",", ""))
            reference = float(str(reference).replace(",", ""))
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, rel_tol=1e-4):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    reference = str(reference).strip()
    prediction = str(prediction).strip()

    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or \
            (prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ['{', "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    if (prediction.startswith("[") and prediction.endswith("]")) and (reference.startswith("[") and reference.endswith("]")) or \
            (prediction.startswith("(") and prediction.endswith(")")) and (reference.startswith("(") and reference.endswith(")")):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all([math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                    for i in range(len(pred_parts))]):
                return True

    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def symbolic_equal(a, b):
    """
    Test symbolic equivalence of two expressions using sympy.

    Parameters:
        a: First expression (str or sympy-parsable).
        b: Second expression (str or sympy-parsable).

    Returns:
        True if simplify(a - b) == 0 or numerical evaluations are close, else False.
    """
    from sympy import simplify, N
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.parsing.latex import parse_latex

    def _parse(s):
        """
        Try parsing a string with sympy's LaTeX then expression parser.

        Parameters:
            s: Input string.

        Returns:
            A sympy expression if parsing succeeds, otherwise the original string.
        """
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except Exception:
                pass
        return s
    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a - b) == 0:
            return True
    except Exception:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=1e-3):
            return True
    except Exception:
        pass
    return False


def symbolic_equal_process(a, b, output_queue):
    """
    Worker target that places symbolic_equal(a, b) on a multiprocessing queue.

    Parameters:
        a: First expression.
        b: Second expression.
        output_queue: Multiprocessing queue to receive the boolean result.
    """
    output_queue.put(symbolic_equal(a, b))


def call_with_timeout(func, *args, timeout=1, **kwargs):
    """
    Invoke a worker function in a subprocess with a timeout.

    Parameters:
        func: Callable that writes its result to an output_queue argument.
        args: Positional arguments forwarded to func.
        timeout: Maximum seconds to wait for the subprocess.
        kwargs: Keyword arguments forwarded to func.

    Returns:
        The value placed on the queue by func, or False if the timeout elapsed.
    """
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return False
    return output_queue.get()


def last_boxed_only_string(string):
    """
    Return the last \\boxed{...} or \\fbox{...} substring from a string.

    Parameters:
        string: Source text from a MATH solution.

    Returns:
        The matched \\boxed{...} substring, or None if not found.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def remove_boxed(s):
    """
    Strip the surrounding \\boxed{...} wrapper from a string.

    Parameters:
        s: A string of the form "\\boxed{...}" or None.

    Returns:
        The inner content, or None on missing/malformed input.
    """
    if s is None:
        return None
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except Exception:
        return None


def fix_fracs(string):
    """
    Normalize \\frac numerator/denominator groupings to use explicit braces.

    Parameters:
        string: LaTeX-like string to normalize.

    Returns:
        Rewritten string with each \\frac followed by {num}{den}.
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string):
    """
    Convert a plain "a/b" integer fraction to LaTeX \\frac{a}{b} form.

    Parameters:
        string: Candidate fraction string.

    Returns:
        \\frac{a}{b} when a and b are integers, otherwise the original string.
    """
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (AssertionError, ValueError):
        return string


def remove_right_units(string):
    """
    Strip a trailing "\\text{ ...}" unit annotation from a math string.

    Parameters:
        string: Input math string.

    Returns:
        The portion of the string before any "\\text{ " unit suffix.
    """
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def fix_sqrt(string):
    """
    Add explicit braces around single-character \\sqrt arguments.

    Parameters:
        string: LaTeX-like string to normalize.

    Returns:
        Rewritten string with \\sqrt arguments wrapped in {...}.
    """
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    """
    Canonicalize a MATH answer string by removing whitespace and LaTeX noise.

    Parameters:
        string: Raw answer string.

    Returns:
        Normalized string ready for equivalence comparison.
    """
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def is_equiv(str1, str2, verbose=False):
    """
    Check whether two MATH answer strings are equivalent after normalization.

    Parameters:
        str1: First answer string.
        str2: Second answer string.
        verbose: If True, print the normalized strings before comparison.

    Returns:
        True if the strings are equal after strip_string or via math_equal, else False.
    """
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return math_equal(ss1, ss2) or ss1 == ss2
    except Exception:
        return math_equal(str1, str1) or str1 == str2


def is_number(s):
    """
    Check whether a string represents a number (incl. unicode numerics).

    Parameters:
        s: Value to test.

    Returns:
        True if `s` parses as float or has a unicode numeric value, else False.
    """
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    """
    Extract the final numeric answer following "The answer is: " from a completion.

    Parameters:
        completion: Model generation string.

    Returns:
        The rounded numeric answer, or None if no valid number is found.
    """
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) and is_number(numerator):
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    frac = Fraction(match.group().replace(',', ''))
                    return round(float(frac.numerator / frac.denominator))
                return None
            if float(match.group().replace(',', '')) == float('inf'):
                return None
            return round(float(match.group().replace(',', '')))
    return None


def batch_data(data_list, batch_size=1):
    """
    Split a list into batches with the last batch absorbing any remainder.

    Parameters:
        data_list: Sequence to split.
        batch_size: Target size of each batch.

    Returns:
        List of batches (sublists) covering all items in data_list.
    """
    n = len(data_list) // batch_size
    out = []
    for i in range(n - 1):
        out.append(data_list[i * batch_size:(i + 1) * batch_size])
    out.append(data_list[max(0, (n - 1) * batch_size):MAX_INT])
    return out


def evaluate_gsm8k(model_dir: str, data_path: str,
                   start: int = 0, end: int = MAX_INT,
                   batch_size: int = 1319, tensor_parallel_size: int = 1,
                   gpu_memory_utilization: float = 0.85,
                   ) -> float:
    """
    Run GSM8K accuracy evaluation through vLLM.

    Parameters:
        model_dir: Path to the HF model directory.
        data_path: Path to the GSM8K test jsonl.
        start: First problem index (inclusive).
        end: Last problem index (exclusive).
        batch_size: Number of prompts per vLLM generate call.
        tensor_parallel_size: vLLM tensor parallel degree.
        gpu_memory_utilization: vLLM GPU memory fraction.

    Returns:
        Accuracy (fraction of problems answered correctly).
    """
    from vllm import LLM, SamplingParams

    gsm8k_ins, gsm8k_answers = [], []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('promt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            gsm8k_ins.append(problem_prompt.format(instruction=item["question"]))
            temp_ans = item['answer'].split('#### ')[1]
            gsm8k_answers.append(int(temp_ans.replace(',', '')))

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('lenght ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER",
                   "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction",
                   "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1024,
                                     stop=stop_tokens)
    print('sampleing =====', sampling_params)
    llm = LLM(model=model_dir, tensor_parallel_size=tensor_parallel_size,
              gpu_memory_utilization=gpu_memory_utilization)

    res_completions = []
    for prompt, _ in zip(batch_gsm8k_ins, gsm8k_answers):
        if not isinstance(prompt, list):
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            res_completions.append(output.outputs[0].text)

    result = []
    invalid_outputs = []
    for prompt, completion, prompt_answer in zip(gsm8k_ins, res_completions, gsm8k_answers):
        y_pred = extract_answer_number(completion)
        if y_pred is not None:
            result.append(float(y_pred) == float(prompt_answer)
                          or math_equal(y_pred, prompt_answer))
        else:
            result.append(False)
            invalid_outputs.append({'question': prompt, 'output': completion,
                                    'answer': prompt_answer})

    acc = sum(result) / len(result)
    print('len invalid outputs ====', len(invalid_outputs))
    print('start===', start, ', end====', end)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)
    return acc


def _math_process_results(doc, completion, answer, invalid_outputs_acc):
    """
    Score a single MATH completion against the gold answer.

    Parameters:
        doc: Prompt/question that produced the completion.
        completion: Model output text.
        answer: Gold answer extracted from the dataset.
        invalid_outputs_acc: List that accumulates unparsable completions.

    Returns:
        True if the extracted answer is equivalent to the gold, else False.
    """
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0].strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        return is_equiv(extract_ans, answer)
    invalid_outputs_acc.append({'question': doc, 'output': completion,
                                'answer': answer})
    return False


def evaluate_math(model_dir: str, data_path: str,
                  start: int = 0, end: int = MAX_INT,
                  batch_size: int = 500, tensor_parallel_size: int = 1,
                  gpu_memory_utilization: float = 0.85,
                  ) -> float:
    """
    Run Hendrycks MATH accuracy evaluation through vLLM.

    Parameters:
        model_dir: Path to the HF model directory.
        data_path: Path to the MATH test jsonl.
        start: First problem index (inclusive).
        end: Last problem index (exclusive).
        batch_size: Number of prompts per vLLM generate call.
        tensor_parallel_size: vLLM tensor parallel degree.
        gpu_memory_utilization: vLLM GPU memory fraction.

    Returns:
        Accuracy (fraction of problems answered correctly).
    """
    from vllm import LLM, SamplingParams

    hendrycks_math_ins, hendrycks_math_answers = [], []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('promt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            hendrycks_math_ins.append(problem_prompt.format(instruction=item["instruction"]))
            solution = item['output']
            hendrycks_math_answers.append(remove_boxed(last_boxed_only_string(solution)))

    print('total length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print('lenght ====', len(hendrycks_math_ins))
    batch_in = batch_data(hendrycks_math_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER",
                   "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction",
                   "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048,
                                     stop=stop_tokens)
    print('sampleing =====', sampling_params)
    llm = LLM(model=model_dir, tensor_parallel_size=tensor_parallel_size,
              gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=128)

    res_completions = []
    for prompt, _ in zip(batch_in, hendrycks_math_answers):
        if not isinstance(prompt, list):
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            res_completions.append(output.outputs[0].text)

    invalid_outputs = []
    results = []
    for prompt, completion, prompt_answer in zip(hendrycks_math_ins, res_completions,
                                                 hendrycks_math_answers):
        results.append(_math_process_results(prompt, completion, prompt_answer,
                                             invalid_outputs))

    acc = sum(results) / len(results) if results else 0.0
    print('len invalid outputs ====', len(invalid_outputs))
    print('start===', start, ', end====', end)
    print('length====', len(results), ', acc====', acc)
    return acc


def _sb_build_prompt(item: dict, lang: str = "en") -> str:
    """
    Build a SafetyBench multiple-choice prompt for a single item.

    Parameters:
        item: Dict containing the "question" and "options" fields.
        lang: Language code ("en" for English, otherwise Chinese).

    Returns:
        Formatted prompt string ending with "Answer:" (or its Chinese equivalent).
    """
    options_str = "".join(f"({chr(ord('A') + i)}) {opt}\n"
                          for i, opt in enumerate(item["options"]))
    if lang == "en":
        return (f"Question: {item['question'].strip()}\nOptions:\n"
                f"{options_str}Answer:")
    return (f"问题：{item['question'].strip()}\n选项：\n"
            f"{options_str}答案：")


def _sb_letter_token_ids(tokenizer, letters="ABCD"):
    """
    Resolve each option letter to a single tokenizer token id.

    Parameters:
        tokenizer: HF tokenizer to query.
        letters: String of letters to encode (default "ABCD").

    Returns:
        List of token IDs, one per letter, used for CLP scoring.
    """
    out = []
    for L in letters:
        ids = tokenizer.encode(L, add_special_tokens=False)
        if len(ids) == 1:
            out.append(ids[0])
            continue
        chosen = None
        for tid in ids:
            dec = tokenizer.decode([tid]).strip()
            if dec == L:
                chosen = tid
                break
        if chosen is None:
            chosen = ids[-1]
        out.append(chosen)
    return out


def evaluate_safetybench(model_dir: str,
                         data_dir: str = "your_data_path",
                         lang: str = "en", n_max: int = 0,
                         batch_size: int = 256,
                         gpu_memory_utilization: float = 0.85,
                         max_model_len: int = 4096,
                         output_json: str | None = None,
                         ) -> dict:
    """
    Run SafetyBench multiple-choice evaluation using CLP letter scoring.

    Parameters:
        model_dir: Path to the HF model directory.
        data_dir: Directory containing test_<lang>.json and answers file.
        lang: Dataset language code ("en" or "zh").
        n_max: Optional cap on number of items (0 = all).
        batch_size: Currently unused, kept for API parity.
        gpu_memory_utilization: vLLM GPU memory fraction.
        max_model_len: vLLM maximum sequence length.
        output_json: Optional path to dump per-task JSON.

    Returns:
        Dict with overall accuracy, parse rate, per-category accuracy, and metadata.
    """
    from vllm import LLM, SamplingParams

    qs = json.loads((Path(data_dir) / f"test_{lang}.json").read_text())
    ans = json.loads((Path(data_dir) / f"test_answers_{lang}.json").read_text())

    items = []
    for q in qs:
        qid = str(q["id"])
        gold = ans.get(qid, {}).get("answer")
        items.append({"id": q["id"], "question": q["question"],
                      "options": q["options"], "answer": gold,
                      "category": q.get("category",
                                        ans.get(qid, {}).get("category", "_unknown"))})
    if n_max > 0:
        items = items[:n_max]
    n = len(items)
    has_answers = all(it["answer"] is not None for it in items)
    print(f"[safetybench] {n} items (lang={lang}), has_answers={has_answers}")

    print(f"[safetybench] loading vLLM: {model_dir}")
    llm = LLM(model=model_dir, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len)
    tokenizer = llm.get_tokenizer()

    abcd_ids = _sb_letter_token_ids(tokenizer, "ABCD")
    print(f"[safetybench] A/B/C/D token IDs: {abcd_ids}")

    prompts = [_sb_build_prompt(it, lang) + "(" for it in items]
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    wall = int(time.time() - t0)

    correct, parsed = 0, 0
    per_cat: dict = {}
    predictions = []
    for it, o in zip(items, outputs):
        logps = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        n_opt = len(it["options"])
        scores = []
        for j in range(n_opt):
            tid = abcd_ids[j]
            lp_obj = logps.get(tid)
            if lp_obj is None:
                scores.append(float("-inf"))
            else:
                scores.append(float(lp_obj.logprob))

        if all(s == float("-inf") for s in scores):
            pred = None
        else:
            pred = max(range(n_opt), key=lambda i: scores[i])
            parsed += 1

        cat = it["category"]
        per_cat.setdefault(cat, [0, 0])
        per_cat[cat][1] += 1
        predictions.append({"id": it["id"], "category": cat, "pred": pred,
                            "scores": scores})
        if has_answers and pred is not None and pred == it["answer"]:
            correct += 1
            per_cat[cat][0] += 1

    if has_answers:
        acc = 100.0 * correct / n
        cat_acc = {k: 100.0 * v[0] / v[1] for k, v in per_cat.items() if v[1] > 0}
    else:
        acc = None
        cat_acc = {}
    parse_rate = 100.0 * parsed / n

    out = {
        "task": "safetybench", "lang": lang, "n": n, "acc": acc,
        "parsed_rate": parse_rate, "by_category": cat_acc,
        "wall_seconds": wall,
        "method": "CLP (logits at last position, pick max over A/B/C/D token IDs)",
        "abcd_token_ids": abcd_ids,
    }
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "by_category"}, indent=2))
    print("by_category:")
    for k, v in sorted(cat_acc.items()):
        print(f"  {k:30s}  {v:6.2f}")
    return out


def _evaluate_lm_eval_task(model_dir: str, task,
                           output_path: str | None = None,
                           gpu_memory_utilization: float = 0.85,
                           max_model_len: int = 4096,
                           batch_size: str | int = "auto",
                           confirm_run_unsafe_code: bool = False,
                           ) -> dict:
    """
    Run one or more lm-evaluation-harness tasks via the vLLM backend.

    Parameters:
        model_dir: Path to the HF model directory.
        task: Single task name or list of task names for lm_eval.
        output_path: Optional directory to write per-task JSON results.
        gpu_memory_utilization: vLLM GPU memory fraction.
        max_model_len: vLLM maximum sequence length.
        batch_size: Batch size for lm_eval (or "auto").
        confirm_run_unsafe_code: Pass-through flag for tasks that exec code.

    Returns:
        Dict returned by lm_eval.simple_evaluate (includes "results").
    """
    from lm_eval import simple_evaluate

    tasks = [task] if isinstance(task, str) else list(task)

    model_args = (f"pretrained={model_dir},dtype=bfloat16,"
                  f"tensor_parallel_size=1,"
                  f"gpu_memory_utilization={gpu_memory_utilization},"
                  f"max_model_len={max_model_len}")

    kwargs = {}
    if confirm_run_unsafe_code:
        kwargs["confirm_run_unsafe_code"] = True

    results = simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        **kwargs,
    )

    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        name = task if isinstance(task, str) else "_".join(tasks)
        (Path(output_path) / f"{name}.json").write_text(
            json.dumps(results.get("results", {}), indent=2)
        )
    print(f"[lm_eval] task={task} results: {results.get('results', {})}")
    return results


_MAGICODER_PROMPT = (
    "You are an exceptionally intelligent coding assistant that "
    "consistently delivers accurate and reliable responses to user "
    "instructions.\n\n"
    "@@ Instruction\n{instruction}\n\n"
    "@@ Response\n{response}"
)


_MAGICODER_N_PROBLEMS_PER_BATCH = {"humaneval": 16, "mbpp": 24}


def _map_humaneval_problem(p: dict) -> tuple[str, str, str]:
    """
    Map a HumanEval problem dict to a (task_id, instruction, response_prefix) tuple.

    Parameters:
        p: HumanEval problem dict with "task_id" and "prompt".

    Returns:
        task_id: Problem identifier string.
        instruction: Magicoder-style natural language instruction.
        response_prefix: Pre-filled response prefix containing the signature.
    """
    base = p["prompt"].strip()
    instruction = (
        "Write a solution to the following problem:\n"
        f"```python\n{base}\n```"
    )
    response_prefix = f"```python\n{base}"
    return p["task_id"], instruction, response_prefix


def _map_mbpp_problem(p: dict) -> tuple[str, str, str]:
    """
    Map an MBPP problem dict to a (task_id, instruction, response_prefix) tuple.

    Parameters:
        p: MBPP problem dict with "task_id" and "prompt" (the prompt contains
            a docstring with NL description followed by an assert).

    Returns:
        task_id: Problem identifier as string.
        instruction: NL description plus the required assertion in a code block.
        response_prefix: Bare "```python" prefix.
    """
    prompt = p["prompt"]
    start_index = prompt.index('"""')
    end_index = prompt.rindex('"""')
    body = prompt[start_index + 3:end_index]
    assert_index = body.index("assert")
    nl = body[:assert_index].strip()
    if not nl.endswith("."):
        nl += "."
    assertion = body[assert_index:].strip()
    instruction = (
        f"{nl} Your code should satisfy the following assertion:\n"
        f"```python\n{assertion}\n```"
    )
    response_prefix = "```python"
    return str(p["task_id"]), instruction, response_prefix


def _truncate_at_fence(text: str) -> str:
    """
    Truncate a generated string at the first triple-backtick fence.

    Parameters:
        text: Raw model completion.

    Returns:
        Text up to (but not including) the first "```", or the full text if absent.
    """
    idx = text.find("```")
    return text[:idx] if idx != -1 else text


def _chunked(seq, n):
    """
    Split a sequence into consecutive chunks of size n.

    Parameters:
        seq: Sequence to chunk.
        n: Chunk size.

    Returns:
        List of chunks (slices of seq).
    """
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _run_evalplus_magicoder(model_dir: str, dataset: str,
                            output_path: str | None,
                            n_problems_per_batch: int | None = None) -> dict:
    """
    Run paper-faithful HumanEval/MBPP via HF generate + EvalPlus scoring.

    Parameters:
        model_dir: Path to the HF model directory.
        dataset: Either "humaneval" or "mbpp".
        output_path: Optional working directory for samples and results.
        n_problems_per_batch: Override for batch size (defaults per dataset).

    Returns:
        Dict of the form {"results": {dataset: {"pass@1": float, "n": int}}}.
    """
    import os as _os
    import shutil
    import subprocess
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              GenerationConfig)
    from evalplus.data import (get_human_eval_plus, get_mbpp_plus,
                               write_jsonl)

    assert dataset in ("humaneval", "mbpp"), dataset
    if n_problems_per_batch is None:
        n_problems_per_batch = _MAGICODER_N_PROBLEMS_PER_BATCH[dataset]

    work = (Path(output_path) if output_path
            else Path(f"./evalplus_{dataset}"))
    work.mkdir(parents=True, exist_ok=True)
    work = work.resolve()
    model_dir_abs = str(Path(model_dir).resolve())

    raw = (get_human_eval_plus() if dataset == "humaneval"
           else get_mbpp_plus())
    mapper = (_map_humaneval_problem if dataset == "humaneval"
              else _map_mbpp_problem)
    items = [mapper(p) for p in raw.values()]
    print(f"[magicoder] {dataset} loaded {len(items)} problems, "
          f"n_problems_per_batch={n_problems_per_batch}")

    print(f"[magicoder] loading tokenizer/model from {model_dir_abs} "
          f"(dtype=bfloat16, device_map='auto')")
    tokenizer = AutoTokenizer.from_pretrained(model_dir_abs, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"


    try:
        import accelerate
        model = AutoModelForCausalLM.from_pretrained(
            model_dir_abs,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except ImportError:
        print("[magicoder] accelerate not installed; loading on cuda:0 "
              "without device_map (functionally equivalent for 7B on 1 GPU)")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir_abs,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    model.eval()

    gen_config = GenerationConfig(
        max_new_tokens=512,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )

    bos_id = tokenizer.bos_token_id
    samples: list[dict] = []
    with torch.no_grad():
        for batch in _chunked(items, n_problems_per_batch):
            prompts = [
                _MAGICODER_PROMPT.format(instruction=instr, response=rp)
                for (_tid, instr, rp) in batch
            ]
            enc = tokenizer(
                prompts,
                add_special_tokens=False,
                return_tensors=None,
                padding=False,
            )


            ids_with_bos = [[bos_id] + ids for ids in enc["input_ids"]]
            max_len = max(len(x) for x in ids_with_bos)
            pad_id = tokenizer.pad_token_id
            input_ids = torch.tensor(
                [[pad_id] * (max_len - len(x)) + x for x in ids_with_bos],
                dtype=torch.long,
            ).to(model.device)
            attention_mask = (input_ids != pad_id).long()

            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )
            gen_only = out[:, input_ids.shape[1]:]
            completions = tokenizer.batch_decode(
                gen_only, skip_special_tokens=True,
            )
            for (tid, _instr, _rp), comp in zip(batch, completions):
                samples.append({
                    "task_id": tid,
                    "completion": _truncate_at_fence(comp),
                })
            print(f"[magicoder] {dataset} {len(samples)}/{len(items)} done")


    del model
    torch.cuda.empty_cache()

    samples_path = work / "samples.jsonl"
    write_jsonl(str(samples_path), samples)
    print(f"[magicoder] wrote {len(samples)} samples -> {samples_path}")

    env = _os.environ.copy()
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    py_bin = _os.path.dirname(sys.executable)
    env["PATH"] = py_bin + _os.pathsep + env.get("PATH", "")

    if dataset == "mbpp":

        sanitized_path = samples_path.with_name(
            samples_path.stem + "-sanitized.jsonl"
        )
        if shutil.which("evalplus.sanitize") is not None:
            subprocess.run(
                ["evalplus.sanitize", "--samples", str(samples_path)],
                cwd=str(work), env=env, check=True,
            )

            default_sanitized = samples_path.with_name(
                samples_path.stem + "-sanitized.jsonl"
            )
            if default_sanitized.exists():
                sanitized_path = default_sanitized
            else:

                hits = list(work.glob(
                    samples_path.stem + "*sanitized*.jsonl"))
                if hits:
                    sanitized_path = hits[0]
        else:
            subprocess.run(
                [sys.executable, "-m", "evalplus.sanitize",
                 "--samples", str(samples_path)],
                cwd=str(work), env=env, check=True,
            )
            hits = list(work.glob(samples_path.stem + "*sanitized*.jsonl"))
            if hits:
                sanitized_path = hits[0]
        eval_input = sanitized_path
        print(f"[magicoder] sanitized -> {eval_input}")
    else:
        eval_input = samples_path


    eval_path = eval_input.with_name(eval_input.stem + "_eval_results.json")
    if eval_path.exists():
        eval_path.unlink()

    print(f"[evalplus] evaluate {dataset} (base only) samples={eval_input}")
    subprocess.run(
        ["evalplus.evaluate", "--dataset", dataset,
         "--samples", str(eval_input),
         "--base_only",
         "--i_just_wanna_run"],
        cwd=str(work), env=env, check=True,
    )
    data = json.loads(eval_path.read_text())
    ev = data["eval"]
    n = len(ev)
    base_pass = sum(1 for v in ev.values() if v[0]["base_status"] == "pass")
    pass1 = base_pass / n if n else 0.0
    print(f"[magicoder] {dataset} pass@1={pass1:.4f}  n={n}")

    results = {"results": {dataset: {"pass@1": pass1, "n": n}}}
    if output_path:
        (Path(output_path) / f"{dataset}.json").write_text(
            json.dumps(results["results"], indent=2)
        )
    return results


def evaluate_humaneval(model_dir: str, output_path: str | None = None,
                       gpu_memory_utilization: float = 0.85,
                       max_model_len: int = 4096) -> dict:
    """
    Run the Magicoder-faithful HumanEval evaluation.

    Parameters:
        model_dir: Path to the HF model directory.
        output_path: Optional directory for samples and per-task JSON.
        gpu_memory_utilization: Accepted for API parity; ignored by HF backend.
        max_model_len: Accepted for API parity; ignored by HF backend.

    Returns:
        Dict of the form {"results": {"humaneval": {"pass@1": float, "n": int}}}.
    """
    del gpu_memory_utilization, max_model_len
    return _run_evalplus_magicoder(model_dir, "humaneval", output_path)


def evaluate_mbpp(model_dir: str, output_path: str | None = None,
                  gpu_memory_utilization: float = 0.85,
                  max_model_len: int = 4096) -> dict:
    """
    Run the Magicoder-faithful MBPP evaluation (with sanitize step).

    Parameters:
        model_dir: Path to the HF model directory.
        output_path: Optional directory for samples and per-task JSON.
        gpu_memory_utilization: Accepted for API parity; ignored by HF backend.
        max_model_len: Accepted for API parity; ignored by HF backend.

    Returns:
        Dict of the form {"results": {"mbpp": {"pass@1": float, "n": int}}}.
    """
    del gpu_memory_utilization, max_model_len
    return _run_evalplus_magicoder(model_dir, "mbpp", output_path)


def evaluate_truthfulqa(model_dir: str, output_path: str | None = None,
                        gpu_memory_utilization: float = 0.85,
                        max_model_len: int = 2048) -> dict:
    """
    Run the lm-eval truthfulqa_mc2 task via vLLM.

    Parameters:
        model_dir: Path to the HF model directory.
        output_path: Optional directory to write per-task JSON results.
        gpu_memory_utilization: vLLM GPU memory fraction.
        max_model_len: vLLM maximum sequence length.

    Returns:
        Dict returned by lm_eval.simple_evaluate for truthfulqa_mc2.
    """
    return _evaluate_lm_eval_task(model_dir, "truthfulqa_mc2",
                                  output_path=output_path,
                                  gpu_memory_utilization=gpu_memory_utilization,
                                  max_model_len=max_model_len)


_VICUNA_SYS = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
_LLAVA_PROMPT = (
    f"{_VICUNA_SYS} USER: <image>\n{{question}}\n"
    "Answer the question using a single word or phrase. ASSISTANT:"
)
_LLAVA_PROMPT_TEXTVQA = (
    f"{_VICUNA_SYS} USER: <image>\nReference OCR token: {{ocr}}\n{{question}}\n"
    "Answer the question using a single word or phrase. ASSISTANT:"
)


def _llava_normalize(s: str) -> str:
    """
    Normalize a VQA prediction or answer string for matching.

    Parameters:
        s: Raw string to normalize.

    Returns:
        Lowercased, whitespace-collapsed string with trailing punctuation removed.
    """
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".,!?")
    return s


def _llava_load_gqa(n_max: int = 0):
    """
    Load the GQA testdev_balanced split paired with its image dataset.

    Parameters:
        n_max: Optional cap on number of items returned (0 = all).

    Returns:
        List of dicts with question_id, question, answer, and image.
    """
    from datasets import load_dataset
    qs = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions",
                      split="testdev")
    imgs = load_dataset("lmms-lab/GQA", "testdev_balanced_images",
                        split="testdev")
    img_map = {row["id"]: row["image"] for row in imgs}
    items = []
    for q in qs:
        if q["imageId"] not in img_map:
            continue
        items.append({
            "question_id": q["id"],
            "question": q["question"],
            "answer": q["answer"],
            "image": img_map[q["imageId"]],
        })
    if n_max > 0:
        items = items[:n_max]
    return items


def _llava_load_textvqa(n_max: int = 0):
    """
    Load the TextVQA validation split.

    Parameters:
        n_max: Optional cap on number of items returned (0 = all).

    Returns:
        List of dicts with question_id, question, answers, image, and ocr_tokens.
    """
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/textvqa", split="validation")
    items = []
    for ex in ds:
        items.append({
            "question_id": ex["question_id"],
            "question": ex["question"],
            "answers": ex["answers"],
            "image": ex["image"],
            "ocr_tokens": ex.get("ocr_tokens", []),
        })
    if n_max > 0:
        items = items[:n_max]
    return items


def _llava_gqa_score(items, predictions):
    """
    Compute exact-match accuracy for GQA predictions.

    Parameters:
        items: List of GQA items with "answer" field.
        predictions: List of model prediction strings aligned with items.

    Returns:
        Accuracy as a percentage (0-100).
    """
    correct = 0
    for it, pred in zip(items, predictions):
        if _llava_normalize(pred) == _llava_normalize(it["answer"]):
            correct += 1
    return 100.0 * correct / len(items)


def _llava_textvqa_score(items, predictions):
    """
    Compute standard VQA accuracy where per-question score = min(#matching/3, 1).

    Parameters:
        items: List of TextVQA items with "answers" lists.
        predictions: List of model prediction strings aligned with items.

    Returns:
        Accuracy as a percentage (0-100).
    """
    total = 0.0
    for it, pred in zip(items, predictions):
        p = _llava_normalize(pred)
        matches = sum(1 for a in it["answers"] if _llava_normalize(a) == p)
        total += min(matches / 3.0, 1.0)
    return 100.0 * total / len(items)


def _evaluate_llava_task(model_dir: str, task: str,
                         output_json: str | None = None,
                         n_max: int = 0,
                         gpu_memory_utilization: float = 0.85,
                         max_model_len: int = 4096) -> dict:
    """
    Run a LLaVA VQA evaluation (GQA or TextVQA) through vLLM.

    Parameters:
        model_dir: Path to the llava-hf converted model directory.
        task: Either "gqa" or "textvqa".
        output_json: Optional path to dump per-task JSON.
        n_max: Optional cap on number of items (0 = all).
        gpu_memory_utilization: vLLM GPU memory fraction.
        max_model_len: vLLM maximum sequence length.

    Returns:
        Dict with task name, item count, accuracy, wall time, and model dir.
    """
    from vllm import LLM, SamplingParams

    print(f"[llava_vqa] loading {task} ...")
    items = _llava_load_gqa(n_max) if task == "gqa" else _llava_load_textvqa(n_max)
    n = len(items)
    print(f"[llava_vqa] {n} items")

    print(f"[llava_vqa] loading vLLM: {model_dir}")
    llm = LLM(model=model_dir, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len, enforce_eager=True,
              limit_mm_per_prompt={"image": 1})

    sp = SamplingParams(temperature=0.0, max_tokens=64,
                        stop=["\nUSER:", "\n\n"])

    if task == "textvqa":
        def fmt(it):
            """
            Format a TextVQA item into a LLaVA prompt including OCR tokens.

            Parameters:
                it: TextVQA item with "question" and "ocr_tokens".

            Returns:
                Formatted prompt string with OCR reference and question.
            """
            ocr = ", ".join(it.get("ocr_tokens") or [])
            return _LLAVA_PROMPT_TEXTVQA.format(question=it["question"], ocr=ocr)
    else:
        def fmt(it):
            """
            Format a GQA item into a LLaVA prompt.

            Parameters:
                it: GQA item with a "question" field.

            Returns:
                Formatted prompt string for the GQA question.
            """
            return _LLAVA_PROMPT.format(question=it["question"])
    prompts = [
        {"prompt": fmt(it), "multi_modal_data": {"image": it["image"]}}
        for it in items
    ]

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    wall = int(time.time() - t0)
    predictions = [o.outputs[0].text for o in outputs]

    if task == "gqa":
        acc = _llava_gqa_score(items, predictions)
    else:
        acc = _llava_textvqa_score(items, predictions)

    res = {"task": task, "n": n, "acc": acc, "wall_seconds": wall, "model": model_dir}
    if output_json:
        from pathlib import Path as _P
        _P(output_json).parent.mkdir(parents=True, exist_ok=True)
        _P(output_json).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


def evaluate_llava_gqa(model_dir: str, output_json: str | None = None,
                      n_max: int = 0,
                      gpu_memory_utilization: float = 0.85,
                      max_model_len: int = 4096) -> dict:
    """
    Run the GQA evaluation for a LLaVA model.

    Parameters:
        model_dir: Path to the llava-hf converted model directory.
        output_json: Optional path to dump per-task JSON.
        n_max: Optional cap on number of items (0 = all).
        gpu_memory_utilization: vLLM GPU memory fraction.
        max_model_len: vLLM maximum sequence length.

    Returns:
        Dict with the GQA accuracy and metadata.
    """
    return _evaluate_llava_task(model_dir, "gqa",
                                output_json=output_json, n_max=n_max,
                                gpu_memory_utilization=gpu_memory_utilization,
                                max_model_len=max_model_len)


def evaluate_llava_textvqa(model_dir: str, output_json: str | None = None,
                          n_max: int = 0,
                          gpu_memory_utilization: float = 0.85,
                          max_model_len: int = 4096) -> dict:
    """
    Run the TextVQA evaluation for a LLaVA model.

    Parameters:
        model_dir: Path to the llava-hf converted model directory.
        output_json: Optional path to dump per-task JSON.
        n_max: Optional cap on number of items (0 = all).
        gpu_memory_utilization: vLLM GPU memory fraction.
        max_model_len: vLLM maximum sequence length.

    Returns:
        Dict with the TextVQA accuracy and metadata.
    """
    return _evaluate_llava_task(model_dir, "textvqa",
                                output_json=output_json, n_max=n_max,
                                gpu_memory_utilization=gpu_memory_utilization,
                                max_model_len=max_model_len)


def evaluate_qwen2_5(model_dir: str, output_dir: str | None = None, **_) -> dict:
    """
    Run the Qwen2.5-7B-Instruct evaluation suite (minerva_math + GPQA).

    Parameters:
        model_dir: Path to the HF model directory.
        output_dir: Optional directory to dump per-task and summary JSON.

    Returns:
        Dict with model metadata and "math" and "gpqa" result sub-dicts.
    """
    out = {"model_tag": "qwen2_5", "model_dir": model_dir}
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    qout = (str(Path(output_dir) / "qwen2_5") if output_dir else None)
    res = _evaluate_lm_eval_task(
        model_dir, ["minerva_math", "leaderboard_gpqa_extended"],
        output_path=qout,
        confirm_run_unsafe_code=True,
    )
    res_dict = res.get("results", {})
    out["math"] = {k: v for k, v in res_dict.items() if k.startswith("minerva_math")}
    out["gpqa"] = {k: v for k, v in res_dict.items() if "gpqa" in k}
    if output_dir:
        (Path(output_dir) / "summary.json").write_text(
            json.dumps(out, indent=2, default=str))
    return out


def evaluate_qwen3guard(model_dir: str, output_dir: str | None = None, **_) -> dict:
    """
    Run the Qwen3Guard-Gen-8B evaluation suite (MMLU + BBQ).

    Parameters:
        model_dir: Path to the HF model directory.
        output_dir: Optional directory to dump per-task and summary JSON.

    Returns:
        Dict with model metadata and "mmlu" and "bbq" result sub-dicts.
    """
    out = {"model_tag": "qwen3guard", "model_dir": model_dir}
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    qout = (str(Path(output_dir) / "qwen3guard") if output_dir else None)
    res = _evaluate_lm_eval_task(
        model_dir, ["mmlu", "bbq"],
        output_path=qout,
    )
    res_dict = res.get("results", {})
    out["mmlu"] = {k: v for k, v in res_dict.items() if k.startswith("mmlu")}
    out["bbq"] = {k: v for k, v in res_dict.items() if k.startswith("bbq")}
    if output_dir:
        (Path(output_dir) / "summary.json").write_text(
            json.dumps(out, indent=2, default=str))
    return out


_BEIT3_DEFAULT_TOKENIZER_DIR = (
    "/NHNHOME/WORKSPACE/0526040001_A/model/beit3_base_coco_captioning"
)
_BEIT3_DEFAULT_COCO_DATA = "./data/coco"
_BEIT3_EVAL_BASELINE_SCRIPT = (
    "/NHNHOME/WORKSPACE/0526040001_A/deltacomp/mydelta/beit3_eval_baseline.py"
)


def evaluate_beit3(model_dir: str,
                   tokenizer_dir: str = _BEIT3_DEFAULT_TOKENIZER_DIR,
                   data_path: str = _BEIT3_DEFAULT_COCO_DATA,
                   output_dir: str | None = None,
                   batch_size: int = 32,
                   limit: int = 0,
                   **_) -> dict:
    """
    Run the BEiT-3 COCO-Captioning evaluation via the baseline subprocess.

    Parameters:
        model_dir: Path to the unilm .pth checkpoint file (not a directory).
        tokenizer_dir: Directory of the BEiT-3 tokenizer assets.
        data_path: Path to the COCO data root.
        output_dir: Optional output directory (defaults to ./result/eval/beit3).
        batch_size: Batch size passed to the baseline script.
        limit: Optional cap on number of items (0 = all).

    Returns:
        Dict loaded from the summary.json produced by the baseline script.
    """
    import subprocess
    out_dir = output_dir or "./result/eval/beit3"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, _BEIT3_EVAL_BASELINE_SCRIPT,
        "--ckpt", model_dir,
        "--tokenizer_dir", tokenizer_dir,
        "--data_path", data_path,
        "--output_dir", out_dir,
        "--batch_size", str(batch_size),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    subprocess.check_call(cmd)

    return json.load(open(Path(out_dir) / "summary.json"))


DEFAULT_DATA_ROOT = "'your_data_path'"


def evaluate(model_tag: str, model_dir: str,
             data_root: str = DEFAULT_DATA_ROOT,
             output_dir: str | None = None) -> dict:
    """
    Dispatch to the appropriate eval suite for the given model_tag.

    Parameters:
        model_tag: Tag selecting the suite (wm, mc, chat, llava, qwen2_5, qwen3guard, beit3).
        model_dir: Path to the HF model directory (or .pth for beit3).
        data_root: Root directory containing dataset files.
        output_dir: Optional directory for per-task JSON and summary.

    Returns:
        Dict containing task scores keyed by sub-task name plus model metadata.
    """
    out: dict = {"model_tag": model_tag, "model_dir": model_dir}
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if model_tag == "wm":
        gsm = evaluate_gsm8k(model_dir, f"{data_root}/GSM8K_test.jsonl")
        out["gsm8k_acc"] = gsm

    elif model_tag == "mc":
        out["mbpp"] = evaluate_mbpp(
            model_dir,
            output_path=str(Path(output_dir) / "mbpp") if output_dir else None,
        ).get("results", {})

    elif model_tag == "chat":
        sb_json = (str(Path(output_dir) / "safetybench.json")
                   if output_dir else None)
        out["safetybench"] = evaluate_safetybench(
            model_dir, data_dir=f"{data_root}/safetybench",
            output_json=sb_json,
        )
        out["truthfulqa"] = evaluate_truthfulqa(
            model_dir,
            output_path=str(Path(output_dir) / "truthfulqa") if output_dir else None,
        ).get("results", {})

    elif model_tag == "llava":


        gqa_json = (str(Path(output_dir) / "gqa.json")
                    if output_dir else None)
        out["gqa"] = evaluate_llava_gqa(model_dir, output_json=gqa_json)
        tvqa_json = (str(Path(output_dir) / "textvqa.json")
                     if output_dir else None)
        out["textvqa"] = evaluate_llava_textvqa(model_dir, output_json=tvqa_json)

    elif model_tag == "qwen2_5":
        res = evaluate_qwen2_5(model_dir, output_dir=output_dir)
        out["math"] = res.get("math")
        out["gpqa"] = res.get("gpqa")

    elif model_tag == "qwen3guard":
        res = evaluate_qwen3guard(model_dir, output_dir=output_dir)
        out["mmlu"] = res.get("mmlu")
        out["bbq"] = res.get("bbq")

    elif model_tag == "beit3":

        out["beit3"] = evaluate_beit3(model_dir, output_dir=output_dir)

    else:
        raise ValueError(f"unknown model_tag={model_tag!r}")

    if output_dir:
        (Path(output_dir) / "summary.json").write_text(
            json.dumps({k: v for k, v in out.items()
                        if k != "safetybench"}, indent=2, default=str))
    return out


def main():
    """
    Parse CLI arguments and invoke evaluate() for the requested model_tag.
    """
    import argparse
    ap = argparse.ArgumentParser(prog="python -m proposed_delta.eval")
    ap.add_argument("--model_tag", required=True,
                    choices=["wm", "mc", "chat", "llava",
                             "qwen2_5", "qwen3guard", "beit3"])
    ap.add_argument("--model_dir", required=True,
                    help="HF model directory (output of merge step); "
                         "for beit3 this is the unilm .pth checkpoint file path")
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
                    help="root containing GSM8K_test.jsonl / MATH_test.jsonl / safetybench/")
    ap.add_argument("--output_dir", default=None,
                    help="optional dir to dump per-task json + summary.json")
    args = ap.parse_args()
    evaluate(args.model_tag, args.model_dir,
             data_root=args.data_root, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
