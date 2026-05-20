"""Reward scoring for math + few-shot RL.

This is the math analogue of ICRL's `verl/utils/reward_score/qa_em_fewshot.py`.
Drop into `verl/utils/reward_score/math_fewshot.py` and route from
`main_ppo_fewshot.py:_select_rm_score_fn` when `data_source == 'math'`.

Combines:
  - accuracy via boxed-answer extraction + Hendrycks-MATH string equivalence
    (re-using ICRL's verl/utils/reward_score/math.py helpers) PLUS a sympy
    fallback for symbolic equivalence (e.g. 100-25*pi == -25*pi+100).
  - format penalty (presence/balance of <think>, <search>, <information>,
    <answer>, and \\boxed{...} inside the answer; multiple <answer> tags
    expected because the prompt contains few-shot demos with their own).

API mirrors qa_em_fewshot for plug-and-play:
  compute_score_fewshot(solution_str, ground_truth, accuracy_weight=0.8,
                        format_weight=0.2, format_score=0.0, return_details=False)

ground_truth is a dict like {'target': '21'} or {'target': '100-25\\pi'}.
"""

from __future__ import annotations

import random
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Reuse Hendrycks MATH string normalization from ICRL's math.py
# (kept inline here so this file is self-contained and copy-pasteable.)
# ---------------------------------------------------------------------------


def _last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    depth = 0
    right = None
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    if right is None:
        return None
    return string[idx: right + 1]


def _remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]
    left = "\\boxed{"
    if s[:len(left)] != left or s[-1] != "}":
        return s
    return s[len(left):-1]


def _fix_fracs(s: str) -> str:
    parts = s.split("\\frac")
    out = parts[0]
    for sub in parts[1:]:
        out += "\\frac"
        if sub and sub[0] == "{":
            out += sub
        else:
            if len(sub) < 2:
                return s
            a, b = sub[0], sub[1]
            rest = sub[2:] if len(sub) > 2 else ""
            if b != "{":
                out += "{" + a + "}{" + b + "}" + rest
            else:
                out += "{" + a + "}" + b + rest
    return out


def _fix_a_slash_b(s: str) -> str:
    if s.count("/") != 1:
        return s
    a, b = s.split("/")
    try:
        a_i, b_i = int(a), int(b)
        if s == f"{a_i}/{b_i}":
            return f"\\frac{{{a_i}}}{{{b_i}}}"
    except ValueError:
        pass
    return s


def _fix_sqrt(s: str) -> str:
    if "\\sqrt" not in s:
        return s
    parts = s.split("\\sqrt")
    out = parts[0]
    for sub in parts[1:]:
        if sub and sub[0] != "{":
            out += "\\sqrt{" + sub[0] + "}" + sub[1:]
        else:
            out += "\\sqrt" + sub
    return out


def _strip_string(s: str) -> str:
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")
    if "\\text{ " in s:
        s = s.split("\\text{ ")[0]
    s = s.replace("\\%", "").replace("\\,", "").replace("\\:", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s and s[0] == ".":
        s = "0" + s
    if s.count("=") == 1 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    return s


def _string_equiv(a: str, b: str) -> bool:
    if a is None or b is None:
        return False
    try:
        return _strip_string(a) == _strip_string(b)
    except Exception:
        return a == b


def _sympy_equiv(a: str, b: str) -> bool:
    """Fallback: try parsing both sides with sympy and check simplify(a-b)==0.

    Best-effort; returns False on any parse error. Useful for cases like
    '100-25*pi' vs '-25\\pi+100' that string-normalization may miss.
    """
    try:
        from sympy import sympify, simplify, latex
        from sympy.parsing.latex import parse_latex
    except Exception:
        return False

    def _try(expr):
        # Try LaTeX first, then plain sympify.
        for fn in (lambda s: parse_latex(s), lambda s: sympify(s)):
            try:
                return fn(expr)
            except Exception:
                continue
        return None

    ea = _try(a)
    eb = _try(b)
    if ea is None or eb is None:
        return False
    try:
        return simplify(ea - eb) == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_final_answer(solution_str: str) -> Optional[str]:
    """Pull the *last* <answer>...</answer> block, then the boxed value inside.

    Few-shot prompts contain demo <answer> tags, so we always take the last one.
    """
    if not solution_str:
        return None
    matches = list(_ANSWER_TAG_RE.finditer(solution_str))
    if not matches:
        return None
    raw = matches[-1].group(1).strip()
    boxed = _last_boxed_only_string(raw)
    if boxed is None:
        # Sometimes the model writes the answer without \\boxed; accept the
        # whole tag content as a fallback.
        return raw
    return _remove_boxed(boxed).strip()


# ---------------------------------------------------------------------------
# Format reward (parallels ICRL's compute_format_score)
# ---------------------------------------------------------------------------


def compute_format_score(solution_str: str, num_examples_in_prompt: int = 0,
                         return_stats: bool = False):
    """Penalty-based format compliance score in [0, 1].

    Important: few-shot prompts contain demo tags too, so we look only at the
    portion AFTER the last demo. We approximate "model-generated portion" as
    "after the last `</answer>` BEFORE the model's actual response starts" —
    but since the prompt is wrapped as a single user message, in practice the
    model's response begins after the last "Actual Problem:" marker. We split
    on that for robustness.
    """
    if not solution_str:
        empty_stats = {"think": 0, "search": 0, "info": 0, "answer": 0}
        return (0.0, empty_stats) if return_stats else 0.0

    text = solution_str
    if "Actual Problem:" in text:
        text = text.split("Actual Problem:")[-1]

    lower = text.lower()
    score = 1.0
    penalties = {
        "no_answer": 0.5,
        "unbalanced_answer": 0.2,
        "no_think": 0.15,
        "unbalanced_think": 0.1,
        "no_search": 0.1,
        "empty_answer": 0.2,
    }

    a_open = lower.count("<answer>")
    a_close = lower.count("</answer>")
    if a_open == 0 or a_close == 0:
        score -= penalties["no_answer"]
    if a_open != a_close:
        score -= penalties["unbalanced_answer"]

    t_open = lower.count("<think>")
    t_close = lower.count("</think>")
    if t_open == 0 or t_close == 0:
        score -= penalties["no_think"]
    if t_open != t_close:
        score -= penalties["unbalanced_think"]

    s_open = lower.count("<search>")
    s_close = lower.count("</search>")
    if s_open == 0:
        score -= penalties["no_search"]

    extracted = extract_final_answer(solution_str)
    if not extracted:
        score -= penalties["empty_answer"]

    score = max(0.0, min(1.0, score))

    if return_stats:
        stats = {
            "think": min(t_open, t_close),
            "search": min(s_open, s_close),
            "info": lower.count("<information>"),
            "answer": min(a_open, a_close),
        }
        return score, stats
    return score


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def compute_accuracy(solution_str: str, ground_truth) -> float:
    """1.0 if predicted boxed value matches gold, else 0.0.

    `ground_truth` can be either a string or a dict {'target': str|[str]}.
    """
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target", "")
    else:
        target = ground_truth
    if isinstance(target, list):
        targets = target
    else:
        targets = [str(target)]

    pred = extract_final_answer(solution_str)
    if pred is None:
        return 0.0

    for t in targets:
        t = str(t).strip()
        if _string_equiv(pred, t):
            return 1.0
        # Numeric tolerance: try as float
        try:
            if abs(float(pred.replace(",", "")) - float(t.replace(",", ""))) < 1e-6:
                return 1.0
        except Exception:
            pass
        if _sympy_equiv(pred, t):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Top-level entry points (mirror qa_em_fewshot signatures)
# ---------------------------------------------------------------------------


def compute_score_fewshot(solution_str, ground_truth,
                          accuracy_weight: float = 0.8,
                          format_weight: float = 0.2,
                          format_score: float = 0.0,
                          return_details: bool = False):
    do_print = random.randint(1, 64) <= 2

    acc = compute_accuracy(solution_str, ground_truth)
    fmt, stats = compute_format_score(solution_str, return_stats=True)
    extracted = extract_final_answer(solution_str)

    score = max(0.0, min(1.0, accuracy_weight * acc + format_weight * fmt))

    if do_print:
        gold = ground_truth.get("target") if isinstance(ground_truth, dict) else ground_truth
        print("-" * 40)
        print(f"[math_fewshot] gold={gold!r} pred={extracted!r}")
        print(f"  acc={acc:.2f} fmt={fmt:.2f} -> {score:.3f}")
        print(f"  tags: think={stats['think']} search={stats['search']} info={stats['info']} answer={stats['answer']}")
        print(f"  solution (truncated): {solution_str[-500:]!r}")

    if return_details:
        return score, acc, fmt, extracted, stats
    return score


def compute_score_em(solution_str, ground_truth, method="strict",
                     format_score: float = 0.0, score: float = 1.0):
    """Pure EM, for ablation."""
    acc = compute_accuracy(solution_str, ground_truth)
    return score if acc >= 0.5 else format_score
