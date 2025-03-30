import re
import pdb

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
ANS_RE2 = re.compile(r"-?[0-9]\d*((\.|,)\d*)?")
INVALID_ANS = "[invalid]"

class GSM8kEvaluator:

    def extract_answer(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            match2 = [""] + [i for i in ANS_RE2.finditer(completion)]
            match2 = match2[-1]
            # breakpoint()
            if match2:
                match_str = match2.group(0).strip()
                match_str = match_str.replace(",", "")
                return match_str
            return INVALID_ANS

    def is_correct(model_completion, gt_example):
        gt_answer = GSM8kEvaluator.extract_answer(gt_example["answer"])
        assert gt_answer != INVALID_ANS
        return GSM8kEvaluator.extract_answer(model_completion) == gt_answer