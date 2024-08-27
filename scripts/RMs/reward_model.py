"""
    Abstract class for reward models.
"""
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class RewardModel:
    def __init__(self):
        raise NotImplementedError

    def pair_wise_scores(self, prompt: List[str], candidates: List[List[str]]):
        """
            Compute pairwise scores for prompt and candidates.
            Args:
            - prompt: (n, )
            - candidates: (n, num_candidates)

            Returns:
            - scores: (n, num_candidates, num_candidates) for each candidate pair
        """
        bsz = len(prompt)
        num_candidates = len(candidates[0])

        formatted = self.format_pair(prompt, candidates)    # size of (bsz, num_candidates * (num_candidates-1))
        formatted = self.regroup_formatted(formatted)       # size of (num_candidates * (num_candidates-1), bsz)

        inputs = self.format_input(formatted)   # size of bsz * dict

        scores = self.get_scores(inputs)

        scores_matrix = torch.zeros(bsz, num_candidates, num_candidates)
        # reshape scores as (bsz, num_candidates, (num_candidates-1))
        scores = scores.view(bsz, num_candidates, num_candidates-1)
        for i in range(num_candidates):
            scores_matrix[:, i, :i] = scores[:, i, :i]
            scores_matrix[:, i, i+1:] = scores[:, i, i:]

        return scores_matrix
    
    def format_pair(self, prompt: List[str], candidate: List[List[str]]):
        """
            Format a batch as model input.
            Args:
            - prompt: (bsz,)
            - candidate: (bsz, num_candidates)

            Returns:
            - formatted: (bsz, num_candidates * (num_candidates-1)) for the input contexts of all pairs
        """
        raise NotImplementedError

    def format_input(self, formatted: List[List[str]]):
        """
            Format input for model.
            Args:
            - formatted: (bsz, num_candidates * (num_candidates-1))

            Returns:
            - inputs: output of tokenizer of RMs
        """
        raise NotImplementedError

    def regroup_formatted(self, formatted: List[List[str]]):
        """
            Regroup formatted inputs.
            Args:
            - formatted: (bsz, num_candidates * (num_candidates-1))

            Returns:
            - regrouped: (num_candidates * (num_candidates-1), bsz)
        """
        # from group by batch to group by pair
        regrouped = []
        for i in range(len(formatted[0])):
            regrouped.append([formatted[j][i] for j in range(len(formatted))])
        return regrouped

    def get_scores(self, inputs):
        """
            Get scores from model.
            Args:
            - inputs: output of tokenizer of RMs, as (bsz, num_candidates * (num_candidates-1), ...)

            Returns:
            - scores: (bsz, num_candidates * (num_candidates-1))
        """
        raise NotImplementedError


class PairPreferenceLlama(RewardModel):
    """
        Following format of https://huggingface.co/RLHFlow/pair-preference-model-LLaMA3-8B
    """
    PLAIN_TEMPLATE="\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"
    PROMPT_TEMPLATE="[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer_plain = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer_plain.chat_template = self.PLAIN_TEMPLATE    # for formatting the context
        self.tokenizer.padding_side = "left"    # to align the output logits

        # tokens
        self.token_id_A = self.tokenizer.encode("A", add_special_tokens=False)
        self.token_id_B = self.tokenizer.encode("B", add_special_tokens=False)
        assert len(self.token_id_A) == 1 and len(self.token_id_B) == 1
        self.token_id_A = self.token_id_A[0]
        self.token_id_B = self.token_id_B[0]

        self.model.to(device)
        self.model.eval()

    def format_pair(self, prompt: List[str], candidate: List[List[str]]):
        formatted = []
        for batch_idx in range(len(prompt)):
            prompt_text = prompt[batch_idx]
            candidates = candidate[batch_idx]
            batch_formatted = []
            for i in range(len(candidates)):
                for j in range(len(candidates)):
                    if i != j:
                        context = [
                            {"role": "user", "content": prompt_text},
                        ]
                        context_str = self.tokenizer_plain.apply_chat_template(context, tokenize=False)
                        prompt = self.PROMPT_TEMPLATE.format(context=context_str, response_A=candidates[i], response_B=candidates[j])
                        message = [
                            {"role": "user", "content": prompt},
                        ]
                        message_str = self.tokenizer.apply_chat_template(message, tokenize=False).replace(self.tokenizer.bos_token, "")
                        batch_formatted.append(message_str)
            formatted.append(batch_formatted)

        return formatted

    def format_input(self, formatted: List[List[str]]):
        # apply left padding to align the output logits
        inputs = []
        for batch in formatted:
            inputs.append(self.tokenizer(batch, padding=True, return_tensors="pt", add_special_tokens=False))
        return inputs

    def get_scores(self, inputs):
        with torch.no_grad():
            scores = []
            for batch_inputs in inputs:
                batch_inputs = {k: v.to(self.model.device) for k, v in batch_inputs.items()}
                outputs = self.model(**batch_inputs)
                # take out the last token logits
                logits = outputs.logits[:, -1]
                # take out the logit for logits_A and logits_B
                logits_A = logits[:, self.token_id_A]
                logits_B = logits[:, self.token_id_B]
                scores.append(logits_A - logits_B)
            scores = torch.stack(scores, dim=1)
        
        return scores