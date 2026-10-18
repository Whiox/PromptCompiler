from pathlib import Path

import tiktoken
import torch

from nanoGPT.model import GPT, GPTConfig


class Model:
    def __init__(
        self,
        experiment_name: str = "nanogpt_v0.4_2_seed42"
    ):
        self.ROOT = Path(__file__).resolve().parent
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.checkpoint = torch.load(
            self.ROOT / "experiments" / experiment_name / "ckpt.pt",
            map_location="cpu",
            weights_only=False,
        )

        self.model = GPT(GPTConfig(**self.checkpoint["model_args"]))
        self.state = {
            key.removeprefix("_orig_mod."): value
            for key, value in self.checkpoint["model"].items()
        }
        self.model.load_state_dict(self.state)
        self.model.to(self.device).eval()

        self.tokenizer = tiktoken.get_encoding("gpt2")


    @torch.inference_mode()
    def generate_scene(self, request: str) -> str:
        prompt = f"### Request:\n{request}\n\n### Scene:\n"
        token_ids = self.tokenizer.encode_ordinary(prompt)

        max_new_tokens = min(160, self.model.config.block_size - len(token_ids))
        if max_new_tokens <= 0:
            raise ValueError("Запрос слишком длинный для заданного контекста")

        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        y = self.model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,
        )

        answer_ids = y[0, len(token_ids):].tolist()
        return self.tokenizer.decode(answer_ids).split("<|endoftext|>", 1)[0].strip()
