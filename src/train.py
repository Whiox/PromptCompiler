"""Fine-tune nanoGPT's GPT-2 on request/scene JSONL pairs."""

from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
from tempfile import NamedTemporaryFile

import tiktoken
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanoGPT.model import GPT


class Trainer:
    def __init__(
        self,
        dataset_dir: str = "v0.2",
        output_dir: str = "gpt2_v0.2_seed42",
        steps: int = 100,
        eval_every: int = 50,
        batch_size: int = 2,
        accumulation: int = 2,
        learning_rate: float = 5e-5,
        context_size: int = 256,
        random_seed: int = 42,
    ) -> None:
        self.dataset_dir = ROOT / "datasets" / dataset_dir
        self.output_dir = ROOT / "experiments" / output_dir
        self.steps = steps
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.accumulation = accumulation
        self.learning_rate = learning_rate
        self.context_size = context_size
        self.random_seed = random_seed

        if min(steps, eval_every, batch_size, accumulation) < 1 or learning_rate <= 0:
            raise ValueError("Training parameters must be positive")
        if not 1 <= context_size <= 1024:
            raise ValueError("GPT-2 context must be between 1 and 1024")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = (
            "bfloat16" if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else "float16" if self.device == "cuda" else "float32"
        )
        self.random = random.Random(random_seed)
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def _prompt(self, request: str) -> str:
        return f"### Request:\n{request}\n\n### Scene:\n"

    def _encode(self, row: dict) -> tuple[list, list]:
        prefix = self.tokenizer.encode_ordinary(self._prompt(row["request"]))
        scene = json.dumps(row["scene"], separators=(",", ":"), ensure_ascii=False)
        answer = self.tokenizer.encode_ordinary(scene) + [self.tokenizer.eot_token]
        # Next-token labels; request positions do not contribute to loss.
        return (prefix + answer)[:-1], [-1] * (len(prefix) - 1) + answer

    def _load_data(self) -> None:
        self.rows = {}
        self.encoded = {}
        for split in ["train", "val"]:
            path = self.dataset_dir / f"{split}.jsonl"
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            self.rows[split] = [json.loads(line) for line in lines if line.strip()]
            if not self.rows[split]:
                raise ValueError(f"Empty {split} dataset")
            self.encoded[split] = [self._encode(row) for row in self.rows[split]]
            if max(len(x) for x, _ in self.encoded[split]) > self.context_size:
                raise ValueError(f"{split} examples exceed context_size")

    def _batch(self, pairs: list) -> tuple[torch.Tensor, torch.Tensor]:
        length = max(len(x) for x, _ in pairs)
        x = torch.full(
            (len(pairs), length), self.tokenizer.eot_token,
            dtype=torch.long, device=self.device,
        )
        y = torch.full_like(x, -1)
        for i, (inputs, targets) in enumerate(pairs):
            x[i, :len(inputs)] = torch.tensor(inputs, device=self.device)
            y[i, :len(targets)] = torch.tensor(targets, device=self.device)
        return x, y

    def _autocast(self):
        if self.device == "cuda":
            return torch.autocast("cuda", dtype=getattr(torch, self.precision))
        return nullcontext()

    def _setup_model(self) -> None:
        torch.manual_seed(self.random_seed)
        self.model = GPT.from_pretrained("gpt2", {"dropout": 0.0})
        self.model.crop_block_size(self.context_size)
        self.model.to(self.device)
        self.optimizer = self.model.configure_optimizers(
            0.01, self.learning_rate, (0.9, 0.95), self.device,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device == "cuda" and self.precision == "float16",
        )

    def _save_config(self) -> None:
        self.config = {
            "dataset": str(self.dataset_dir), "out": str(self.output_dir),
            "steps": self.steps, "eval_every": self.eval_every,
            "batch_size": self.batch_size, "accumulation": self.accumulation,
            "lr": self.learning_rate, "context": self.context_size,
            "seed": self.random_seed, "model": "gpt2", "device": self.device,
            "precision": self.precision, "torch": str(torch.__version__),
            "loss": "answer and EOS only",
        }
        content = json.dumps(self.config, indent=2)
        (self.output_dir / "config.json").write_text(content, encoding="utf-8")
        # Preserve the exact input examples used for this run.
        for split, rows in self.rows.items():
            content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            (self.output_dir / f"{split}.jsonl").write_text(content, encoding="utf-8")

    def _save_checkpoint(self, step: int) -> None:
        checkpoint = {
            "model": self.model.state_dict(),
            "model_args": asdict(self.model.config),
            "iter_num": step, "config": self.config,
        }
        # Write through Python's file handle, then replace the old checkpoint
        # only after the new file has been completely written and closed.
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="wb", dir=self.output_dir, prefix="ckpt_", suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                torch.save(checkpoint, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self.output_dir / "ckpt.pt")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @torch.inference_mode()
    def _generate(self, request: str) -> str:
        ids = self.tokenizer.encode_ordinary(self._prompt(request))
        if len(ids) >= self.context_size:
            raise ValueError("Validation prompt exceeds context_size")
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        result = []
        for _ in range(min(192, self.context_size - len(ids))):
            with self._autocast():
                logits, _ = self.model(x)
            token = int(logits[0, -1].argmax())
            if token == self.tokenizer.eot_token:
                break
            result.append(token)
            x = torch.cat([x, torch.tensor([[token]], device=self.device)], dim=1)
        return self.tokenizer.decode(result)

    @torch.inference_mode()
    def _evaluate(self, step: int) -> None:
        self.model.eval()
        losses = []
        # Fixed validation subset; test.jsonl is not used during training.
        for pair in self.encoded["val"][:16]:
            x, y = self._batch([pair])
            with self._autocast():
                _, loss = self.model(x, y)
            losses.append(float(loss))

        selected, seen = [], set()
        for row in self.rows["val"]:
            key = json.dumps(row["scene"], sort_keys=True)
            if key not in seen:
                selected.append(row)
                seen.add(key)
            if len(selected) == 4:
                break

        samples = []
        for row in selected:
            text = self._generate(row["request"])
            try:
                parsed, valid = json.loads(text), True
            except json.JSONDecodeError:
                parsed, valid = None, False
            samples.append({
                "request": row["request"], "expected": row["scene"],
                "generated": text, "json_valid": valid,
                "exact_match": parsed == row["scene"],
            })
        metrics = {
            "step": step, "val_answer_loss": sum(losses) / len(losses),
            "val_loss_examples": len(losses), "generated_examples": len(samples),
            "json_valid_count": sum(s["json_valid"] for s in samples),
            "exact_match_count": sum(s["exact_match"] for s in samples),
        }
        content = json.dumps({**metrics, "samples": samples}, ensure_ascii=False, indent=2)
        (self.output_dir / f"step_{step:04}.json").write_text(content, encoding="utf-8")
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        self.model.train()

    def _train_step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(self.accumulation):
            pairs = self.random.choices(self.encoded["train"], k=self.batch_size)
            x, y = self._batch(pairs)
            with self._autocast():
                _, loss = self.model(x, y)
            total += float(loss.detach())
            self.scaler.scale(loss / self.accumulation).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return total / self.accumulation

    def run(self) -> None:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {self.output_dir}")
        self._load_data()
        print(
            f"Device: {self.device}; train: {len(self.rows['train'])}; "
            f"val: {len(self.rows['val'])}", flush=True,
        )
        self._setup_model()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()
        self._evaluate(0)
        for step in range(1, self.steps + 1):
            loss = self._train_step()
            if step % 10 == 0:
                print(f"step {step}/{self.steps}: train_answer_loss={loss:.4f}", flush=True)
            if step % self.eval_every == 0 or step == self.steps:
                self._evaluate(step)
                self._save_checkpoint(step)
        print(f"Saved: {self.output_dir / 'ckpt.pt'}", flush=True)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
