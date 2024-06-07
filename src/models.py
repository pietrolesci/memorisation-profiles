# ======================================================================================================================
# For inference optimisations look here:
#   https://huggingface.co/docs/transformers/v4.37.0/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention
# ======================================================================================================================

import os
from pathlib import Path
from typing import Any, Literal

import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from lightning_utilities.core.rank_zero import rank_zero_info
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationConfig
from transformers.tokenization_utils import PreTrainedTokenizerBase


class HFCausalLM:
    """This class allows to load any Huggingface model and adapters."""

    AUTO_CONFIG_CLASS: type[AutoConfig] = AutoConfig
    AUTO_MODEL_CLASS: type[AutoModelForCausalLM] = AutoModelForCausalLM
    AUTO_TOKENIZER_CLASS: type[AutoTokenizer] = AutoTokenizer

    _model: _FabricModule
    _config: PretrainedConfig | None = None
    _tokenizer: PreTrainedTokenizerBase | None = None
    _generation_config: GenerationConfig | None = None
    _fabric: Fabric | None = None

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        subfolder: str | None = None,
        cache_dir: str | Path | None = None,
        accelerator: str = "gpu",
        precision: _PRECISION_INPUT | None = "bf16-true",
        deterministic: bool | Literal["warn_only"] = "warn_only",
        tf32_mode: Literal["highest", "high", "medium"] = "high",
        # convert_better_transformer: bool = False,  # NOTE: removing because it completely destroys model's capability!
    ) -> None:
        super().__init__()
        self._model_name = model_name
        if revision is not None and subfolder is not None:
            revision += "/" + subfolder
        self._revision = revision
        self._cache_dir = cache_dir
        self._fabric = Fabric(accelerator=accelerator, precision=precision, devices=1)
        self.set_deterministic(deterministic)
        self.set_torch_matmul_precision(tf32_mode)
        # self.init_model(convert_better_transformer)
        self.init_model()

    def set_torch_matmul_precision(self, tf32_mode: Literal["highest", "high", "medium"] = "highest") -> None:
        # equivalent to `torch.backends.cudnn.allow_tf32 = True`
        # convolutions are not changed, to do that you need
        # `torch.backends.cudnn.allow_tf32 = True`
        torch.set_float32_matmul_precision(tf32_mode)

    def set_deterministic(self, deterministic: bool | Literal["warn_only"]) -> None:
        kwargs = {}
        if isinstance(deterministic, str):
            assert deterministic == "warn_only", "deterministic can be a bool or `warn_only`"
            deterministic, kwargs = True, {"warn_only": True}

        # NOTE: taken from the lightning Trainer
        torch.use_deterministic_algorithms(deterministic, **kwargs)

        if deterministic:
            # fixing non-deterministic part of horovod
            # https://github.com/Lightning-AI/lightning/pull/1572/files#r420279383
            os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"

            # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            # Enable CUDNN deterministic mode
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False  # type: ignore

    @property
    def config(self) -> PretrainedConfig | None:
        return self._config

    @property
    def fabric(self) -> Fabric:
        assert self._fabric is not None
        return self._fabric

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        return self._tokenizer

    def __getattr__(self, attr: str) -> Any:
        """Forward call to underlying model."""
        if attr not in self.__dict__:
            return getattr(self._model, attr)
        return getattr(self, attr)

    def summary(self) -> None:
        print(
            f"Total num params: {self._model.num_parameters(only_trainable=False) / 1e6:.01f}M\n"
            f"Of which trainable: {self._model.num_parameters(only_trainable=True) / 1e6:.01f}M\n"
            f"With a memory footprint of {self._model.get_memory_footprint() / 1e9:.02f}GB\n"
            f"Total memory allocated {torch.cuda.max_memory_allocated() / 1e9:.02f}GB"
        )

    # def init_model(self, convert_better_transformer: bool) -> None:
    def init_model(self) -> None:
        # Init model directly on device
        with self.fabric.init_module():
            model = self.AUTO_MODEL_CLASS.from_pretrained(
                self._model_name, revision=self._revision, cache_dir=self._cache_dir
            )

            # if convert_better_transformer:
            #     model = model.to_bettertransformer()

        # Cast model to correct data type
        self._model = self.fabric.setup_module(model)

        # Set eval mode
        self._model.eval()
        print("Model set in `eval` mode")

        # Init configuration
        self._config = self.AUTO_CONFIG_CLASS.from_pretrained(
            self._model_name, revision=self._revision, cache_dir=self._cache_dir
        )

        # Init tokenizer
        self._tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(self._model_name, revision=self._revision)
        if not self._tokenizer.pad_token_id:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            self._tokenizer.pad_token = self._tokenizer.eos_token
            rank_zero_info("Tokenizer does not have pad token. Setting the <eos> token as pad.")

    def init_dataloader(self, dataloader: DataLoader) -> _FabricDataLoader:
        return self.fabric.setup_dataloaders(dataloader, move_to_device=False)  # type: ignore

    def transfer_to_device(self, batch: Any) -> Any:
        # Fabric knows how to handle non-gpu stuff so the batch can have anything inside
        # also for things already on device this is a no-op
        return self.fabric.to_device(batch)

    @torch.inference_mode()
    def generate(self, inputs: torch.LongTensor, generation_config: GenerationConfig) -> torch.LongTensor:
        assert self.tokenizer
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        return self._model.generate(inputs, generation_config=generation_config)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self._model(input_ids).logits

    @torch.inference_mode()
    def compute_statistics(self, token_ids: torch.LongTensor, only_sequence_surprisal: bool = False) -> dict:
        """More efficient alternative that manually computes surprisal"""
        # Forward pass
        labels = token_ids.clone()
        logprobs = self.forward(token_ids).log_softmax(-1)

        # Shift so that tokens < n predict n
        shift_labels = labels[..., 1:].contiguous()
        shift_logprobs = logprobs[..., :-1, :].contiguous()

        # Get the log-probability of the true token
        # (batch, seq, vocab = 1), there is an extra dim that makes it broadcastable to `shift_logprobs`
        true_logprobs = shift_logprobs.take_along_dim(dim=-1, indices=shift_labels[..., None])

        # Get surprisal, aka the negative of the log-probability
        sup = true_logprobs.squeeze(-1).neg()

        if only_sequence_surprisal:
            sequence_sup = sup.mean(-1)
            return {"sup_seq": sequence_sup.cpu().numpy().tolist()}

        # Get the rank of the true token
        # how many bigger token have log-probability bigger than the true token? this is the rank of the true token
        rank = (shift_logprobs > true_logprobs).long().sum(-1)

        # Get the entropy
        #  - \sum logp * p
        entropy = (shift_logprobs * shift_logprobs.exp()).sum(-1).neg()

        return {
            "sup": sup.cpu().numpy().tolist(),  # convertion to numpy works because model is outputting float32 somehow
            "rank": rank.cpu().numpy().tolist(),
            "entropy": entropy.cpu().numpy().tolist(),
        }

    # def compute_token_surprisal(self, token_ids: torch.LongTensor) -> list[float]:
    #     # Move to device
    #     token_ids = self.transfer_to_device(token_ids)

    #     # Forward pass
    #     labels = token_ids.clone()
    #     logits = self.forward(token_ids)

    #     # Shift so that tokens < n predict n
    #     shift_labels = labels[..., 1:].contiguous()
    #     shift_logits = logits[..., :-1, :].contiguous()

    #     # Rearrange to compute per-token loss
    #     b, s = shift_labels.size()
    #     shift_labels = rearrange(shift_labels, "b s -> (b s)")
    #     shift_logits = rearrange(shift_logits, "b s v -> (b s) v")
    #     token_surprisal = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="none")
    #     token_surprisal = rearrange(token_surprisal, "(b s) -> b s", b=b, s=s).cpu().numpy().tolist()

    #     return token_surprisal

    # def compute_token_accuracy(
    #     self, token_ids: torch.LongTensor, context_length: int = 32, continuation_length: int = 32
    # ) -> list[float]:
    #     # Move to device after subsetting -- this saves memory
    #     token_ids = self.transfer_to_device(token_ids[:, : continuation_length + context_length])

    #     # Prepare data
    #     context_ids = token_ids[:, :context_length]
    #     true_continuation_ids = token_ids[:, context_length:]

    #     # Generate from model
    #     generation_config = GenerationConfig(
    #         do_sample=False,
    #         min_length=continuation_length + context_length,
    #         max_length=continuation_length + context_length,
    #     )
    #     with torch.backends.cuda.sdp_kernel(enable_flash=True):
    #         pred_continuation_ids = self.generate(context_ids, generation_config)[:, context_length:]  # type: ignore

    #     # Compute accuracy
    #     token_accuracy = (true_continuation_ids == pred_continuation_ids).float().mean(-1).cpu().numpy().tolist()

    #     return token_accuracy

    # @torch.inference_mode()
    # def _compute_statistics(self, token_ids: torch.LongTensor) -> dict:
    #     # Forward pass
    #     labels = token_ids.clone()
    #     logits = self.forward(token_ids)

    #     # Shift so that tokens < n predict n
    #     shift_labels = labels[..., 1:].contiguous()
    #     shift_logits = logits[..., :-1, :].contiguous()

    #     # Predictions
    #     preds = shift_logits.argmax(-1)
    #     entropy = entr(shift_logits.softmax(-1)).sum(dim=-1)

    #     # Token surprisal
    #     b, s = shift_labels.size()
    #     shift_labels = rearrange(shift_labels, "b s -> (b s)")
    #     shift_logits = rearrange(shift_logits, "b s v -> (b s) v")
    #     token_surprisal = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="none")
    #     token_surprisal = rearrange(token_surprisal, "(b s) -> b s", b=b, s=s)

    #     return {
    #         "preds": preds.cpu().numpy().tolist(),
    #         "entropy": entropy.cpu().numpy().tolist(),
    #         "surprisal": token_surprisal.cpu().numpy().tolist(),
    #     }
