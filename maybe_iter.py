# if charmap error: copy $env:PYTHONUTF8="1" in terminal


import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AdamW,
    SchedulerType,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2ForCTC,
    get_scheduler,
    is_wandb_available,
    set_seed,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from transformers.utils import send_example_telemetry

from torch.utils.tensorboard import SummaryWriter, writer
from torch.utils.data import DataLoader, TensorDataset
from loadmp3 import BIDSBrainVisionDataset

from typing import List, Dict, Union
import numpy as np

#product quantization in code

logger = get_logger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on {device}")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="hf-internal-testing/librispeech_asr_dummy", #  MLCommons/peoples_speech
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_names",
        nargs="+",
        type=str,
        required=False,
        default=["clean"],
        help="The configuration names of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_split_names",
        nargs="+",
        type=str,
        required=False,
        default=["validation", "test"], #each about 600h (30k in total)
        help="The names of the training data set splits to use (via the datasets library).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--preprocessing_only",
        action="store_true",
        help="Only run the preprocessing script to be cached for future use",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=int,
        default=10,
        help="Percentage of training data that should be used for validation if no validation is present in dataset.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=2,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--saving_steps",
        type=int,
        default=10,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--audio_column_name",
        type=str,
        default="audio",
        help="Column in the dataset that contains speech file path. Defaults to 'audio'",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="patrickvonplaten/wav2vec2-base-v2", 
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_cache_file_name",
        type=str,
        default=None,
        help="Path to the train cached file name",
    )
    parser.add_argument(
        "--validation_cache_file_name",
        type=str,
        default=None,
        help="Path to the validation cached file name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000
        
        ,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=32000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="./wav2vec2-pretrained-demo", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_gumbel_temperature",
        type=float,
        default=2.0,
        help="Maximum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--min_gumbel_temperature",
        type=float,
        default=0.5,
        help="Minimum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--gumbel_temperature_decay", type=float, default=0.999995, help="Decay of gumbel temperature during training."
    )
    parser.add_argument(
        "--max_duration_in_seconds",
        type=float,
        default=20.0,
        help="Filter out audio files that are longer than `max_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--min_duration_in_seconds",
        type=float,
        default=2.0,
        help="Filter out audio files that are shorter than `min_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=None,
        help=(
            "If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the"
            " use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.98,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-06,
        help="Epsilon for AdamW optimizer",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--mask_time_prob",
        type=float,
        default=0.65,
        help=(
            "Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked in the"
            " contrastive task. If omitted, will pull value from model config."
        ),
    )
    parser.add_argument(
        "--mask_time_length",
        type=int,
        default=5,
        help=(
            "Length of each vector mask span to mask along the time axis in the contrastive task."
            " If omitted, will pull value from model config."
        ),
    )
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def sliding_windows(data, window_size, sfreq):
    """Create sliding windows from data based on window size and sampling frequency."""
    step = int(window_size * sfreq)
    data_length = len(data)
    windows = [data[x:x + step] for x in range(0, data_length - step + 1, step)]
    return windows  # List of windowed segments
    
writer = SummaryWriter(log_dir="logging_events_real_data")

@dataclass
class DataCollatorForWav2Vec2Pretraining:
    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 5
    window_size_secs: float = 2.0 ########## bei windwos: 2 mal 16000=32000/input_values(320000) = 10 windows per batch

def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """Create a batch of sliding windows with padding, masking, and negative sampling."""
        
        # Prepare sliding window features
        wind_features = []
        for feature in features:
            input_values = feature["input_values"]
            windows = sliding_windows(input_values, self.window_size_secs, self.feature_extractor.sampling_rate)
            
            for window in windows:
                wind_input = self.feature_extractor(
                    window, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
                )
                wind_features.append({"input_values": wind_input.input_values[0]})
        
        # Pad the batch
        batch = self.feature_extractor.pad(
            wind_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Compute masking and negative sampling
        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        mask_indices_seq_length = int(mask_indices_seq_length)

        # Masked time step preparation
        if batch.get("attention_mask") is not None:
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # Sample mask time indices
        mask_time_indices = self._compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # Sample negative indices
        sampled_negative_indices = self._sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )

        # Convert mask indices to torch tensors and add to batch
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch

    def _compute_mask_indices(self, features_shape, mask_time_prob, mask_time_length, attention_mask=None):
        """Compute random mask indices for masked time steps."""
        batch_size, sequence_length = features_shape
        mask = np.zeros((batch_size, sequence_length), dtype=bool)
        
        for i in range(batch_size):
            if attention_mask is not None:
                seq_len = attention_mask[i].sum()  # Only consider non-padded length
            else:
                seq_len = sequence_length
            
            num_masked_spans = int(mask_time_prob * seq_len / mask_time_length)
            
            for _ in range(num_masked_spans):
                start_idx = np.random.randint(0, seq_len - mask_time_length)
                mask[i, start_idx:start_idx + mask_time_length] = True
        
        return mask

    def _sample_negative_indices(self, features_shape, num_negatives, mask_time_indices=None):
        """Sample negative indices for contrastive learning."""
        batch_size, sequence_length = features_shape
        negatives = np.zeros((batch_size, sequence_length, num_negatives), dtype=int)
        
        for i in range(batch_size):
            for j in range(sequence_length):
                if mask_time_indices is None or mask_time_indices[i, j]:
                    neg_indices = np.random.choice(sequence_length - 1, num_negatives, replace=False)
                    negatives[i, j] = [idx if idx < j else idx + 1 for idx in neg_indices]
        
        return negatives

# In main function:
def main():
    args = parse_args()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
    config = Wav2Vec2Config.from_pretrained(args.model_name_or_path)
    accelerator = Accelerator()
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()

    optimizer = AdamW(
        list(model.parameters()),
        lr=args.learning_rate,
        betas=[args.adam_beta1, args.adam_beta2],
        eps=args.adam_epsilon,
    )

    train_dataset = BIDSBrainVisionDataset(
    directory="data",
    output_dir="output_flac",
    feature_extractor=feature_extractor,
    target_sr=16000,
    debugging_mode=True
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    try:
        if os.path.isdir(args.output_dir):
            print(f"Loading model from {args.output_dir}...")
            model = Wav2Vec2ForPreTraining.from_pretrained(args.output_dir).to(device)
            processor = feature_extractor
        else:
            raise ValueError(f"No model found at {args.output_dir}")
        
    data_collator = DataCollatorForWav2Vec2Pretraining(
        model=model,
        feature_extractor=feature_extractor,
        padding=True,
        mask_time_prob=0.065,
        mask_time_length=10,
    )

# Prepare DataLoader with the data collator
    train_dataloader = DataLoader(
        train_dataset,  # Your dataset object
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Forward and backward pass, etc.
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            num_losses = batch["mask_time_indices"].sum()
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
            )
            percent_masked = num_losses / sub_attention_mask.sum()

            gumbel_temperature = max(
                    args.max_gumbel_temperature * args.gumbel_temperature_decay**completed_steps,
                    args.min_gumbel_temperature,
                )
            if hasattr(model, "module"):
                model.module.set_gumbel_temperature(gumbel_temperature)
            else:
                model.set_gumbel_temperature(gumbel_temperature)

            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()


            # 6. Log all results
            if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0:
                print("all good")            
                loss.detach()           
                outputs.contrastive_loss.detach()
                outputs.diversity_loss.detach()

                if accelerator.state.num_processes > 1:
                    loss = accelerator.gather_for_metrics(loss).sum()
                    outputs.contrastive_loss = accelerator.gather_for_metrics(outputs.contrastive_loss).sum()
                    outputs.diversity_loss = accelerator.gather_for_metrics(outputs.diversity_loss).sum()
                    percent_masked = accelerator.gather_for_metrics(percent_masked).sum()

                train_logs = {
                    "loss": (loss * args.gradient_accumulation_steps) / num_losses,
                    "constrast_loss": outputs.contrastive_loss / num_losses,
                    "div_loss": outputs.diversity_loss / num_losses,
                    "%_mask_idx": percent_masked / accelerator.num_processes,
                    "ppl": outputs.codevector_perplexity,
                    "lr": torch.tensor(optimizer.param_groups[0]["lr"]),
                    "temp": torch.tensor(gumbel_temperature),
                    "grad_norm": torch.tensor(grad_norm),
                }
                log_str = ""
                for k, v in train_logs.items():
                    log_str += "| {}: {:.3e}".format(k, v.item())

                if accelerator.is_local_main_process:
                    print("still all good")
                    progress_bar.write(log_str)

                    if is_wandb_available():
                        wandb.log(train_logs)

                    writer.add_scalar("loss/train", float(train_logs["loss"]), step)
                    writer.add_scalar("div_loss/train", float(train_logs["div_loss"]), step)
                    writer.add_scalar("learning_rate/train", float(train_logs["lr"].item()), step)
                    writer.add_scalar("grad_norm/train", float(train_logs["grad_norm"].item()), step)
                    writer.add_scalar("test_value", 1.0, 0)
                    writer.flush()

            # save model every `args.saving_steps` steps
            if (step + 1) % (args.gradient_accumulation_steps * args.saving_steps) == 0:
                if (args.push_to_hub and epoch < args.num_train_epochs - 1) or args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )

                if (args.push_to_hub and epoch < args.num_train_epochs - 1) and accelerator.is_main_process:
                    api.upload_folder(
                        commit_message=f"Training in progress epoch {epoch}",
                        folder_path=args.output_dir,
                        repo_id=repo_id,
                        repo_type="model",
                        token=args.hub_token,
                    )

            # if completed steps > `args.max_train_steps` stop
            if completed_steps >= args.max_train_steps:
                break

        # 7. Validate!
        model.eval()

        # init logs
        val_logs = {
            "val_loss": 0,
            "val_contrastive_loss": 0,
            "val_diversity_loss": 0,
            "val_num_losses": 0,
        }
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch.pop("sub_attention_mask", None)
                outputs = model(**batch)

            val_logs["val_loss"] += outputs.loss
            val_logs["val_contrastive_loss"] += outputs.contrastive_loss
            val_logs["val_diversity_loss"] += outputs.diversity_loss
            val_logs["val_num_losses"] += batch["mask_time_indices"].sum()

        # sum over devices in multi-processing
        if accelerator.num_processes > 1:
            val_logs = {k: accelerator.gather_for_metrics(v).sum() for k, v in val_logs.items()}

        val_logs = {k: v / val_logs["val_num_losses"] for k, v in val_logs.items()}

        log_str = ""
        for k, v in val_logs.items():
            log_str += "| {}: {:.3e}".format(k, v.item())

        if accelerator.is_local_main_process:
            progress_bar.write(log_str)
            if is_wandb_available():
                wandb.log(val_logs)

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                if args.push_to_hub:
                    api.upload_folder(
                        commit_message="End of training",
                        folder_path=args.output_dir,
                        repo_id=repo_id,
                        repo_type="model",
                        token=args.hub_token,
                    )

if __name__ == "__main__":
    main()

#tensorboard --logdir=logs