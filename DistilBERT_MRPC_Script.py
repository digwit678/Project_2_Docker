#!pip install -q torch transformers pytorch_lightning==1.9.5 datasets
from datetime import datetime
from typing import Optional
import numpy as np

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import datasets
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything

print("Entered Script")
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='GLUE Transformer Training Script')
parser.add_argument('--checkpoint_dir', type=str, default='models', help='Directory to save model checkpoints')
parser.add_argument('--learning_rate', type=float, default=5.89e-05, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation')
parser.add_argument('--weight_decay', type=float, default=0.0519, help='Weight decay rate')

args = parser.parse_args()


class GLUEDataModule(LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
            self,
            model_name_or_path: str,
            task_name: str = "mrpc",
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features


class GLUETransformer(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            num_labels: int,
            task_name: str,
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            eval_splits: Optional[list] = None,
            **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        # Existing logging code...

        # Aggregate predictions and labels
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()

        # Calculate and log validation loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        # Ensure 'labels' is a tensor and then calculate accuracy
        if isinstance(labels, torch.Tensor):
            total_predictions = labels.size(0)
        elif isinstance(labels, np.ndarray):
            total_predictions = labels.shape[0]
        else:
            total_predictions = len(labels)  # Assuming labels is a list

        correct_predictions = (preds == labels).sum()
        accuracy = correct_predictions / total_predictions
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        # Safely handle the 'outputs' list
        if not outputs:
            # Handle case when 'outputs' is empty
            print('No outputs to process for training epoch end.')
            self.log('train_loss_epoch', float('nan'), on_epoch=True, prog_bar=True, logger=True)
            return

        # Collect 'train_loss' from each output if available
        train_losses = [x.get('train_loss') for x in outputs if x.get('train_loss') is not None]

        # Calculate and log mean training loss if available
        if train_losses:
            train_loss_mean = torch.stack(train_losses).mean()
            self.log('train_loss_epoch', train_loss_mean, on_epoch=True, prog_bar=True, logger=True)
        else:
            # Handle case when 'train_losses' is empty
            print('No training losses recorded in this epoch.')
            self.log('train_loss_epoch', float('nan'), on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                                      eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


from torch.utils.tensorboard import SummaryWriter
import psutil

# Log the input and output resources
writer = SummaryWriter()
print("Before Seed")
seed_everything(42)
print("After Seed")
# Create a SummaryWriter instance for TensorBoard logging
# writer = SummaryWriter(log_dir=f'runs/glue_experiment_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}')

import pytorch_lightning as pl


# ... (rest of your imports and class definitions)

def run_training_iteration(iteration):
    print(f"Starting run_training_iteration {iteration}")
    # Set up data module and model
    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
        max_seq_length=128,  # Assuming a default value
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size  # Assuming eval batch size is the same as train batch size
    )
    dm.setup("fit")
    model = GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=dm.num_labels,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size
    )

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"checkpoint-{iteration}" + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,  # Save the best checkpoint based on the lowest validation loss
        verbose=True,
        monitor='val_loss',  # Or another metric that you wish to monitor
        mode='min'  # Or 'max' if the monitored metric should be maximized
    )

    # Set up trainer
    trainer = Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=TensorBoardLogger('lightning_logs', name=f'run_{iteration}'),
        default_root_dir=args.checkpoint_dir,
        callbacks=[checkpoint_callback]  # Add checkpoint_callback to the list of callbacks
    )

    # Run training
    trainer.fit(model, datamodule=dm)


print("Before training loop")
training_runs = 2
# Loop for 20 training runs
for i in range(1, training_runs):
    print(f"Iteration {i} started.")
    run_training_iteration(i)
    print(f"Iteration {i} finished.")

