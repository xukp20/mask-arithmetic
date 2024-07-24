from transformers import Trainer
import wandb
import os
import torch
from transformers.trainer_utils import EvalLoopOutput

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_history = []
        self.wandb_step = 1

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        loss = self.compute_loss(model, inputs)
        if isinstance(loss, dict):
            logs = {f"{key}": value.item() for key, value in loss.items()}
            loss_to_backward = loss['loss']  # Assuming 'loss' is the main loss to backward
        else:
            logs = {"loss": loss.item()}
            loss_to_backward = loss
        
        self.log(logs)
        
        if self.args.gradient_accumulation_steps > 1:
            loss_to_backward = loss_to_backward / self.args.gradient_accumulation_steps
        
        loss_to_backward.backward()
        return loss_to_backward.detach()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if isinstance(loss, dict):
            loss = loss['loss']  # Assuming 'loss' is the main loss

        loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        logits = outputs.logits
        labels = inputs['labels']
        
        return (loss, logits, labels)

    # def log(self, logs):
    #     if self.state.epoch is not None:
    #         logs["epoch"] = round(self.state.epoch, 2)

    #     output = {**logs, **{"step": self.wandb_step}}
    #     self.state.log_history.append(output)
    #     self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    #     wandb.log(logs, step=self.wandb_step)
    #     self.wandb_step += 1