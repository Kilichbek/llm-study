import os
import torch
import wandb
from tqdm import tqdm

class Trainer:
    def __init__(
            self, 
            model, # language model
            optimizer, # optimizer
            train_loader, # training data loader
            val_loader, # validation data loader
            max_iters, # max number of iterations
            eval_iters, # number of iterations to evaluate
            eval_interval,  # number of iterations between evaluations
            checkpoint_path=None, # path to save model checkpoints
            device="cpu", # device to run model on
        ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.train_set_iter = iter(self.train_loader)
        self.val_set_iter = iter(self.val_loader)
    
    def compute_loss(self, data_iter, epoch, split='train'):
        self.model.eval()
        running_loss = 0.0
        with tqdm(desc=f"Step {epoch} - {split}", unit="iters", total=self.eval_iters) as pbar:
            for i in range(self.eval_iters):
                x, y = next(data_iter)
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.model.loss(logits, y)
                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss / (i + 1))
                pbar.update()
        loss = running_loss / self.eval_iters
        return loss

    def train(self):
        """
        Training loop for a language model.
        """
        best_val_loss = float("inf")
        for epoch in range(self.max_iters):
            self.model.train()
            running_loss = 0.0
            x, y = next(self.train_set_iter)
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.model.loss(logits, y)
            loss.backward()
            self.optimizer.step()

            if epoch % self.eval_interval == 0:
                train_loss = self.compute_loss(self.train_set_iter, epoch, split='train')
                val_loss = self.compute_loss(self.val_set_iter, epoch, split='val')
                print(f"Step {epoch} - train loss: {train_loss:.4f} - val loss: {val_loss:.4f}")
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.checkpoint_path is not None:
                        model_name = f"model-{epoch}.pt"
                        model_save_path = os.path.join(self.checkpoint_path, model_name)
                        torch.save(self.model.state_dict(), model_save_path)
                        print(f"Step {epoch} - saved checkpoint to {model_save_path}")