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

    def train(self, epoch):
        """
        Training loop for a language model.
        """
        self.model.train()
        running_loss = 0.0
        with tqdm(desc=f"Iter {epoch} - train", unit="iters", total=len(self.train_loader)) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.model.loss(logits, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss / (i + 1))
                pbar.update()

        loss = running_loss / len(self.train_loader)
        return loss
    
    def evaluate(self, epoch):
        """
        Evaluation loop for a language model.
        """
        self.model.eval()
        running_loss = 0.0
        with tqdm(desc=f"Iter {epoch} - validation", unit="iters", total=len(self.val_loader)) as pbar:
            for i, (x, y) in enumerate(self.val_loader):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.model.loss(logits, y)
                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss / (i + 1))
                pbar.update()

        loss = running_loss / len(self.val_loader)
        return loss

    def run(self):
        """
        Training loop for a language model.
        """
        best_val_loss = float("inf")
        for epoch in range(self.max_iters):
            train_loss = self.train(epoch)
            
            if epoch % self.eval_interval == 0:
                val_loss = self.evaluate(epoch)
                wandb.log({'val_loss': val_loss})
                print(f"Epoch {epoch} - train loss: {train_loss:.4f} - val loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_name = f"model-{epoch}.pt"
                    checkpoint_path = os.path.join(self.checkpoint_path, checkpoint_name)
                    torch.save(self.model.state_dict(), checkpoint_path)
            wandb.log({'train_loss': train_loss})

          