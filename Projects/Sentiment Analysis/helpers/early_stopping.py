import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, path='checkpoint.pt', verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement before stopping.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): File path to save the best model.
            verbose (bool): If True, prints messages during training.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0  # Counts how many epochs have passed without improvement
        self.best_score = None  # Tracks the best (lowest) validation loss seen so far
        self.early_stop = False  # Flag to indicate if early stopping should trigger

    def __call__(self, val_loss, model):
        """
        Call this method after each epoch with the current validation loss and model.
        """
        score = -val_loss  # Since lower loss is better, we negate to treat higher as better

        # First call: set the initial best score and save the model
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)

        # If no significant improvement
        elif score < self.best_score + self.min_delta:
            self.counter += 1  # Increase counter since no improvement
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                # If patience is exceeded, trigger early stopping
                self.early_stop = True

        # If improved
        else:
            self.best_score = score  # Update best score
            self._save_checkpoint(model)  # Save model checkpoint
            self.counter = 0  # Reset counter

    def _save_checkpoint(self, model):
        """
        Saves model state_dict to disk if validation loss improves.
        """
        if self.verbose:
            print(f"Validation loss improved. Saving model to {self.path}")
        torch.save(model.state_dict(), self.path)