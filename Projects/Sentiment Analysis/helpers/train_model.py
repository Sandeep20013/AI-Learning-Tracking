import torch
from  torch import nn
from helpers.metrics import accuracy_fn
from timeit import default_timer as timer
from helpers.timer_help import print_train_time
from tqdm.auto import tqdm
def train_step(model:nn.Module,
                train_dataloader: torch.utils.data,
                optimizer: torch.optim,
                loss_fn: torch.nn.Module,
                device: torch.device):
    model.train()
    train_loss, train_acc = 0,0
    for X, y in train_dataloader:
        X = X.to(device)
        y = y.to(device)
        # Reset optimizer to 0 
        optimizer.zero_grad()
        # Make preds
        y_pred = model(X).squeeze(1)
        # Calcualte loss
        loss = loss_fn(y_pred, y.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X.size(0) # Multiply just in case last batch size is less than 32
        preds = (torch.sigmoid(y_pred) >= 0.5).float()
        train_acc += accuracy_fn(preds, y) * X.size(0)
    avg_loss = train_loss / len(train_dataloader.dataset)
    avg_acc = train_acc / len(train_dataloader.dataset)
    print(f"Train Loss: {avg_loss:.4f} | Train Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc

def validate_step(model: nn.Module,
                  test_dataloader: torch.utils.data,
                  loss_fn: torch.nn.Module,
                  device: torch.device):
    test_loss, test_acc = 0, 0
    total_samples = 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            test_pred = model(X_test).squeeze(1)
            # Pass raw logits to loss_fn
            test_loss += loss_fn(test_pred, y_test.float()).item() * X_test.size(0)
            preds = (torch.sigmoid(test_pred) >= 0.5).float()
            test_acc += accuracy_fn(preds, y_test) * X_test.size(0)
            total_samples += y_test.size(0)
    avg_loss = test_loss / total_samples
    avg_acc = test_acc / total_samples
    print(f"Test Loss: {avg_loss:.4f} |  Test Accuracy: {avg_acc:.4f}")
    return avg_loss

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    epochs=5,
    early_stopping=None,
    show_progress=True
):
    torch.manual_seed(42)
    model = model.to(device)
    start = timer()

    progress = tqdm(range(epochs)) if show_progress else range(epochs)
    for epoch in progress:
        print(f"Epoch {epoch + 1}/{epochs}")
        train_step(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate_step(model, val_loader, loss_fn, device)
        
        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    end = timer()
    print_train_time(start, end, str(device))
    return model
