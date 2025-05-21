def accuracy_fn(preds, labels):
    return (preds == labels).sum().item() / len(labels)