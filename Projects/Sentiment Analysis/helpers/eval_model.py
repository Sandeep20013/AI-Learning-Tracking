from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score,confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import torch
def evaluate_model(
    model,
    dataloader,
    device = 'cpu',
    threshold = 0.5,
    plot_cm = True
    ):
    model.eval()
    y_true, y_pred_probs = [], []
    with torch.inference_mode():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            test_pred = model(X_batch).squeeze(1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred_probs.extend(torch.sigmoid(test_pred).cpu().numpy())
    y_pred = [1 if p >= threshold else 0 for p in y_pred_probs]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_probs)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nEvaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    if plot_cm:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}



