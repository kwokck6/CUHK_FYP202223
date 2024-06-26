import matplotlib.pyplot as plt
from datasets import ShopeeTL
import metrics
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, CenterCrop, ConvertImageDtype, Normalize
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from triplet_loss import TripletLoss
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# metadata and constants
np.random.seed(1028)
torch.manual_seed(1028)
torch.cuda.manual_seed_all(1028)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training on {device}")
n_epoch = 10
batch_size = 3
model_name = "resnet50_tl"
writer = SummaryWriter(model_name)

# preprocessing and batching
img_preprocess = Compose([
    Resize(384),
    CenterCrop(384),
    ConvertImageDtype(torch.float),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
dataset = {"train": ShopeeTL(img_transform=img_preprocess, split="train"),
           "val": ShopeeTL(img_transform=img_preprocess, split="val"),
           "test": ShopeeTL(img_transform=img_preprocess)}

print("Creating data loaders...")
def get_weights():
    counts = [0] * dataset["train"].num_classes
    categories = dataset["train"].img_labels["is_positive"]
    for cate, count in categories.value_counts().items():
        counts[dataset["train"].class_to_idx[cate]] = count
    num_df_data = float(sum(counts))

    for idx, count in enumerate(counts):
        if count != 0:
            counts[idx] = num_df_data / float(count)
    return torch.DoubleTensor([counts[dataset["train"].class_to_idx[cate]] for cate in categories])

train_weights = get_weights()
train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
train_loader = DataLoader(dataset["train"], batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset["val"], batch_size, shuffle=False)
test_loader = DataLoader(dataset["test"], batch_size, shuffle=False)

# models and optimizers
num_classes = dataset["train"].num_classes
print(f"num_classes = {num_classes}")
model = resnet50(num_classes=num_classes).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, verbose=True)
loss_fn = TripletLoss().to(device)

# training
train_losses = []
val_losses = []
train_scores = []
val_scores = []
train_ndcgs = []
val_ndcgs = []

best_val_score = 0.
for e in tqdm(range(n_epoch), desc="Training model..."):  # progress bar for counting epochs    
    model.train()
    train_loss = 0.
    train_total = 0
    train_preds = []
    for (train_img1, train_img2, train_img3), train_text, train_y in tqdm(train_loader, desc=f"EPOCH {e + 1} - training"):  # progress bar for training
        optimizer.zero_grad()
        train_pred1 = model(train_img1.to(device))
        train_pred2 = model(train_img2.to(device))
        train_pred3 = model(train_img3.to(device))
        loss = loss_fn(train_pred1, train_pred2, train_pred3)
        train_loss += loss.item() * len(train_y)
        train_total += len(train_y)
        train_preds.append(train_pred1.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
    train_losses.append(train_loss / train_total)
    dist_mat, idx_mat = metrics.neighbours(np.vstack(train_preds))
    train_score, train_threshold = metrics.best_f1_score(dataset['train'].img_labels, dist_mat, idx_mat)
    train_scores.append(train_score)

    val_loss = 0
    val_total = 0
    val_preds = []
    with torch.no_grad():
        for (val_img1, val_img2, val_img3), val_text, val_y in tqdm(val_loader, desc=f"EPOCH {e + 1} - validation"):  # progress bar for validation
            val_pred1 = model(val_img1.to(device))
            val_pred2 = model(val_img2.to(device))
            val_pred3 = model(val_img3.to(device))
            val_loss += loss_fn(val_pred1, val_pred2, val_pred3).item() * len(val_y)
            val_total += len(val_y)
            val_preds.append(val_pred1.detach().cpu().numpy())
        val_losses.append(val_loss / val_total)
        dist_mat, idx_mat = metrics.neighbours(np.vstack(val_preds))
        val_score, val_threshold = metrics.best_f1_score(dataset['val'].img_labels, dist_mat, idx_mat)
        val_scores.append(val_score)

    scheduler.step(val_loss)

    print("\n")
    if val_score >= best_val_score:
        print("New best model found on validation set!")
        best_val_score = val_score
        torch.save(model.state_dict(), f"models/{model_name}.pth")

    print(f"EPOCH {e + 1} | Train loss: {train_loss / train_total:.4f} | Train F1: {train_score:.4f} | Validation loss: {val_loss / val_total:.4f} | Validation F1: {val_score:.4f}")
    writer.add_scalars('Loss', {"train": train_loss / train_total, "val": val_loss / val_total}, e)
    writer.add_scalars('F1 Score', {"train": train_score, "val": val_score}, e)

# save loss and accuracy plots
fig = plt.figure()
plt.plot(range(1, n_epoch + 1), train_losses, label="Training loss")
plt.plot(range(1, n_epoch + 1), val_losses, label="Validation loss")
plt.legend()
plt.title(f"Loss of {model_name}")
plt.savefig(f"plots/{model_name}_loss.png")

fig = plt.figure()
plt.plot(range(1, n_epoch + 1), train_scores, label="Training F1 score")
plt.plot(range(1, n_epoch + 1), val_scores, label="Validation F1 score")
plt.legend()
plt.title(f"F1 Score of {model_name}")
plt.savefig(f"plots/{model_name}_score.png")

# prediction
# load best model
if os.path.exists(f'models/{model_name}.pth'):
    model.load_state_dict(torch.load(f"models/{model_name}.pth", map_location="cuda"))
model.eval()
predictions = []
targets = []
with torch.no_grad():
    for (val_img1, val_img2, val_img3), val_text, val_img3 in tqdm(val_loader, desc="Retrieving best embedding..."):
        val_pred = model(val_img1.to(device))
        val_preds.append(val_pred.detach().cpu().numpy())
    val_dist_mat, val_idx_mat = metrics.neighbours(np.vstack(val_preds))
    val_score, val_threshold = metrics.best_f1_score(dataset['val'].img_labels, val_dist_mat, val_idx_mat)

    test_scores = []
    test_loss = 0. 
    test_count = 0
    test_total = 0
    for (test_img1, test_img2, test_img3), test_text, test_y in tqdm(test_loader, desc="Predicting..."):
        test_pred1 = model(test_img1.to(device))
        test_pred2 = model(test_img2.to(device))
        test_loss += loss_fn(test_pred1, test_pred2, test_y).item() * len(test_y)
        test_total += len(test_y)
        predictions.append(test_pred1)
        targets.append(test_y[0])
    test_pred = model(torch.vstack(predictions).to(device))
    test_dist_mat, test_idx_mat = metrics.neighbours(test_pred)
    test_score = metrics.f1_score(dataset['test'].img_labels, test_dist_mat, test_idx_mat, val_threshold)
    test_ndcg = metrics.ndcg(dataset['test'].img_labels, idx_mat)
    print(f"Loss: {test_loss / test_total:.4f} | Threshold: {val_threshold:.2f} | F1 Score: {test_score:.4f} | NDCG: {test_ndcg:.4f}")

predictions = torch.vstack(predictions)
targets = torch.hstack(targets)

writer.add_embedding(predictions, metadata=targets.tolist(), tag="Test embeddings")
writer.flush()
writer.close()
