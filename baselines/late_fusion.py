import matplotlib.pyplot as plt
from datasets import FashProd, SigirFarfetch
import numpy as np
import os
import seaborn as sns
import torch
from torch.nn import CrossEntropyLoss, Linear, Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnext50_32x4d
from torchvision.transforms import Compose, Resize, CenterCrop, ConvertImageDtype, Normalize
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, get_linear_schedule_with_warmup
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# metadata and constants
sns.set()
np.random.seed(1028)
torch.manual_seed(1028)
torch.cuda.manual_seed_all(1028)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training on {device}")

n_epoch = 1
batch_size = 1
model_name = "late_fusion_fash"
writer = SummaryWriter(model_name)

# preprocessing
padding_idx = 1
text_preprocess = lambda s: s.translate(s.maketrans('-\'', '  ', '!#$%^&*()[]\{\}\"<>?,.;:\\~+=0123456789')).lower()
img_preprocess = Compose([
    Resize(224),
    CenterCrop(224),
    ConvertImageDtype(torch.float),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
dataset = {"train": FashProd(img_transform=img_preprocess, text_transform=text_preprocess, split="train"),
           "val": FashProd(img_transform=img_preprocess, text_transform=text_preprocess, split="val"),
           "test": FashProd(img_transform=img_preprocess, text_transform=text_preprocess, split="test")}

# init loader and batching
print("Creating data loaders...")
def get_weights():
    counts = [0] * dataset["train"].num_classes
    categories = dataset["train"].img_labels["category"]
    for cate, count in categories.value_counts().items():
        counts[dataset["train"].class_to_idx[cate]] = count
    num_df_data = float(sum(counts))
    counts = [(num_df_data / float(count) if count else 0) for count in counts]
    return torch.DoubleTensor([counts[dataset["train"].class_to_idx[cate]] for cate in categories])

def collate_fn(batch):
    img, text, target = zip(*batch)
    return torch.stack(img), text, torch.tensor(target)

train_weights = get_weights()
train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
train_loader = DataLoader(dataset["train"], batch_size, sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(dataset["val"], batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(dataset["test"], batch_size, shuffle=False, collate_fn=collate_fn)

# models and optimizers
num_classes = dataset["train"].num_classes
print(f"num_classes = {num_classes}")
input_dim = 768

class LateFusion(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=num_classes)
        self.resnext = resnext50_32x4d(num_classes=num_classes)
        # self.weight_avg = Linear(2, 1)

    def forward(self, img_input, text_input):
        img_output = self.resnext(img_input).softmax(1)
        text_output = self.roberta(**text_input).logits.softmax(1)
        score = (img_output + text_output) / 2
        return score
    
model = LateFusion(num_classes).to(device)
tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataset['train']) * n_epoch)
loss_fn = CrossEntropyLoss().to(device)

def batch_preprocess(batch_text):
    # takes in a list of strings and gives a dictionary
    return tokenizer(batch_text, padding="longest", truncation=True, max_length=256, return_tensors='pt')

# training
train_losses = []
val_losses = []
train_accs = []
val_accs = []

best_val_loss = float("inf")
for e in tqdm(range(n_epoch), desc="Training model..."):  # progress bar for counting epochs
    model.train()
    train_loss = 0.
    train_count = 0
    train_total = 0
    for train_img, train_text, train_y in tqdm(train_loader, desc=f"EPOCH {e + 1} - training"):  # progress bar for tra$        optimizer.zero_grad()
        train_input = batch_preprocess(list(train_text))
        train_input['labels'] = train_y
        train_pred = model(train_img.to(device), train_input.to(device))
        loss = loss_fn(train_pred, train_y.long().to(device))
        train_loss += loss.item() * len(train_y)
        train_count += (train_pred.argmax(1) == train_y.to(device)).sum().item()
        train_total += len(train_y)
        loss.backward()
        optimizer.step()
    train_losses.append(train_loss / train_total)
    train_acc = train_count / train_total
    train_accs.append(train_acc)

    val_loss = 0.
    val_count = 0
    val_total = 0
    with torch.no_grad():
        for val_img, val_text, val_y in tqdm(val_loader, desc=f"EPOCH {e + 1} - validation"):  # progress bar for valid$            val_input = batch_preprocess(list(val_text))
            val_input = batch_preprocess(list(val_text))
            val_input['labels'] = val_y
            val_pred = model(val_img.to(device), val_input.to(device))
            val_loss += loss_fn(val_pred, val_y.long().to(device)).item() * len(val_y)
            val_count += (val_pred.argmax(1) == val_y.to(device)).sum().item()
            val_total += len(val_y)
        val_losses.append(val_loss / val_total)
        val_acc = val_count / val_total
        val_accs.append(val_acc)

        scheduler.step(val_loss)

    print("\n")
    if val_loss <= best_val_loss:
        print("New best model found on validation set!")
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"models/{model_name}.pth")

    print(f"EPOCH {e + 1} | Train loss: {train_loss / train_total:.4f} | Train accuracy: {train_acc * 100:.2f}% | Validation loss: {val_loss / val_total:.4f} | Validation accuracy: {val_acc * 100:.2f}%")
    writer.add_scalars('Loss', {"train": train_loss / train_total, "val": val_loss / val_total}, e)
    writer.add_scalars('Accuracy', {"train": train_acc * 100, "val": val_acc * 100}, e)

# save loss and accuracy plots
fig = plt.figure()
plt.plot(range(1, n_epoch + 1), train_losses, label="Training loss")
plt.plot(range(1, n_epoch + 1), val_losses, label="Validation loss")
plt.legend()
plt.title(f"Loss of {model_name}")
plt.savefig(f"plots/{model_name}_loss.png")

fig = plt.figure()
plt.plot(range(1, n_epoch + 1), train_accs, label="Training accuracy")
plt.plot(range(1, n_epoch + 1), val_accs, label="Validation accuracy")
plt.legend()
plt.title(f"Accuracy of {model_name}")
plt.savefig(f"plots/{model_name}_acc.png")

# prediction
# load best model
if os.path.exists(f'models/{model_name}.pth'):
    model.load_state_dict(torch.load(f"models/{model_name}.pth", map_location="cuda"))
model.eval()
predictions = []
targets = []
with torch.no_grad():
    test_accs = []
    test_loss = 0.
    test_count = 0
    test_total = 0

    for test_img, test_text, test_y in tqdm(test_loader, desc="Predicting..."):
        test_input = batch_preprocess(list(test_text))
        test_input['labels'] = test_y
        test_pred = model(test_img.to(device), test_input.to(device))
        test_loss += loss_fn(test_pred, test_y.to(device).long()).item() * len(test_y)
        test_count += (test_pred.argmax(1) == test_y.to(device)).sum().item()
        test_total += len(test_y)
        predictions.append(test_pred)
        targets.append(test_y)
    print(f"Loss: {test_loss / test_total:.4f} | Accuracy: {test_count / test_total * 100:.2f}%")

predictions = torch.vstack(predictions)
targets = torch.hstack(targets)

writer.add_embedding(predictions, metadata=targets.tolist(), tag="Test embeddings")
writer.flush()
writer.close()
