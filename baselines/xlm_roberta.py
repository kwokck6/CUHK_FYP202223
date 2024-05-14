import matplotlib.pyplot as plt
from datasets import FashProd, SigirFarfetch, Shopee
import numpy as np
import os
import seaborn as sns
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
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
batch_size = 2
model_name = "xlm_roberta_fash"
writer = SummaryWriter(model_name)

# preprocessing
padding_idx = 1
img_preprocess = lambda x: 0
text_preprocess = lambda s: s.translate(s.maketrans('-\'', '  ', '!#$%^&*()[]\{\}\"<>?,.;:\\~+=0123456789')).lower()  # remove punctuation marks and numbers

dataset = {'train': FashProd(img_transform=img_preprocess, text_transform=text_preprocess, split='train'),
           'val': FashProd(img_transform=img_preprocess, text_transform=text_preprocess, split='val'),
           'test': FashProd(img_transform=img_preprocess, text_transform=text_preprocess, split='test')}

# init loader and batching
print("Creating data loaders...")
def get_weights():
    counts = [0] * dataset["train"].num_classes
    categories = dataset["train"].img_labels["category"]
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
input_dim = 768
tokenizer = XLMRobertaTokenizerFast.from_pretrained('roberta-base')
model = XLMRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_classes).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, verbose=True)

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
    for train_img, train_text, train_y in train_loader:  # progress bar for training
        optimizer.zero_grad()
        train_input = batch_preprocess(list(train_text))
        train_text = train_input['input_ids'].to(device)
        train_attn_mask = train_input['attention_mask'].to(device)
        train_y = train_y.to(device)
        train_pred = model(input_ids=train_text, attention_mask=train_attn_mask, labels=train_y)
        loss = train_pred.loss
        train_loss += loss.item() * len(train_y)
        train_count += (train_pred.logits.softmax(1).argmax(1) == train_y).sum().item()
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
        for val_img, val_text, val_y in tqdm(val_loader, desc=f"EPOCH {e + 1} - validation"):  # progress bar for validation
            val_input = batch_preprocess(list(val_text))
            val_text = val_input['input_ids'].to(device)
            val_attn_mask = val_input['attention_mask'].to(device)
            val_y = val_y.to(device)
            val_pred = model(input_ids=val_text, attention_mask=val_attn_mask, labels=val_y)
            val_loss += val_pred.loss.item() * len(val_y)
            val_count += (val_pred.softmax(1).argmax(1) == val_y.to(device)).sum().item()
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
        test_text = test_input['input_ids'].to(device)
        test_attn_mask = test_input['attention_mask'].to(device)
        test_y = test_y.to(device)
        test_pred = model(input_ids=test_text, attention_mask=test_attn_mask, labels=test_y)
        test_loss += test_pred.loss.item() * len(test_y)
        test_count += (test_pred.logits.softmax(1).argmax(1) == test_y).sum().item()
        test_total += len(test_y)
        predictions.append(test_pred)
        targets.append(test_y)
    print(f"Loss: {test_loss / test_total:.4f} | Accuracy: {test_count / test_total * 100:.2f}%")

predictions = torch.vstack(predictions)
targets = torch.hstack(targets)

writer.add_embedding(predictions, metadata=targets.tolist(), tag="Test embeddings")
writer.flush()
writer.close()
