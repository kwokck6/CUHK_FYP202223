import matplotlib.pyplot as plt
from contrastive_loss import ContrastiveLoss
from datasets import ShopeeCL
import metrics
import numpy as np
import os
import seaborn as sns
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaModel
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
model_name = "roberta_cl"
writer = SummaryWriter(model_name)

# preprocessing
padding_idx = 1
img_preprocess = lambda x: 0
text_preprocess = lambda s: s.translate(s.maketrans('-\'', '  ', '!#$%^&*()[]\{\}\"<>?,.;:\\~+=0123456789')).lower()  # remove punctuation marks and numbers
dataset = {'train': ShopeeCL(img_transform=img_preprocess, text_transform=text_preprocess, split='train'),
           'val': ShopeeCL(img_transform=img_preprocess, text_transform=text_preprocess, split='val'),
           'test': ShopeeCL(img_transform=img_preprocess, text_transform=text_preprocess)}

# init loader and batching
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

val_loader = DataLoader(dataset["val"], batch_size, shuffle=False)
test_loader = DataLoader(dataset["test"], batch_size, shuffle=False)

# models and optimizers
num_classes = dataset["train"].num_classes
print(f"num_classes = {num_classes}")
input_dim = 768
tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')
model = RobertaModel.from_pretrained('distilroberta-base').to(device)  # embed_dim = 128
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, verbose=True)
loss_fn = ContrastiveLoss(margin=1.0)

def batch_preprocess(batch_text):
    # takes in a list of strings and gives a dictionary
    return tokenizer(batch_text, padding="longest", truncation=True, max_length=256, return_tensors='pt')

# training
train_losses = []
val_losses = []
train_scores = []
val_scores = []

best_val_score = 0.
for e in tqdm(range(n_epoch), desc="Training model..."):  # progress bar for counting epochs
    train_weights = get_weights()
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
    train_loader = DataLoader(dataset["train"], batch_size, sampler=train_sampler)

    model.train()
    train_loss = 0.
    train_total = 0
    train_preds = []
    for train_img1, (train_text1, train_text2), (train_y1, train_y2) in tqdm(train_loader, desc=f"EPOCH {e + 1} - training"):  # progress bar for training
        optimizer.zero_grad()
        train_input1 = batch_preprocess(list(train_text1))
        train_text1 = train_input1['input_ids'].to(device)
        train_attn_mask1 = train_input1['attention_mask'].to(device)
        train_pred1 = model(input_ids=train_text1, attention_mask=train_attn_mask1)

        train_input2 = batch_preprocess(list(train_text2))
        train_text2 = train_input2['input_ids'].to(device)
        train_attn_mask2 = train_input2['attention_mask'].to(device)
        train_pred2 = model(input_ids=train_text2, attention_mask=train_attn_mask2)

        # print(train_pred1.pooler_output.shape)

        train_y = (train_y1 == train_y2).long().to(device)
        loss = loss_fn(train_pred1.pooler_output, train_pred2.pooler_output, train_y)
        train_loss += loss.item() * len(train_y)
        train_total += len(train_y)
        train_preds.append(train_pred1.pooler_output.detach().cpu().numpy())
        # loss.backward()
        # optimizer.step()
    train_losses.append(train_loss / train_total)
    train_dist_mat, train_idx_mat = metrics.neighbours(np.vstack(train_preds))
    train_score, train_threshold = metrics.best_f1_score(dataset['train'].img_labels, train_dist_mat, train_idx_mat)
    train_scores.append(train_score)

    val_loss = 0.
    val_total = 0
    val_preds = []
    with torch.no_grad():
        for val_img, (val_text1, val_text2), (val_y1, val_y2) in tqdm(val_loader, desc=f"EPOCH {e + 1} - validation"):  # progress bar for validation
            val_input1 = batch_preprocess(list(val_text1))
            val_text1 = val_input1['input_ids'].to(device)
            val_attn_mask1 = val_input1['attention_mask'].to(device)
            val_pred1 = model(input_ids=val_text1, attention_mask=val_attn_mask1)
            
            val_input2 = batch_preprocess(list(val_text2))
            val_text2 = val_input2['input_ids'].to(device)
            val_attn_mask2 = val_input2['attention_mask'].to(device)
            val_pred2 = model(input_ids=val_text2, attention_mask=val_attn_mask2)
            
            val_y = (val_y1 == val_y2).long().to(device)
            val_preds.append(val_pred1.pooler_output.detach().cpu().numpy())
            val_loss += loss_fn(val_pred1.pooler_output, val_pred2.pooler_output, val_y).item() * len(val_y)
            val_total += len(val_y)
        val_losses.append(val_loss / val_total)
        val_dist_mat, val_idx_mat = metrics.neighbours(np.vstack(val_preds))
        val_score, val_threshold = metrics.best_f1_score(dataset['val'].img_labels, val_dist_mat, val_idx_mat)
        val_scores.append(val_score)
        
    scheduler.step(val_loss)
    
    print("\n")
    if val_score >= best_val_score:
        print("New best model found on validation set!")
        best_val_score = val_score
        torch.save(model.state_dict(), f"models/{model_name}.pth")

    print(f"EPOCH {e + 1} | Train loss: {train_loss / train_total:.4f} | Train score: {train_score:.4f} | Validation loss: {val_loss / val_total:.4f} | Validation score: {val_score:.4f}")
    writer.add_scalars('Loss', {"train": train_loss / train_total, "val": val_loss / val_total}, e)
    writer.add_scalars('F1 Score', {"train": train_score * 100, "val": val_score * 100}, e)

# save loss and score plots
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
plt.title(f"Accuracy of {model_name}")
plt.savefig(f"plots/{model_name}_score.png")

# prediction
# load best model
if os.path.exists(f'models/{model_name}.pth'):
    model.load_state_dict(torch.load(f"models/{model_name}.pth", map_location="cuda"))
model.eval()
val_preds = []
predictions = []
targets = []
with torch.no_grad():
    for val_img, (val_text1, val_text2), val_y in tqdm(val_loader, desc="Retrieving best embedding..."):
        val_input1 = batch_preprocess(list(val_text1))
        val_text1 = val_input1['input_ids'].to(device)
        val_attn_mask1 = val_input1['attention_mask'].to(device)
        val_pred1 = model(input_ids=val_text1, attention_mask=val_attn_mask1)
        val_preds.append(val_pred1.pooler_output.detach().cpu().numpy())
    val_dist_mat, val_idx_mat = metrics.neighbours(np.vstack(val_preds))
    val_score, val_threshold = metrics.best_f1_score(dataset['val'].img_labels, val_dist_mat, val_idx_mat)
    
    test_scores = []
    test_loss = 0.
    test_total = 0
    for test_img, (test_text1, test_text2), (test_y1, test_y2) in tqdm(test_loader, desc="Predicting..."):
        test_input1 = batch_preprocess(list(test_text1))
        test_text1 = test_input1['input_ids'].to(device)
        test_attn_mask1 = test_input1['attention_mask'].to(device)
        test_pred1 = model(input_ids=test_text1, attention_mask=test_attn_mask1)

        test_input2 = batch_preprocess(list(test_text2))
        test_text2 = test_input2['input_ids'].to(device)
        test_attn_mask2 = test_input2['attention_mask'].to(device)
        test_pred2 = model(input_ids=test_text2, attention_mask=test_attn_mask2)

        test_y = (test_y1 == test_y2).long().to(device)
        test_loss += loss_fn(test_pred1.pooler_output, test_pred2.pooler_output, test_y.long().to(device)).item() * len(test_y)
        test_total += len(test_y)
        predictions.append(test_pred1.pooler_output.detach().cpu().numpy())
        targets.append(test_y)
    test_dist_mat, test_idx_mat = metrics.neighbours(np.vstack(predictions))
    test_score, test_threshold = metrics.f1_score(dataset['test'].img_labels, test_dist_mat, test_idx_mat, val_threshold)
    test_ndcg = metrics.ndcg(dataset['test'].img_labels, test_idx_mat)
    print(f"Loss: {test_loss / test_total:.4f} | Threshold: {val_threshold:.2f} | F1 Score: {test_score:.4f} | NDCG: {test_ndcg:.4f}")

predictions = torch.tensor(np.vstack(predictions))
targets = torch.hstack(targets)

writer.add_embedding(predictions, metadata=targets.tolist(), tag="Test embeddings")
writer.flush()
writer.close()
