import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

model_name = 'resnext50_32x4d_fash'
num_epoch = 10
train_losses = [1.8283, 1.0423, 0.8799, 0.8117, 0.7625, 0.7414, 0.7072, 0.6902, 0.6736, 0.6508]
train_accs   = [0.5404, 0.7020, 0.7432, 0.7604, 0.7735, 0.7816, 0.7909, 0.7966, 0.7999, 0.8048]
val_losses   = [1.2026, 1.0740, 0.8795, 0.8154, 0.7812, 0.7497, 0.7385, 0.7319, 0.6795, 0.6901]
val_accs     = [0.6726, 0.6790, 0.7450, 0.7626, 0.7733, 0.7738, 0.7815, 0.7875, 0.7916, 0.7977]

fig = plt.figure()
plt.plot(range(1, num_epoch + 1), train_losses, label='Training loss')
plt.plot(range(1, num_epoch + 1), val_losses, label='Validation loss')
plt.title(f'Loss of {model_name}')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.savefig(f'{model_name}_loss.png')

fig = plt.figure()
plt.plot(range(1, num_epoch + 1), train_accs, label='Training accuracy')
plt.plot(range(1, num_epoch + 1), val_accs, label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title(f'Accuracy of {model_name}')
plt.savefig(f'{model_name}_acc.png')
