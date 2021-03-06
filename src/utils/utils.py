"""Utils for training models or get the data"""
import torch

def validate(model, valid_dl, loss_func):
    """Validation of the model"""
    batches_losses = [model.validation_step(batch, loss_func) for batch in valid_dl]
    epoch_loss = torch.cat(batches_losses).mean()

    return epoch_loss

def fit(epochs, model, loss_func, opt, train_dl):
    """Function for training models"""
    for _ in range(epochs):
        model.train()
        for batch in train_dl:
            loss_batch = model.training_step(batch, loss_func)
            loss_batch.backward()
            opt.step()
            opt.zero_grad()
