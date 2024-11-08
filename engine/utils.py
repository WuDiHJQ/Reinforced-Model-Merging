import os
import torch
import random
import evaluate
import open_clip
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = self.new_iter()

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = self.new_iter()
            data = next(self._iter)
        return data

    def new_iter(self):
        return iter(self.dataloader)


def load_clip_features(class_names, model, device):
    """Create CLIP target labels for class names. Return a normalized tensor of shape (num_classes, 512)."""
    text_inputs = torch.cat(
        [open_clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def reset_random(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def round_list(my_list, significant_figures):

    rounded_list = []

    for number in my_list:
        rounded_list.append(round(number, significant_figures))

    return rounded_list


def round_nestedList(nested_list, significant_figures):

    rounded_nestedList = []
    for sublist in nested_list:

        if isinstance(sublist[0], list):
            rounded_sublist = round_nestedList(sublist, significant_figures)
        else:
            rounded_sublist = round_list(sublist, significant_figures)

        rounded_nestedList.append(rounded_sublist)

    return rounded_nestedList


def predict_mulChoice(transformer, batch):

    # Compute log p(y|x)
    (
        score_ofChoices,
        logProbs_ofAllChoicesIds,
        len_allChoices,
    ) = compute_logProb_ofAllChoices(
        transformer,
        batch["input_ids"],
        batch["input_mask"],
        batch["all_choices_ids"],
        batch["all_choices_mask"],
    )

    _, predicted_choice = torch.max(score_ofChoices, dim=1)

    return (
        predicted_choice.cpu().numpy().tolist(),
        round_nestedList(score_ofChoices.cpu().numpy().tolist(), 5),
        round_nestedList(logProbs_ofAllChoicesIds.cpu().numpy().tolist(), 4),
        len_allChoices.cpu().numpy().tolist(),
    )


def compute_logProb_ofAllChoices(
    transformer,
    input_ids,
    input_masks,
    allChoices_ids,
    allChoices_masks,
):
    """
    Computes log probabilties for all the choices. This should be used when the number of
    choices is greater than 1. It computes the encoder hidden representations once, broadcasts
     it to match the number of choices, and then computes the log prob for each choice.

    Args:
        input_ids: [batch_size, max_input_len]
        input_masks: [batch_size, max_input_len]
        allChoices_ids: [batch_size x num_choices, max_choice_len]
        allChoices_masks: [batch_size x num_choices, max_choice_len]

    Returns:
        logProbs_ofAllChoices: [batch_size, num_choices]
        logProbs_ofAllChoicesIds_zeroOutPadIds: [batch_size, num_choices, max_choice_len]
        len_allChoices: [batch_size, num_choices ]
    """
    encoder_outputs = transformer.get_encoder()(input_ids, input_masks)

    num_choices = allChoices_ids.shape[0] // input_masks.shape[0]

    input_masks = torch.repeat_interleave(input_masks, num_choices, dim=0)
    encoder_outputs = (torch.repeat_interleave(encoder_outputs[0], num_choices, dim=0),)

    transformer_outputs = transformer(
        attention_mask=input_masks,
        encoder_outputs=encoder_outputs,
        labels=allChoices_ids,
    )

    logits_ofAllChoices = transformer_outputs[1].float()
    maxChoice_len = logits_ofAllChoices.shape[1]
    vocab_size = logits_ofAllChoices.shape[-1]

    logProbs_ofAllChoices_ids = -F.cross_entropy(
        logits_ofAllChoices.view(-1, vocab_size),
        allChoices_ids.view(-1),
        reduction="none",
    )

    logProbs_ofAllChoices_ids = logProbs_ofAllChoices_ids.reshape(-1, num_choices, maxChoice_len)
    allChoices_masks = allChoices_masks.reshape(-1, num_choices, maxChoice_len)

    logProbs_ofAllChoicesIds_zeroOutPadIds = logProbs_ofAllChoices_ids * allChoices_masks
    logProbs_ofAllChoices = torch.sum(logProbs_ofAllChoicesIds_zeroOutPadIds, dim=2)

    len_allChoices = torch.sum(allChoices_masks, dim=2)

    return (logProbs_ofAllChoices,logProbs_ofAllChoicesIds_zeroOutPadIds,len_allChoices)


def compute_forward_loss(transformer, batch):
    transformer_outputs = transformer(
        input_ids=batch["input_ids"],
        attention_mask=batch["input_mask"],
        labels=batch["target_ids"],
    )

    # [batch_size, max_target_len, vocab_size]
    target_logits = transformer_outputs[1].float()
    vocab_size = target_logits.shape[-1]

    # Compute the log probability of the ids for all choices with respect to the logits
    # [batch_size x max_target_len]
    logProbs_ofTargetIds = F.cross_entropy(
        target_logits.reshape(-1, vocab_size),
        batch["target_ids"].reshape(-1),
        reduction="none",
    )
    # Zero out log_probs for target_ids with no loss
    target_mask = batch["target_mask"].reshape(-1)
    logProbs_ofTargetIds_zeroOutPadIds = logProbs_ofTargetIds * target_mask

    loss = torch.sum(logProbs_ofTargetIds_zeroOutPadIds) / torch.sum(target_mask)

    return loss


def validate_CV(model, data_iter, clip_features, iter_idx, data_scale):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    correct = 0
    total = 0

    losses = []
    criterion = nn.CrossEntropyLoss().to(device)
    clip_features = clip_features[iter_idx] if iter_idx is not None else clip_features

    ################ set iterator ################
    if data_scale == 1.0:
        iterator = data_iter.new_iter()
    else:
        iterator = range(int(data_scale*len(data_iter.dataloader)))

    with torch.no_grad(), autocast():
        for i in iterator:
            if data_scale == 1.0:
                images, target = i
            else:
                images, target = data_iter.next()
            images, target = images.to(device), target.to(device)
            encodings = model.encode_image(images)
            encodings = encodings[iter_idx] if iter_idx is not None else encodings
            normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)       
            logits = (100.0 * normed_encodings @ clip_features.T)
            pred = logits.argmax(dim=1)
            loss = criterion(logits, target)
            losses.append(loss.item())

            for gt, p in zip(target, pred):
                is_correct = (gt == p).item()
                correct += is_correct

            total += images.shape[0]
            
    val_acc = correct / total
    val_loss = np.mean(losses)
    
    return val_acc, val_loss


def validate_NLP(transformer, data_iter, data_scale):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transformer.eval()
    metric = evaluate.load("accuracy")

    ################ set iterator ################
    if data_scale == 1.0:
        iterator = data_iter.new_iter()
    else:
        iterator = range(int(data_scale * 100))

    with torch.no_grad():
        for i in iterator:
            batch = i if data_scale == 1.0 else data_iter.next()
            
            with autocast(dtype=torch.bfloat16):
                predicted_choice,_,_,_, = predict_mulChoice(transformer, batch)

            metric.add_batch(
                predictions=predicted_choice,
                references=batch["lbl"],
            )

    score = metric.compute()

    return score["accuracy"]


def save_figure(env_name, run_num):
    fig_width = 10
    fig_height = 6

    # smooth out rewards to get a smooth and a less smooth (var) plot lines
    window_len_smooth = 20
    min_window_len_smooth = 1
    linewidth_smooth = 1.5
    alpha_smooth = 1

    window_len_var = 5
    min_window_len_var = 1
    linewidth_var = 2
    alpha_var = 0.1

    # make directory for saving figures
    run_dir = os.path.join("runs", env_name)
    log_path = os.path.join(run_dir, 'log_%d.csv'%run_num)
    fig_path = os.path.join(run_dir, 'fig_%d.png'%run_num)

    print("loading data from : " + log_path)
    data = pd.read_csv(log_path)
    data = pd.DataFrame(data)
    print("data shape : ", data.shape)

    ax = plt.gca()

    # smooth out rewards to get a smooth and a less smooth (var) plot lines
    data['reward_smooth'] = data['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
    data['reward_var'] = data['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

    # plot the lines
    data.plot(kind='line', x='timestep', y='reward_smooth', ax=ax,color='red', linewidth=linewidth_smooth, alpha=alpha_smooth)
    data.plot(kind='line', x='timestep', y='reward_var', ax=ax,color='red', linewidth=linewidth_var, alpha=alpha_var)

    # keep alternate elements (reward_smooth_i) in the legend
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    new_labels = []
    for i in range(len(handles)):
        if(i%2 == 0):
            new_handles.append(handles[i])
            new_labels.append(labels[i])
    ax.legend(new_handles, new_labels, loc=2)

    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Rewards", fontsize=12)

    plt.title(env_name, fontsize=14)

    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)
   
    plt.savefig(fig_path)
    print("figure saved at : ", fig_path)