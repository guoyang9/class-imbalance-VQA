import json

import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

import utils.config as config


def compute_score_with_logits(logits, labels):
    _, log_index = logits.max(dim=1, keepdim=True)
    scores = labels.gather(dim=1, index=log_index)
    return scores


def saved_for_eval(dataloader, results, question_ids, answer_preds):
    """ Save as a format accepted by the evaluation server. """
    _, answer_ids = answer_preds.max(dim=1)
    answers = [dataloader.dataset.label2ans[i] for i in answer_ids]
    for q, a in zip(question_ids, answers):
        entry = {
            'question_id': q.item(),
            'answer': a,
        }
        results.append(entry)
    return results


def train(model, optim, train_loader, loss_fn, tracker, writer, tb_count):
    loader = tqdm(train_loader, ncols=0)
    loss_trk = tracker.track('loss', tracker.MovingMeanMonitor(momentum=0.99))
    acc_trk = tracker.track('acc', tracker.MovingMeanMonitor(momentum=0.99))

    for v, q, a, a_m, a_s, bias, q_id in loader:
        v = v.cuda()
        q = q.cuda()
        a = a.cuda()
        a_m = a_m.cuda()
        a_s = a_s.cuda()
        bias = bias.cuda()

        pred, mask, hidden, _ = model(v, q)
        dict_args = {'bias': bias, 'hidden': hidden}
        if config.use_miu:
            dict_args['miu'] = a_s
            dict_args['mask'] = a_m
        loss = loss_fn(pred, a, **dict_args)
        if config.use_mask:
            loss_mask = F.binary_cross_entropy_with_logits(mask, a_m)
            loss += loss_mask

        # writer.add_scalars('data/losses', {
        #     'loss_back': loss - loss_mask,
        #     'loss_mask': loss_mask
        # }, tb_count)
        # tb_count += 1

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()
        optim.zero_grad()

        batch_score = compute_score_with_logits(pred, a.data)

        fmt = '{:.4f}'.format
        loss_trk.append(loss.item())
        acc_trk.append(batch_score.mean())
        loader.set_postfix(loss=fmt(loss_trk.mean.value),
                            acc=fmt(acc_trk.mean.value))
    return tb_count


def evaluate(model, dataloader, epoch=0, write=False):
    score = 0
    results = [] # saving for evaluation
    for v, q, a, a_m, a_s, _, q_id in tqdm(dataloader, leave=False):
        v = v.cuda()
        q = q.cuda()
        pred, _, _, _ = model(v, q)
        if write:
            results = saved_for_eval(dataloader, results, q_id, pred)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
    score = score / len(dataloader.dataset)

    if write:
        print("saving prediction results to disk...")
        result_file = 'vqa_{}_{}_{}_{}_results.json'.format(
            config.task, config.test_split, config.version, epoch)
        with open(result_file, 'w') as fd:
            json.dump(results, fd)
    return score
