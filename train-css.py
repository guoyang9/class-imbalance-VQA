"""
This is a re-implementation for CSS in CVPR 2020.
Note that this code does not support test in VQA datasets.
"""
import os
import sys
import json
import random
import argparse
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils.utils as utils
import utils.config as config
from utils.dataset import Dictionary, VQAFeatureDataset


import modules.base_model as base_model
from utils.losses import LearnedMixin, LearnedMixinH, FocalLoss, Plain

seed = 1111
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True


_keep_qtype = True
_topq = 1 # number of masked words
_topv = -1 # number of masked objects
_top_hint = 9 # number of hint objects
_qvp = 5 # ratio of q_bias and v_bias
_mode = 'q_v_debias' # ['q_debias', 'v_debias', 'q_v_debias']


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


def loss_and_back(loss_fn, model, optim, pred, mask, a, a_s, a_m, dict_args, compute_grad=None):
    """ The loss and optim are folded here. """
    if config.use_miu:
        dict_args['miu'] = a_s
        dict_args['mask'] = a_m
    loss = loss_fn(pred, a, **dict_args)
    if config.use_mask:
        loss_mask = F.binary_cross_entropy_with_logits(mask, a_m)
        loss += loss_mask

    grad = None # for possible gradient computation
    if compute_grad is not None:
        grad = torch.autograd.grad((pred * (a > 0).float()).sum(),
                                   compute_grad, create_graph=True)[0]

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optim.step()
    optim.zero_grad()

    return loss, grad


def q_debias(model, optim, loss_fn, dataset, v, q, a, a_m, a_s, bias, type_mask, notype_mask):
    sen_mask = type_mask if _keep_qtype else notype_mask

    # first train
    pred, mask, hidden, word_emb = model(v, q)
    dict_args = {'bias': bias, 'hidden': hidden}

    loss1, word_grad = loss_and_back(loss_fn, model, optim,
                    pred, mask, a, a_s, a_m, dict_args, compute_grad=word_emb)
    batch_score = compute_score_with_logits(pred, a.data)

    # second train
    word_grad_cam = word_grad.sum(dim=2)
    word_grad_cam_sigmoid = torch.exp(word_grad_cam * sen_mask)
    word_grad_cam_sigmoid = word_grad_cam_sigmoid * sen_mask
    word_idx = torch.argsort(word_grad_cam_sigmoid, dim=1, descending=True)[:, :_topq]
    q2 = torch.scatter(q, 1, word_idx, dataset.dictionary.padding_idx) # word mask
    pred, _, _, _ = model(v, q2)

    pred_idx = torch.argsort(pred, dim=1, descending=True)[:, :5]
    a2 = torch.scatter(a, 1, pred_idx, 0)

    # third train
    pred, mask, hidden, _ = model(v, q2)
    dict_args = {'bias': bias, 'hidden': hidden}
    loss2, _ = loss_and_back(loss_fn, model, optim, pred, mask, a2, a_s, a_m, dict_args)

    return batch_score, loss1 + loss2


def v_debias(model, optim, loss_fn, v, q, a, a_m, a_s, bias, hint):
    # first train
    pred, mask, hidden, _ = model(v, q)
    dict_args = {'bias': bias, 'hidden': hidden}

    loss1, visual_grad = loss_and_back(loss_fn, model, optim,
                                    pred, mask, a, a_s, a_m, dict_args, compute_grad=v)
    batch_score = compute_score_with_logits(pred, a.data)

    # second train
    v_mask = torch.zeros(v.shape[0], v.shape[1], device=v.device)
    v_idx = torch.argsort(hint, dim=1, descending=True)[:, :_top_hint]
    visual_grad_cam = visual_grad.sum(dim=2)
    v_grad = visual_grad_cam.gather(dim=1, index=v_idx)

    if _topv == -1:
        v_grad_score, _ = v_grad.sort(1, descending=True)
        v_grad_score = F.softmax(v_grad_score , dim=1) # why time 10?
        v_grad_sum = torch.cumsum(v_grad_score, dim=1)
        v_grad_mask = (v_grad_sum <= 0.65).long()
        v_grad_mask[:, 0] = 1 # make sure at least mask one object
        v_mask_ind = v_grad_mask * v_idx
        for x in range(a.shape[0]):
            num = len(torch.nonzero(v_grad_mask[x]))
            v_mask[x].scatter_(0, v_mask_ind[x, :num], 1)
    else:
        v_grad_idx = torch.argsort(v_grad, dim=1, descending=True)[:, :_topv]
        v_star = v_idx.gather(1, v_grad_idx)
        v_mask.scatter_(1, v_star, 1.0)

    pred, _, _, _ = model(v, q, v_mask)

    pred_idx = torch.argsort(pred, dim=1, descending=True)[:, :5]
    a2= torch.scatter(a, 1, pred_idx, 0)

    # third train
    v_mask = 1 - v_mask
    pred, mask, hidden, _ = model(v, q, v_mask)
    dict_args = {'bias': bias, 'hidden': hidden}
    loss2, _ = loss_and_back(loss_fn, model, optim, pred, mask, a2, a_s, a_m, dict_args)

    return batch_score, loss1 + loss2


def train(model, optim, train_loader, loss_fn, tracker, writer, tb_count):
    loader = tqdm(train_loader, ncols=0)
    loss_trk = tracker.track('loss', tracker.MovingMeanMonitor(momentum=0.99))
    acc_trk = tracker.track('acc', tracker.MovingMeanMonitor(momentum=0.99))

    dataset = train_loader.dataset
    for v, q, hint, type_mask, notype_mask, a, a_m, a_s, bias, q_id in loader:
        v = v.cuda().requires_grad_()
        q = q.cuda()
        a = a.cuda()
        a_m = a_m.cuda()
        a_s = a_s.cuda()
        bias = bias.cuda()
        hint = hint.cuda()
        type_mask = type_mask.cuda()
        notype_mask = notype_mask.cuda()

        random_num = random.randint(1, 10)
        if random_num <= _qvp:
            batch_score, loss = q_debias(model, optim, loss_fn,
                        dataset, v, q, a, a_m, a_s, bias, type_mask, notype_mask)
        else:
            batch_score, loss = v_debias(model, optim, loss_fn,
                        v, q, a, a_m, a_s, bias, hint)

        fmt = '{:.4f}'.format
        loss_trk.append(loss.item())
        acc_trk.append(batch_score.mean())
        loader.set_postfix(loss=fmt(loss_trk.mean.value),
                           acc=fmt(acc_trk.mean.value))
    return tb_count


def evaluate(model, dataloader, epoch=0, write=False):
    score = 0
    results = [] # saving for evaluation
    for v, q, _, _, _, a, a_m, a_s, _, q_id in tqdm(dataloader, leave=False):
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


class CssVQAFeatureDataset(VQAFeatureDataset):
    def __init__(self, name, dictionary):
        VQAFeatureDataset.__init__(self, name, dictionary)
        self.name = name
        data_name = 'cp' + config.version \
                if config.cp_data else config.version
        self.hintscore = json.load(open(os.path.join(
            'css-data', data_name+'_hintscore_'+name+'.json'), 'r'))
        self.type_mask = json.load(open(os.path.join(
            'css-data', data_name + '_type_mask.json'), 'r'))
        self.notype_mask = json.load(open(os.path.join(
            'css-data', data_name + '_notype_mask.json'), 'r'))

    def __getitem__(self, index):
        entry = self.entries[index]
        if config.in_memory:
            features = self.features[entry['image']]
            spatials = self.spatials[entry['image']]
        else:
            features, spatials = self.load_image(entry['image'])

        question_id = entry['question_id']
        question = entry['q_token']
        answer = entry['answer']
        q_type = answer['question_type']
        labels = answer['labels']
        scores = answer['scores']
        mask_labels = self.answer_mask[q_type]['mask']
        mask_scores = self.answer_mask[q_type]['weight']

        hint = torch.tensor(self.hintscore[str(question_id)])

        type_mask = 0
        notype_mask = 0
        if self.name == 'train':
            type_mask = torch.tensor(self.type_mask[str(question_id)])
            notype_mask = torch.tensor(self.notype_mask[str(question_id)])

        target = torch.zeros(self.num_ans_candidates)
        target_mask = torch.zeros(self.num_ans_candidates)
        target_score = torch.ones(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
            target_mask.scatter_(0, mask_labels, 1.0)
            target_score.scatter_(0, mask_labels, mask_scores)

        bias = entry['bias'] if 'bias' in entry else 0
        return features, question, \
               hint, type_mask, notype_mask, \
               target, target_mask, target_score, bias, question_id

    def __len__(self):
        return len(self.entries)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of running epochs')
    parser.add_argument('--loss-fn', type=str, default='Plain',
                        help='chosen loss function')
    parser.add_argument('--num-hid', type=int, default=1024,
                        help='number of dimension in last layer')
    parser.add_argument('--model', type=str, default='baseline_newatt',
                        help='model structure')
    parser.add_argument('--name', type=str, default='exp0',
                        help='saved model name')
    parser.add_argument('--name-new', type=str, default=None,
                        help='combine with fine-tune')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='training batch size')
    parser.add_argument('--fine-tune', action='store_true',
                        help='fine tuning with our loss')
    parser.add_argument('--resume', action='store_true',
                        help='whether resume from checkpoint')
    parser.add_argument('--not-save', action='store_true',
                        help='do not overwrite the old model')
    parser.add_argument('--test', dest='test_only', action='store_true',
                        help='test one time')
    parser.add_argument('--eval-only', action='store_true',
                        help='evaluate on the val set one time')
    parser.add_argument("--gpu", type=str, default='0',
                        help='gpu card ID')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    print_keys = ['cp_data', 'version', 'train_set', 'use_mask', 'use_miu', 'ft_lr']
    print_dict = {key: getattr(config, key) for key in print_keys}
    pprint(print_dict, width=150)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    if 'log' not in args.name:
        args.name = 'logs/' + args.name + '.pth'
    if args.test_only or args.fine_tune or args.eval_only:
        args.resume = True
    if args.resume and not args.name:
        raise ValueError("Resuming requires folder name!")
    if args.resume:
        logs = torch.load(args.name)
        print("loading logs from {}".format(args.name+ '.pth'))

    # ------------------------DATASET CREATION--------------------
    dictionary = Dictionary.load_from_file(config.dict_path)
    if args.test_only:
        eval_dset = CssVQAFeatureDataset('test', dictionary)
    else:
        train_dset = CssVQAFeatureDataset('train', dictionary)
        eval_dset = CssVQAFeatureDataset('val', dictionary)
    if config.train_set == 'train+val' and not args.test_only:
        train_dset = train_dset + eval_dset
        eval_dset = CssVQAFeatureDataset('test', dictionary)
    if args.eval_only:
        eval_dset = CssVQAFeatureDataset('val', dictionary)

    tb_count = 0
    writer = SummaryWriter() # for visualization

    if not config.train_set == 'train+val' and 'LM' in args.loss_fn:
        utils.append_bias(train_dset, eval_dset, len(eval_dset.label2ans))

    # ------------------------MODEL CREATION------------------------
    constructor = 'build_{}'.format(args.model)
    model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()
    model.w_emb.init_embedding(config.glove_embed_path)

    model = nn.DataParallel(model).cuda()
    optim = torch.optim.Adamax(model.parameters())

    if args.loss_fn == 'Plain':
        loss_fn = Plain()
    elif args.loss_fn == 'LMH':
        loss_fn = LearnedMixinH(hid_size=args.num_hid).cuda()
    elif args.loss_fn == 'LM':
        loss_fn = LearnedMixin(hid_size=args.num_hid).cuda()
    elif args.loss_fn == 'Focal':
        loss_fn = FocalLoss()
    else:
        raise RuntimeError('not implement for {}'.format(args.loss_fn))

    # ------------------------STATE CREATION------------------------
    eval_score, best_val_score, start_epoch, best_epoch = 0.0, 0.0, 0, 0
    tracker = utils.Tracker()
    if args.resume:
        model.load_state_dict(logs['model_state'])
        optim.load_state_dict(logs['optim_state'])
        if 'loss_state' in logs:
            loss_fn.load_state_dict(logs['loss_state'])
        start_epoch = logs['epoch']
        best_epoch = logs['epoch']
        best_val_score = logs['best_val_score']
        if args.fine_tune:
            print('best accuracy is {:.2f} in baseline'.format(100 * best_val_score))
            args.epochs = start_epoch + 10 # 10 more epochs
            for params in optim.param_groups:
                params['lr'] = config.ft_lr

            # if you want save your model with a new name
            if args.name_new:
                if 'log' not in args.name_new:
                    args.name = 'logs/' + args.name_new+ '.pth'
                else:
                    args.name = args.name_new+ '.pth'

    eval_loader = DataLoader(eval_dset,
                    args.batch_size, shuffle=False, num_workers=4)
    if args.test_only or args.eval_only:
        evaluate(model, eval_loader, write=True)
    else:
        train_loader = DataLoader(
            train_dset, args.batch_size, shuffle=True, num_workers=4)

        for epoch in range(start_epoch, args.epochs):
            print("training epoch {:03d}".format(epoch))
            tb_count = train(model, optim, train_loader, loss_fn, tracker, writer, tb_count)

            if not (config.train_set == 'train+val' and epoch in range(args.epochs - 3)):
                # save for the last three epochs
                write = True if config.train_set == 'train+val' else False
                print("validating after epoch {:03d}".format(epoch))
                model.train(False)
                eval_score = evaluate(model, eval_loader, epoch, write=write)
                model.train(True)
                print("eval score: {:.2f} \n".format(100 * eval_score))

            if eval_score > best_val_score:
                best_val_score = eval_score
                best_epoch = epoch
                results = {
                    'epoch': epoch + 1,
                    'best_val_score': best_val_score,
                    'model_state': model.state_dict(),
                    'optim_state': optim.state_dict(),
                    'loss_state': loss_fn.state_dict(),
                }
                if not args.not_save:
                    torch.save(results, args.name)
        print("best accuracy {:.2f} on epoch {:03d}".format(
            100 * best_val_score, best_epoch))