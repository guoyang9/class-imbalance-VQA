import torch.nn as nn
import utils.config as config

from modules.fc import FCNet
from modules.classifier import SimpleClassifier
from modules.attention import Attention, NewAttention
from modules.answer_tune import AnswerMask
from modules.language_model import WordEmbedding, QuestionEmbedding


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, answer_mask):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.answer_mask = answer_mask

    def forward(self, v, q, v_mask=None):
        """
        Forward=

        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]
        v_mask: [batch, num_objs] - mainly useful for css

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb, v_mask)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        if config.use_mask:
            mask = self.answer_mask(q_emb)
            logits *= mask
        else:
            mask = None
        return logits, mask, joint_repr, w_emb


def build_baseline(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    a_mask = AnswerMask(num_hid, dataset.num_ans_candidates)
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, a_mask)


def build_baseline_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    a_mask = AnswerMask(num_hid, dataset.num_ans_candidates)
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, a_mask)
