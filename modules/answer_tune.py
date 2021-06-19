import torch
import torch.nn as nn
import torch.nn.functional as F


class AnswerMask(nn.Module):
    def __init__(self, num_hid, num_ans, dropout=0.5):
        """ dropout rate for Plain should be 0.0"""
        super(AnswerMask, self).__init__()
        layers = [
            nn.Dropout(dropout),
            nn.Linear(num_hid, num_ans),
        ]

        self.mask = nn.Sequential(*layers)

    def forward(self, q):
        q = self.mask(q)
        # q = torch.sigmoid(q).clone()
        q = F.softplus(q).clamp(max=1.0)
        return q
