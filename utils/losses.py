import torch
import torch.nn.functional as F

# borrowed from https://github.com/nv-nguyen/template-pose/blob/ce1ffead1887b54efc8031e8e2442ba884e512ec/lib/losses/contrast_loss.py
def cosine_similarity(a, b, normalize=True):
    if normalize:
        w1 = a.norm(p=2, dim=1, keepdim=True)
        w2 = b.norm(p=2, dim=1, keepdim=True)
        sim_matrix = torch.mm(a, b.t()) / (w1 * w2.t()).clamp(min=1e-8)
    else:
        sim_matrix = torch.mm(a, b.t())

    sim_matrix = torch.clamp(sim_matrix, min=0.0005, max=0.9995)

    return sim_matrix

def get_contrast_loss(pos_feats, neg_feats, cls_labels):
    pos_sim = cosine_similarity(pos_feats, pos_feats)  # (N1,C)x(C,N1) = N1xN1
    neg_sim = cosine_similarity(pos_feats, neg_feats)  # (N1,C)x(C,N2) = N1xN2

    if torch.sum(cls_labels) > 0:
        neg_loss = torch.mean(-torch.log(1. - neg_sim))
        pos_loss = torch.mean(-torch.log(pos_sim))
        # pos_loss[pos_loss < 0] = 0
        # pos_loss = torch.mean(pos_loss)
    else:
        neg_loss = torch.Tensor([0.]).cuda()
        pos_loss = torch.Tensor([0.]).cuda()

    return pos_loss, neg_loss

def get_classification_loss(cls_preds, cls_labels):
    cls_loss = F.multilabel_soft_margin_loss(cls_preds, cls_labels)

    return cls_loss