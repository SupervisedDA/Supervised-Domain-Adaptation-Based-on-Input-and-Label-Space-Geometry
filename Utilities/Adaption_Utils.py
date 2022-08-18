import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
## CCSA
def CCSA_Loss(x, y, class_eq):
    margin =1
    x=torch.nn.functional.normalize(x, p=2, dim = 1)
    y=torch.nn.functional.normalize(y, p=2, dim = 1)
    dist = F.pairwise_distance(x, y)
    loss =class_eq * dist.pow(2) #if eqaul classes: penalize dist
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2) #if different - penalize if larger than margin
    return loss.mean()

## dSNE
def dSNE_Loss(fts, ys, ftt, yt):
    bs_tgt =len(fts)
    bs_src = len(ftt)
    embed_size = fts.shape[1]
    margin = 1
    fts=torch.nn.functional.normalize(fts, p=2, dim = 1)
    ftt=torch.nn.functional.normalize(ftt, p=2, dim = 1)

    fts_rpt = torch.broadcast_to(fts.unsqueeze( dim=0), size=(bs_tgt, bs_src, embed_size))
    ftt_rpt = torch.broadcast_to(ftt.unsqueeze( dim=1), size=(bs_tgt, bs_src, embed_size))

    # fts_rpt=fts.expand((bs_tgt, bs_src, embed_size))
    # ftt_rpt=ftt.expand((bs_tgt, bs_src, embed_size))

    dists = torch.sum(torch.square(ftt_rpt - fts_rpt), axis=2)
    # dists = torch.sqrt(torch.sum(torch.square(ftt_rpt - fts_rpt), axis=2))
    # dists = torch.sum(torch.abs(ftt_rpt - fts_rpt), axis=2)

    yt_rpt = torch.broadcast_to(yt.unsqueeze(dim=1), size=(bs_tgt, bs_src)).type(torch.int32)
    ys_rpt = torch.broadcast_to(ys.unsqueeze(dim=0), size=(bs_tgt, bs_src)).type(torch.int32)
    # yt_rpt = yt.expand((bs_tgt, bs_src)).type(torch.int32)
    # ys_rpt = ys.expand((bs_tgt, bs_src)).type(torch.int32)

    # y_same =((yt_rpt-ys_rpt)==0).type(torch.float32)
    # y_diff = 1-y_same

    y_same = ((yt_rpt - ys_rpt) == 0)
    y_diff = torch.logical_not(y_same)

    intra_cls_dists = dists * y_same
    inter_cls_dists = dists * y_diff

    (max_dists,_) = torch.max(dists, axis=1, keepdims=True)
    max_dists = max_dists.expand((bs_tgt, bs_src))

    revised_inter_cls_dists = torch.where(y_same, max_dists, inter_cls_dists)

    (max_intra_cls_dist,_) = torch.max(intra_cls_dists, axis=1)
    (min_inter_cls_dist,_) = torch.min(revised_inter_cls_dists, axis=1)

    # loss = torch.sum(torch.relu(max_intra_cls_dist - min_inter_cls_dist + margin))
    loss = torch.mean(torch.relu(max_intra_cls_dist - min_inter_cls_dist + margin))
    return loss


## UDA - CORAL
def CORAL_Loss(source, target):
    """
    From the paper, the vectors that compose Ds and Dt are D-dimensional vectors
    :param source: torch tensor: source data (Ds) with dimensions DxNs
    :param target: torch tensor: target data (Dt) with dimensons DxNt
    """

    d = source.size(1)  # d-dimensional vectors (same for source, target)

    source_covariance = compute_covariance(source)
    target_covariance = compute_covariance(target)


    # plt.figure();
    # plt.subplot(121)
    # plt.imshow(source.detach().cpu());
    # plt.subplot(122)
    # plt.imshow(target.detach().cpu()); plt.show()
    #
    #
    # plt.figure();
    # plt.subplot(121)
    # plt.imshow(source_covariance.detach().cpu());
    # plt.subplot(122)
    # plt.imshow(target_covariance.detach().cpu()); plt.show()

    # take Frobenius norm (https://pytorch.org/docs/stable/torch.html)
    loss = torch.norm(torch.mul((source_covariance - target_covariance),
                                (source_covariance - target_covariance)), p="fro")
    # loss=torch.norm(logm(source_covariance)- logm(target_covariance))
    # loss = torch.norm(torch.mm((source_covariance-target_covariance),
    # 							(source_covariance-target_covariance)), p="fro")
    loss = loss / (4 * d * d)
    return loss

def compute_covariance(data):
    """
    Compute covariance matrix for given dataset as shown in paper (eqs 2 and 3).
    :param data: torch tensor: input source/target data
    """

    # data dimensions: nxd (this for Ns or Nt)
    n = data.size(0)  # get batch size
    # print("compute covariance bath size n:", n)

    # check gpu or cpu support
    if data.is_cuda:
        device = torch.device("cuda")
        # device = torch.device("cuda:%g" % hp.GPU)
        device = data.device
    else:
        device = torch.device("cpu")

    # proper matrix multiplication for right side of equation (2)
    ones_vector = torch.ones(n).resize_(1, n).to(device=device)  # 1xN dimensional vector (transposed)
    one_onto_D = torch.mm(ones_vector, data)
    mult_right_terms = torch.mm(one_onto_D.t(), one_onto_D)
    mult_right_terms = torch.div(mult_right_terms, n)  # element-wise divison

    # matrix multiplication for left side of equation (2)
    mult_left_terms = torch.mm(data.t(), data)

    covariance_matrix = 1 / (n - 1) * torch.add(mult_left_terms, -1 * (mult_right_terms))

    return covariance_matrix


## UDA experiments
# MDD
# from dalib.adaptation.mdd import MarginDisparityDiscrepancy, ImageClassifier
#
# # DAN
# from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy, ImageClassifier
# from dalib.modules.kernels import GaussianKernel
#
# # DANN
# from dalib.modules.domain_discriminator import DomainDiscriminator
# from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
#
#
# # CDAN
# from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
# from cdannn import ConditionalDomainAdversarialLoss, ImageClassifier

## Ours
def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

def GetSacleFac(d,DiscardDiag=True):
    if DiscardDiag:
        [min_per_row, _] = torch.min(d , dim=1)
    else:
        [min_per_row, _] = torch.min(d + torch.max(d) * torch.eye(n=d.shape[0], m=d.shape[1]).to(device), dim=1)
    # return 1e-4+torch.max(min_per_row)
    return torch.max(min_per_row)

def GetWeightMatrix(pairwise_distances, weight_K=5, weight_kernel_scale='Fixed', weight_use_scale_grad=True):
    if weight_K > 0:
        sorted, indices = torch.sort(pairwise_distances)
        weight_matrix = torch.zeros_like(pairwise_distances)
        for k in range(weight_K):
            weight_matrix[range(weight_matrix.shape[0]), indices[:, k]] = (1 / weight_K)
        k_pairwise_distances = pairwise_distances * weight_matrix
        weight_matrix = torch.diag(1 / torch.sum(k_pairwise_distances, dim=1)) @ k_pairwise_distances

    else:
        # weight_matrix=nn.Softmax(dim=1)(-pairwise_distances/abs(K))
        if weight_kernel_scale == 'Fixed':
            scale = torch.tensor(1.0)
        if weight_kernel_scale == 'Auto':
            scale = GetSacleFac(pairwise_distances,DiscardDiag=True)
        if weight_kernel_scale == 'Median':
            scale = torch.median(pairwise_distances)
        if not (weight_use_scale_grad):
            scale = scale.item()
        # weight_matrix=nn.Softmax(dim=1)(-pairwise_distances/scale.item())
        weight_matrix = nn.Softmax(dim=1)(-pairwise_distances / scale)
        # weight_matrix=1- torch.diag(1/torch.sum(pairwise_distances,dim=1)) @ pairwise_distances
    return weight_matrix

def ToOneHot(Y,n_classes):
    OneHotY = torch.zeros(len(Y), n_classes).to(Y.device)
    OneHotY[range(len(Y)), Y] = 1
    return OneHotY

def GetCDCATerm(src_feature,tgt_feature,src_label,tgt_label,n_classes,hp):
    labels_s_dist = ToOneHot(src_label, n_classes)
    labels_t_dist = ToOneHot(tgt_label, n_classes)

    pairwise_cross_distances = torch.cdist(tgt_feature, src_feature) ** 2
    # --- E_{Dt}[ f_s(x)-f_t(x)|] --- (predictions based on NNs from the source dataset)
    weight_matrix_t2s = GetWeightMatrix(pairwise_cross_distances, weight_K=hp.weight_K,
                                        weight_kernel_scale=hp.weight_kernel_scale,
                                        weight_use_scale_grad=hp.weight_use_scale_grad)
    labels_t_nn = weight_matrix_t2s @ labels_s_dist  # pred=E[f(s)]
    # ---E_{Ds}[|f_s(x)-f_t(x)|] --- (predictions based on NNs from the target dataset)
    weight_matrix_s2t = GetWeightMatrix(pairwise_cross_distances.T, weight_K=hp.weight_K,
                                        weight_kernel_scale=hp.weight_kernel_scale,
                                        weight_use_scale_grad=hp.weight_use_scale_grad)
    labels_s_nn = weight_matrix_s2t @ labels_t_dist  # pred=E[f(t)]
    #
    loss_t_nn = nn.CrossEntropyLoss()(labels_t_nn, tgt_label)
    loss_s_nn = nn.CrossEntropyLoss()(labels_s_nn, src_label)
    loss_cdca = torch.min(loss_s_nn, loss_t_nn)
    return loss_cdca

def GetUDATerm(src_feature,tgt_feature,hp):
    if hp.UDA == 'CORAL':
        loss_uda= CORAL_Loss(src_feature, tgt_feature)
    if hp.UDA == 'MMD':
        # #using first+second moment:
        # loss_uda=torch.norm(torch.mean(src_feature,dim=0)-torch.mean(tgt_feature,dim=0))
        # loss_uda+=torch.norm(torch.mean(src_feature**2,dim=0)-torch.mean(tgt_feature**2,dim=0))
        # #using torch_two_sample package:
        # mmd=MMDStatistic(hp.BatchSize,hp.BatchSize)
        # loss_uda=mmd(tgt_feature,src_feature,alphas=[1])
        # mmd=MMDStatistic(n_t,n_t)
        # loss_uda=mmd(hidden_target,hidden_source[random.sample(range(n_s),n_t)],alphas=[1])
        # #using my implementation:
        loss_uda = MMD(src_feature, tgt_feature, \
                      kernel='rbf', mmd_kernel_scale=hp.mmd_kernel_scale,
                      mmd_kernel_bandwidth=hp.mmd_kernel_bandwidth)
    # if hp.UDA == 'MyAdv':
    #     # ---adverserial domain confusion loss---
    #     output_dc_src = modelDC(src_feature)
    #     output_dc_trg = modelDC(tgt_feature)
    #     # opt1 - no concat
    #     loss_uda = int(not (hp.UseTgtAdvOnly)) * nn.CrossEntropyLoss()(output_dc_src,
    #                                                                   torch.zeros(len(src_feature)).long().to(device)) + \
    #               nn.CrossEntropyLoss()(output_dc_trg, torch.ones(len(tgt_feature)).long().to(device))
    # if hp.UDA == 'DAN':
    #     loss_uda = mkmmd_loss(src_feature, tgt_feature)
    # if hp.UDA == 'DANN':
    #     loss_uda = domain_adv(src_feature, tgt_feature)
    #     domain_acc = domain_adv.domain_discriminator_accuracy
    # if hp.UDA == 'CDAN':
    #     # loss_uda = domain_adv(src_feature, tgt_feature)
    #     loss_uda = domain_adv(src_pred, src_feature, tgt_pred, tgt_feature)
    # if hp.UDA == 'MDD':
    #     loss_uda = -mdd(y_s, y_s_adv, y_t, y_t_adv)

    return loss_uda



