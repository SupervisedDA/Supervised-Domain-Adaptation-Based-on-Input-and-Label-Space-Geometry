import os
exec(open(os.path.join("Scripts", "Imports.py")).read())
from Utilities.Configuration_Utils import *
from Utilities.Data_Utils import *
from Utilities.Adaption_Utils import *
from Utilities.Model_Utils import *
from Utilities.Evaluation_Utils import *

parser = argparse.ArgumentParser()
parser=GetParser(parser)

def RunCofiguration(hp):
    print('--------- Beginning experiment %s ---------'%hp.ExpName)
    ## dataset
    print('Loading datasets')
    train_dataset, test_dataset, val_dataset, n_classes = GetDatasets(hp)

    [train_loader, test_loader, val_loader] = [DataLoader(x, batch_size=hp.BatchSize,
                                                          shuffle=True, num_workers=0, drop_last=False) for x in \
                                               [train_dataset, test_dataset, val_dataset]]

    [train_iter, test_iter, val_iter] = [MyForeverDataIterator(x, hp.BatchSize) for x in
                                         [train_loader, test_loader, val_loader]]

    ## model+optimizer
    device = torch.device("cuda:%g" % hp.GPU if (torch.cuda.is_available() and hp.GPU >= 0) else "cpu")
    net = GetModel(hp, device)

    self = net
    net.to(device)

    if hp.Optimizer=='Adadelta':
        optimizer = torch.optim.Adadelta(net.parameters(),lr=hp.LearningRate, weight_decay=hp.WD)

    if hp.Optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=hp.LearningRate, momentum=0.9, weight_decay=hp.WD,
                                    nesterov=True)
    elif hp.Optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=hp.LearningRate, weight_decay=hp.WD)

    ## Logging
    # hp.LogToWandb = LogToWandb
    if hp.LogToWandb:
        wandb.login()
        wandb.init(project=hp.ProjectName, entity="orkatz")
        wandb.run.name = hp.ExpName
        # wandb.watch(model, log="all")
        config = wandb.config
        config.args = vars(hp)

    ## Train model
    train_losses_src, train_losses_tgt, test_losses_src, test_losses_tgt = [], [], [], []
    train_accs_src, train_accs_tgt, test_accs_src, test_accs_tgt = [], [], [], []
    losses_dd, losses_nn = [], []

    if n_classes == -1:
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
    else:
        criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(hp.NumberOfBatches),
                desc="Training models for %s" % hp.ExpName,
                bar_format="{desc:20}{percentage:2.0f}{r_bar}")
    for batch in pbar:
        net.train()
        src_img, src_label, tgt_img, tgt_label = next(train_iter)
        src_img, tgt_img = (x.to(device, dtype=torch.float) for x in [src_img, tgt_img])

        src_label, tgt_label = (x.to(device, dtype=torch.long) for x in [src_label, tgt_label])
        src_pred, src_feature = net(src_img)
        tgt_pred, tgt_feature = net(tgt_img)

        optimizer.zero_grad()

        # -----Explicit losses-----
        # ---src explicit loss---
        loss_s = criterion(src_pred, src_label)
        # ---tgt explicit loss---
        loss_t = criterion(tgt_pred, tgt_label)

        # -----DA losses-----
        if hp.Method == 'SDA_IO':
            loss_cdca = GetCDCATerm(src_feature, tgt_feature, src_label, tgt_label, n_classes, hp)
            loss_uda = GetUDATerm(src_feature, tgt_feature, hp)
            loss = hp.Coeffs[0] * loss_s + hp.Coeffs[1] * loss_t + hp.Coeffs[2] * loss_uda + hp.Coeffs[3] * loss_cdca
        if hp.Method == 'dSNE':
            loss_dsne = dSNE_Loss(src_feature, src_label, tgt_feature, tgt_label)
            loss = hp.Coeffs[0] * loss_s + hp.Coeffs[1] * loss_t + hp.Coeffs[4] * loss_dsne
        if hp.Method == 'CCSA':
            loss_csca = CCSA_Loss(src_feature, tgt_feature,
                                  (src_label == tgt_label).float())
            loss = hp.Coeffs[0] * loss_s + hp.Coeffs[4] * loss_csca

        loss.backward()
        optimizer.step()

        # ----- Logger -----
        # train_losses_src.append(loss_s.item())
        # train_losses_tgt.append(loss_t.item())
        # if batch % hp.ValMonitoringFactor == 0:
        #     test_losses_src.append(test_loss_s.item())
        #     test_losses_tgt.append(test_loss_t.item())
        # losses_dd.append(loss_dd.item())
        # losses_nn.append(loss_nn.item())
        # train_accs_src.append(train_acc_src)
        # train_accs_tgt.append(train_acc_tgt)
        # if batch % hp.ValMonitoringFactor == 0:
        #     test_accs_src.append(test_acc_src)
        #     test_accs_tgt.append(test_acc_tgt)

    ##
    SrcTestAccs, TgtTestAccs = [
        EvalAcc(net, test_loader, BatchLim=100, Domain=x, Text='Evaluating test datapoints from the %s domain' % x) for
        x in ['Src', 'Tgt']]
    SrcTrainAccs, TgtTrainAccs = [
        EvalAcc(net, train_loader, BatchLim=100, Domain=x, Text='Evaluating train datapoints from the %s domain' % x)
        for x in ['Src', 'Tgt']]

    if hp.LogToWandb:
        1
    else:
        print('Results for experiment: %s' % hp.ExpName)
        print('     Source domain:')
        print('         Train Acc = %g' % (np.mean(SrcTrainAccs)))
        print('         Test Acc = %g' % (np.mean(TgtTrainAccs)))
        print('     Target domain:')
        print('         Train Acc=%g' % (np.mean(SrcTestAccs)))
        print('         Test Acc=%g' % (np.mean(TgtTestAccs)))

##
if __name__ == "__main__":
    args = parser.parse_args()
    hp=GetConfFromArgs(args)

    hp.Src='A'
    hp.Tgt='W'
    RunCofiguration(hp)


