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
    net.to(device)

    if hp.Optimizer=='Adadelta':
        optimizer = torch.optim.Adadelta(net.parameters(),lr=hp.LearningRate, weight_decay=hp.WD)

    if hp.Optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=hp.LearningRate, momentum=0.9, weight_decay=hp.WD,
                                    nesterov=True)
    elif hp.Optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=hp.LearningRate, weight_decay=hp.WD)

    ## Logging
    if hp.LogToWandb:
        wandb.login()
        wandb.init(project=hp.ProjectName, entity=hp.WadbUsername)
        wandb.run.name = hp.ExpName
        config = wandb.config
        config.args = vars(hp)
    if hp.LogToWandb:
        wandb.login()
        wandb.init(project=hp.ProjectName, entity=hp.WadbUsername)
        wandb.run.name = hp.ExpName
        config = wandb.config
        config.args = vars(hp)

    ## Train model
    train_losses_src, train_losses_tgt, val_losses_src, val_losses_tgt = [], [], [], []
    train_accs_src, train_accs_tgt, val_accs_src, val_accs_tgt = [], [], [], []

    if n_classes == -1:
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
    else:
        criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(hp.NumberOfBatches),
                desc="Training model",
                bar_format="{desc:20}{percentage:2.0f}{r_bar}")
    for batch in pbar:
        net.train()
        src_img, src_label, tgt_img, tgt_label = next(train_iter)
        if not(len(src_label)==len(tgt_label)):
            continue
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

        # ----- Train Monitoring -----
        if hp.MonitorTraining:
            _, idx = src_pred.max(dim=1)
            train_acc_src = (idx == src_label).sum().cpu().item() / len(idx)
            _, idx = tgt_pred.max(dim=1)
            train_acc_tgt = (idx == tgt_label).sum().cpu().item() / len(idx)
        # ----- Val Monitoring -----
        if batch % hp.ValMonitoringFactor == 0:
            with torch.no_grad():
                net.eval()
                src_img, src_label, tgt_img, tgt_label = next(val_iter)
                if not (len(src_label) == len(tgt_label)):
                    continue
                src_img, tgt_img = (x.to(device, dtype=torch.float) for x in [src_img, tgt_img])
                src_label, tgt_label = (x.to(device, dtype=torch.long) for x in [src_label, tgt_label])
                src_pred = net(src_img)
                tgt_pred = net(tgt_img)
                val_loss_s = criterion(src_pred, src_label)
                val_loss_t = criterion(tgt_pred, tgt_label)

                _, idx = src_pred.max(dim=1)
                val_acc_src = (idx == src_label).sum().cpu().item() / len(idx)

                _, idx = tgt_pred.max(dim=1)
                val_acc_tgt = (idx == tgt_label).sum().cpu().item() / len(idx)
        # ----- Logging -----
        exec(open(os.path.join("Scripts", "LoggingScript.py")).read())

    ## Evaluate on test
    SrcTestAccs, TgtTestAccs = [
        EvalAcc(net, test_loader, BatchLim=100, Domain=x, Text='Evaluating test datapoints from the %s domain' % x) for
        x in ['Src', 'Tgt']]
    SrcTrainAccs, TgtTrainAccs = [
        EvalAcc(net, train_loader, BatchLim=100, Domain=x, Text='Evaluating train datapoints from the %s domain' % x)
        for x in ['Src', 'Tgt']]

    print('Results for experiment: %s' % hp.ExpName)
    print('     Source domain:')
    print('         Train Acc = %g' % (np.mean(SrcTrainAccs)))
    print('         Test Acc = %g' % (np.mean(TgtTrainAccs)))
    print('     Target domain:')
    print('         Train Acc=%g' % (np.mean(SrcTestAccs)))
    print('         Test Acc=%g' % (np.mean(TgtTestAccs)))

    if hp.LogToWandb:
        wandb.run.summary["Source Train Acc"] = np.mean(SrcTrainAccs)
        wandb.run.summary["Target Train Acc"] = np.mean(TgtTrainAccs)
        wandb.run.summary["Source Test Acc"] = np.mean(SrcTestAccs)
        wandb.run.summary["Target Test Acc"] = np.mean(TgtTestAccs)


##
if __name__ == "__main__":
    args = parser.parse_args()
    hp=GetConfFromArgs(args)
    hp.WadbUsername='orkatz'
    hp.ProjectName = 'SDA_Experiments_Project_2'
    hp.Method='CCSA'
    RunCofiguration(hp)


