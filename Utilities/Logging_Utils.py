import wandb
import numpy as np

class MyLogger:
    def __init__(self):
        # loss
        self.train_losses_src= []
        self.train_losses_tgt= []
        self.val_losses_src= []
        self.val_losses_tgt= []

        # acc
        self.train_accs_src= []
        self.train_accs_tgt= []
        self.val_accs_src= []
        self.val_accs_tgt = []

    def update(self,hp,batch,loss_s,loss_t,val_loss_s,val_loss_t,train_acc_src,train_acc_tgt,val_acc_src,val_acc_tgt):
        self.train_losses_src.append(loss_s.item())
        self.train_losses_tgt.append(loss_t.item())
        if batch % hp.ValMonitoringFactor == 0:
            self.val_losses_src.append(val_loss_s.item())
            self.val_losses_tgt.append(val_loss_t.item())
        self.train_accs_src.append(train_acc_src)
        self.train_accs_tgt.append(train_acc_tgt)
        if batch % hp.ValMonitoringFactor == 0:
            self.val_accs_src.append(val_acc_src)
            self.val_accs_tgt.append(val_acc_tgt)

    def update_progress_bar(self,hp,batch,progress_bar):
        if len(self.val_accs_src) > 10:
            progress_bar.set_postfix({'train_loss_src': np.mean(np.log10(self.train_losses_src[-10:])),
                                      'train_loss_tgt': np.mean(np.log10(self.train_losses_tgt[-10:])),
                                      'train_acc_src': np.mean(self.train_accs_src[-10:]),
                                      'train_acc_tgt': np.mean(self.train_accs_tgt[-10:]),
                                      'val_acc_tgt': self.val_accs_tgt[-1] if batch > hp.ValMonitoringFactor else -1})

def log_to_wandb(hp,batch,loss_s,loss_t,val_loss_s,val_loss_t,train_acc_src,train_acc_tgt,val_acc_src,val_acc_tgt):
    wandb.log({
        'SrcTrainLoss': loss_s.item(),
        'TgtTrainLoss': loss_t.item(),
        'SrcTrainAcc': train_acc_src,
        'TgtTrainAcc': train_acc_tgt})
    if batch % hp.ValMonitoringFactor == 0:
        wandb.log({
            'TgtValLoss': val_loss_t,
            'TgtValAcc': val_acc_tgt})
    return 1

def log_results(hp,src_train_accs,tgt_train_accs,src_test_accs,tgt_test_accs):

    print('Results for experiment: %s' % hp.ExpName)
    print('     Source domain:')
    print('         Train Acc = %g' % (np.mean(src_train_accs)))
    print('         Test Acc = %g' % (np.mean(tgt_train_accs)))
    print('     Target domain:')
    print('         Train Acc=%g' % (np.mean(src_test_accs)))
    print('         Test Acc=%g' % (np.mean(tgt_test_accs)))

    if hp.LogToWandb:
        wandb.run.summary["Source Train Acc"] = np.mean(src_train_accs)
        wandb.run.summary["Target Train Acc"] = np.mean(tgt_train_accs)
        wandb.run.summary["Source Test Acc"] = np.mean(src_test_accs)
        wandb.run.summary["Target Test Acc"] = np.mean(tgt_test_accs)