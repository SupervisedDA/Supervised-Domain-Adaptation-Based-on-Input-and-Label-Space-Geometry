# ----- Logger -----
train_losses_src.append(loss_s.item())
train_losses_tgt.append(loss_t.item())
if batch % hp.ValMonitoringFactor == 0:
    val_losses_src.append(val_loss_s.item())
    val_losses_tgt.append(val_loss_t.item())
train_accs_src.append(train_acc_src)
train_accs_tgt.append(train_acc_tgt)
if batch % hp.ValMonitoringFactor == 0:
    val_accs_src.append(val_acc_src)
    val_accs_tgt.append(val_acc_tgt)

# ----- Console logger -----
if len(val_accs_src) > 10:
    pbar.set_postfix({'train_loss_src': np.mean(np.log10(train_losses_src[-10:])),
                      'train_loss_tgt': np.mean(np.log10(train_losses_tgt[-10:])),
                      'train_acc_src': np.mean(train_accs_src[-10:]),
                      'train_acc_tgt': np.mean(train_accs_tgt[-10:]),
                      'val_acc_tgt': val_acc_tgt if batch > hp.ValMonitoringFactor else -1
                      })

# ----- Wandb Logger -----
if hp.LogToWandb:
    wandb.log({
        'SrcTrainLoss': loss_s.item(),
        'TgtTrainLoss': loss_t.item(),
        'SrcTrainAcc': train_acc_src,
        'TgtTrainAcc': train_acc_tgt})
    if batch % hp.ValMonitoringFactor == 0:
        wandb.log({
            'TgtValLoss': val_loss_t,
            'TgtValAcc': val_acc_tgt})