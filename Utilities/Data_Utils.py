from office31 import office31
from mnistusps import mnistusps
from Utilities.Digits_Utils import *
from Utilities.Office_Utils import *
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

##
def get_datasets(hp):
    if hp.Src in ['M', 'U']:
        train, val, test = mnistusps(
            source_name="mnist" if hp.Src == 'M' else "usps",
            target_name="mnist" if hp.Tgt == 'M' else "usps",
            seed=np.random.randint(100),
            num_source_per_class=200,
            num_target_per_class=hp.SamplesPerClass,
            same_to_diff_class_ratio=3,
            image_resize=(16, 16),
            group_in_out=True,  # groups data: ((img_s, img_t), (lbl_s, _lbl_t))
            framework_conversion="pytorch",
            data_path=os.path.join('Datasets', 'digits'),  # downloads to "~/data" per default
        )
        train_dataset = MyDigits(train, transform=None)
        test_dataset = MyDigits(test, transform=None)
        val_dataset = MyDigits(val, transform=None)
        n_classes = 10

    elif hp.Src in ['A', 'W', 'D']:
        mapper = {'A': "amazon", 'W': "webcam", 'D': "dslr"}
        train, val, test = office31(
            source_name=mapper[hp.Src],
            target_name=mapper[hp.Tgt],
            seed=np.random.randint(100),
            same_to_diff_class_ratio=3,
            image_resize=(240, 240),
            group_in_out=True,  # groups data: ((img_s, img_t), (lbl_s, _lbl_t))
            framework_conversion="pytorch",
            office_path=os.path.join('Datasets', 'office31'),
            # automatically downloads to "~/data"
        )
        train_dataset = MyOffice(train, transform=office_train_transform)
        test_dataset = MyOffice(test, transform=office_val_transform)
        val_dataset = MyOffice(val, transform=office_val_transform)
        n_classes = 31

    [train_loader, test_loader, val_loader] = [DataLoader(dataset, batch_size=hp.BatchSize,
                                                          shuffle=True, num_workers=0, drop_last=False)
                                               for dataset in [train_dataset, test_dataset, val_dataset]]
    return train_loader, test_loader, val_loader, n_classes

##
def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.
    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to
    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


##
class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)

class ForeverBatchesIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, batch_size, device=None):
        self.iter = ForeverDataIterator(data_loader, device)
        self.batch_size = batch_size

    def __next__(self):
        data = next(self.iter)
        while not (data[0].shape[0] == self.batch_size):
            ex_data = next(self.iter)
            data[0] = torch.cat((data[0], ex_data[0]), dim=0)
            data[1] = torch.cat((data[1], ex_data[1]), dim=0)
            if data[0].shape[0] > self.batch_size:
                data[0] = data[0][:self.batch_size]
                data[1] = data[1][:self.batch_size]
        while not (data[2].shape[0] == self.batch_size):
            ex_data = next(self.iter)
            data[2] = torch.cat((data[2], ex_data[2]), dim=0)
            data[3] = torch.cat((data[3], ex_data[3]), dim=0)
            if data[2].shape[0] > self.batch_size:
                data[2] = data[2][:self.batch_size]
                data[3] = data[3][:self.batch_size]
        # print(not (data[0].shape[0] == self.batch_size) or not(data[1].shape[0] == self.batch_size))
        return data

    def __len__(self):
        return len(self.iter.data_loader)

