from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
# from opts import opts
from opts2 import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from torchsummary import summary


# def main(opt, qtepoch=[0,]):
class main(object):
    def __init__(self, opt):
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
        Dataset = get_dataset(opt.dataset, opt.task)
        opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
        self.opt = opt
        print(opt)

        self.logger = Logger(opt)

        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
        
        print('Creating model...')
        model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = model

        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        self.optimizer = optimizer

        start_epoch = 0
        if opt.load_model != '':
            model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

        Trainer = train_factory[opt.task]
        trainer = Trainer(opt, model, optimizer)
        trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
        self.trainer = trainer

        print('Setting up data...')
        val_loader = torch.utils.data.DataLoader(
            Dataset(opt, 'val'), 
            batch_size=1, 
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        self.val_loader = val_loader

        if opt.test:
            _, preds = trainer.val(0, val_loader)
            val_loader.dataset.run_eval(preds, opt.save_dir)
            return

        train_loader = torch.utils.data.DataLoader(
            Dataset(opt, 'train'), 
            batch_size=opt.batch_size, 
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.train_loader = train_loader

        self.best = 1e10
    
    def train(self, epoch):
        mark = epoch if self.opt.save_all else 'last'
        log_dict_train, _ = self.trainer.train(epoch, self.train_loader)
        self.logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            self.logger.scalar_summary('train_{}'.format(k), v, epoch)
            self.logger.write('{} {:8f} | '.format(k, v))
        if self.opt.val_intervals > 0 and epoch % self.opt.val_intervals == 0:
            save_model(os.path.join(self.opt.save_dir, 'model_{}.pth'.format(mark)), 
                    epoch, self.model, self.optimizer)
        with torch.no_grad():
            log_dict_val, preds = self.trainer.val(epoch, self.val_loader)
        for k, v in log_dict_val.items():
            self.logger.scalar_summary('val_{}'.format(k), v, epoch)
            self.logger.write('{} {:8f} | '.format(k, v))
        if log_dict_val[self.opt.metric] < self.best:
            self.best = log_dict_val[self.opt.metric]
            save_model(os.path.join(self.opt.save_dir, 'model_best.pth'), 
                    epoch, self.model)
        else:
            save_model(os.path.join(self.opt.save_dir, 'model_last.pth'), 
                    epoch, self.model, self.optimizer)
        self.logger.write('\n')
        if epoch in self.opt.lr_step:
            lr = self.opt.lr * (0.1 ** (self.opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)