from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from pruners import pruner_factory
from utils import *


def train():
    export_root = setup_train(args)
    print('model')
    model = model_factory(args)
    print('loader')
    train_loader, val_loader, test_loader = dataloader_factory(args)
    print('pruner')
    pruner = pruner_factory(args, model)
    print('trainer')
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root, pruner)
    print('train')
    load_pretrained_weights(model, './experiments/test_2019-12-02_0/models/best_acc_model.pth')
    #trainer.train()
    #trainer.test()
    trainer.prune()

    i = 0
    for name, p in model.bert.named_parameters():
        print(f'[{i}]')
        i += 1
        print(name)
        print(p.requires_grad)
        print(p.size())
        print(len(p.data.size()))
    #test_result = test_with(trainer.best_model, test_loader)
    #save_test_result(export_root, test_result)


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')