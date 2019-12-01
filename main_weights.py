from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *


def train():
    export_root = setup_train(args)
    model = model_factory(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    #trainer.train()
    print("Model's state_dict:")
    for param_tensor in model.bert.state_dict():
      print(param_tensor, "\t", model.bert.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in trainer.optimizer.state_dict():
        print(var_name, "\t", trainer.optimizer.state_dict()[var_name])
    torch.save(model, './initmodel.pth')
    #test_result = test_with(trainer.best_model, test_loader)
    #save_test_result(export_root, test_result)


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')