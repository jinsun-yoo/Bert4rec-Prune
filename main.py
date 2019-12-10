from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from pruners import pruner_factory
from utils import *
from utils import scatterplot

def train():
    from models import model_factory
    from dataloaders import dataloader_factory
    from trainers import trainer_factory
    from pruners import pruner_factory
    from utils import *
    from utils import scatterplot

    from torch.utils.tensorboard import SummaryWriter
    from torchvision import datasets, transforms



    export_root = setup_train(args)
    test_result_root = 'experiments/testresults'
    test_result_title = export_root[12:]
    test_result_title += '.txt'
    model = model_factory(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    pruner = pruner_factory(args, model)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root, pruner)
    #load_pretrained_weights(model, './experiments/ml-1m.pth')
    trainer.train()
    trainer.test()

    if args.prune:
        trainer.prune()
    #pruner.print_mask(model)
    #pruner.print_percentage(model)
    i = 0
    test_result = trainer.test()

    save_test_result(export_root, test_result)
    save_test_result(test_result_root, test_result, test_result_title)
    print(test_result_root)


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
