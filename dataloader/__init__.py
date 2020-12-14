from .ade20k import ADE20K
from .Road_Extraction import RoadSegmentation
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    if args.dataset == 'road':
        train_set = RoadSegmentation(args, split='train', base_dir='dataset/road')
        val_set = RoadSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'ade20k':
        train_set = ADE20K(args, split='train', base_dir='dataset/ade20k')
        val_set = ADE20K(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'deepglobe':
        train_set = RoadSegmentation(args, split='train', base_dir='dataset/deepglobe')
        val_set = RoadSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError
