from .btcv import BTCV
from .amos import AMOS
from .msd import MSD
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def get_dataloader(args):
    # transform_train = transforms.Compose([
    #     transforms.Resize((args.image_size,args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_train_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.Resize((args.image_size, args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])
    
    if args.dataset == 'btcv':
        '''btcv data'''
        btcv_train_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        btcv_test_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(btcv_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'amos':
        '''amos data'''
        amos_train_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        amos_test_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(amos_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(amos_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif "Dataset" in args.dataset:
        '''MSD data'''
        msd_train_dataset = MSD(args, args.dataset, fold=args.fold, transform = None, transform_msk= None, mode = 'train', prompt=args.prompt)
        msd_test_dataset = MSD(args, args.dataset,  fold=args.fold, transform = None, transform_msk= None, mode = 'val', prompt=args.prompt)

        nice_train_loader = DataLoader(msd_train_dataset, batch_size=1, shuffle=True, num_workers=args.n_train_workers, pin_memory=True)
        nice_test_loader = DataLoader(msd_test_dataset, batch_size=1, shuffle=False, num_workers=args.n_test_workers, pin_memory=True)
        '''end'''
    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader