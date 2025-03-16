from .tn3k import tn3kDataSet

def build_dataset(args):
    """
    Function to build and return dataset objects based on the provided arguments.    
    
    Returns:
    If 'manner' is 'test': Returns a test dataset object.
    Otherwise: Returns a tuple of dataset objects ('train_data', 'train_u_data', 'valid_data').
    """

    if args.manner == 'test':
        if args.dataset == 'tn3k':
            test_data = tn3kDataSet(args.root, args.expID, mode='test')
        return test_data
    else:
        if args.dataset == 'tn3k':
            train_data = tn3kDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = tn3kDataSet(args.root, args.expID, mode='valid')
            test_data = tn3kDataSet(args.root, args.expID, mode='test')
            train_u_data = None
            if args.manner == 'semi' or args.manner == 'self':
                train_u_data = tn3kDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        return train_data, train_u_data, valid_data


