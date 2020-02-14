from model import classifier

def get_model(args):
    if args.task == 'fundus':
        return classifier.MyModel(args)




