from model import classifier, segmentor

def get_model(args):
    if args.task == 'fundus' or args.task == 'mnist' or args.task == 'iqa':
        return classifier.MyModel(args)
    elif args.task == 'optic':
        return segmentor.MyModel(args)





