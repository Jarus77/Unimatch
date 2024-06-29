import argparse
import os
import numpy as np




def get_args():

    parser=argparse.ArgumentParser(description='Stranger-Section-2')
    parser.add_argument('--project_name',type=str,default='ss2_idx_3')
    parser.add_argument('--model_name',type=str,default='debug')
    parser.add_argument('--use_checkpoint',action='store_true')


    data_root=r'C:\Users\suraj\OneDrive\sprasad2\unimatch_from_scratch'

    parser.add_argument('--data_root',type=str,default=data_root)
    parser.add_argument('--dataset',type=str,required= True)
    parser.add_argument('--nclass',type=str,required=True)


    parser.add_argument('--num_samples',type=int,required=False)
    parser.add_argument('--num_epochs',type=int,required=True)
    parser.add_argument('--save_path',type=str,required=True)


    args=parser.parse_args()

    labeled_data_dir=os.path.join(data_root,f'\labeled\{args.dataset}')
    unlabeled_data_dir=os.path.join(data_root,f'\unlabeled\{args.dataset}')
    val_data_dir=os.path.join(data_root,f'\labeled\{args.dataset}')


    parser.add_argument('--labeled_data_dir',type=str,default=labeled_data_dir)

    parser.add_argument('--unlabeled_data_dir',type=str,default=unlabeled_data_dir)
    parser.add_argument('--val_data_dir',type=str,default=val_data_dir)




    args=parser.parse_args()

    return args



def count_params(model):

    param_num=sum(p.numel() for p in model.parameters()) # numel Returns the total number of elements in the input tensor.
    return param_num/1e6


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


class AverageMeter(object):
    """Computes and stores the average, current value, and variance"""

    def __init__(self, length=0, track_variance=False):
        self.length = length
        self.track_variance = track_variance
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
            self.squared_sum = 0.0  # For variance calculation
        self.val = 0.0
        self.avg = 0.0
        self.var = 0.0  # Initialize variance

    def update(self, val, num=1):
        if self.length > 0:
            assert num == 1  # Avoid bad usage; refine if needed
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
            if self.track_variance:
                self.var = np.var(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count
            if self.track_variance:
                self.squared_sum += (val ** 2) * num
                mean_squared = self.avg ** 2
                mean_of_squares = self.squared_sum / self.count
                self.var = mean_of_squares - mean_squared