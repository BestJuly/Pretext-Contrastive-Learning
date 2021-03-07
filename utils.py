import csv
import os
import torch
import pdb

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data.item()

    return n_correct_elems / batch_size

def calculate_top1_top5(outputs, targets):
    top1 = 0
    top5 = 0
    out = torch.mean(torch.nn.Softmax(dim=1)(outputs),0)
    result_top5 = torch.topk(out, 5)[1]
    #print(targets[0], result_top5)
    if targets[0] in result_top5:
      top5 = 1
    if targets[0] == result_top5[0]:
      top1 = 1
    return top1, top5, out

def get_latest_checkpoint(filepath):
    models = []
    for parent, dirnames, filenames in os.walk(filepath):
      for filename in filenames:
        if filename[-3:] == 'pth':
          models.append(filename)
    models = sorted(models, key=lambda x: int(x[5:-4]))
    if len(models) < 1:
      return 'empty', -1
    else:
      model_name = models[-1]
      epoch_num = int(model_name[5:-4])
      return model_name, epoch_num
