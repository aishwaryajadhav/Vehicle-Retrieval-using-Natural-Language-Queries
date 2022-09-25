import io
import logging
import os
import  torch
import colorlog
from collections import OrderedDict
import pdb


def load_new_model_from_checkpoint_ts1(model, cp_path):
    checkpoint = torch.load(cp_path)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        # print(name,": ",v.shape)
        if(name[:15] != "vis_backbone_bk" and name[:16] != "domian_vis_fc_bk" and name[:7] != "attpool"  and name[:19] != "domian_vis_fc_merge" and name[:13] != "vis_motion_fc" and name[:14] != "lang_motion_fc" and name[:6] != "id_cls"):
            new_state_dict[name] = v
        
    # print(new_state_dict.keys())
    model.load_state_dict(new_state_dict)
    return model

def load_new_model_from_checkpoint(model, cp_path, num_classes, embed_dim):
    checkpoint = torch.load(cp_path)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        # print(name,": ",v.shape)
        if(name[:15] != "vis_backbone_bk" and name[:16] != "domian_vis_fc_bk" and name[:7] != "attpool"  and name[:19] != "domian_vis_fc_merge" and name[:13] != "vis_motion_fc" and name[:14] != "lang_motion_fc"):
            if(name == "id_cls.3.weight" or name == "id_cls2.3.weight" or name == "id_cls3.3.weight"):
                new_state_dict[name] = torch.zeros((num_classes, embed_dim))
            elif(name == "id_cls.3.bias" or name == "id_cls2.3.bias" or name == "id_cls3.3.bias"):
                new_state_dict[name] = torch.zeros((num_classes))
            elif(name == "vis_car_fc.2.weight" or name == "lang_car_fc.2.weight"):
                new_state_dict[name] = torch.zeros((embed_dim, embed_dim))
            elif(name == "vis_car_fc.2.bias" or name == "lang_car_fc.2.bias"):
                new_state_dict[name] = torch.zeros((embed_dim))
            
            else:
                new_state_dict[name] = v
        
    # print(new_state_dict.keys())
    model.load_state_dict(new_state_dict)
    return model

def load_new_model_from_checkpoint_stage2(model, cp_path,  efficient_net=False):
    
    checkpoint = torch.load(cp_path)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        if(name[:12] != "vis_backbone" and name[:10] != "vis_car_fc" and name[:11] != "lang_car_fc" and name[:19] != "domian_vis_fc_merge" and name[:13] != "domian_vis_fc" and name[:15] != "vis_backbone_bk"):
            new_state_dict[name] = v
    
    if(efficient_net):
        orig_dict = model.state_dict() 
        # x = [s for s in new_state_dict.keys() if s not in orig_dict.keys()]
        orig_dict.update(new_state_dict)
        new_state_dict = orig_dict

    model.load_state_dict(new_state_dict)
    return model  


class TqdmToLogger(io.StringIO):
    logger = None
    level = None
    buf = ''

    def __init__(self):
        super(TqdmToLogger, self).__init__()
        self.logger = get_logger('tqdm')

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.info(self.buf)


def get_logger(logger_name='default', debug=False, save_to_dir=None):
    if debug:
        log_format = (
            '%(asctime)s - '
            '%(levelname)s : '
            '%(name)s - '
            '%(pathname)s[%(lineno)d]:'
            '%(funcName)s - '
            '%(message)s'
        )
    else:
        log_format = (
            '%(asctime)s - '
            '%(levelname)s : '
            '%(name)s - '
            '%(message)s'
        )
    bold_seq = '\033[1m'
    colorlog_format = f'{bold_seq} %(log_color)s {log_format}'
    colorlog.basicConfig(format=colorlog_format, datefmt='%y-%m-%d %H:%M:%S')
    logger = logging.getLogger(logger_name)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if save_to_dir is not None:
        fh = logging.FileHandler(os.path.join(save_to_dir, 'log', 'debug.log'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        fh = logging.FileHandler(
            os.path.join(save_to_dir, 'log', 'warning.log'))
        fh.setLevel(logging.WARNING)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        fh = logging.FileHandler(os.path.join(save_to_dir, 'log', 'error.log'))
        fh.setLevel(logging.ERROR)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
    
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # print(output.shape)
        # print(target.shape)
        # print(batch_size)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        # pdb.set_trace()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        del output
        del target
        del pred
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)