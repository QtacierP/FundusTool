import torch.utils.tensorboard as tsb
from tqdm import tqdm
from time import time
import sys
import numpy as np
import os
from torchvision.utils import make_grid
import torch
from matplotlib.pyplot import imshow, show
from torch.utils.tensorboard.summary import convert_to_HWC, make_image
from torch.utils.tensorboard.writer import make_np


class MyLogger():
    # A Keras style Logger
    def __init__(self, args, max_epoch, batch_size,
                 losses_name, step, model, metric='val_kappa', current_epoch=1, optimizer=None, warmup_scheduler=None,
                 lr_scheduler=None, weighted_sampler=None):
        self.callbacks = []
        self.max_epoch = max_epoch
        self.epoch = current_epoch
        self.batch_size = batch_size
        self.step = step
        self.losses = {}
        self.args = args
        self.model_path = args.model_path
        self.model = model
        self.ckpt_path = args.checkpoint
        self.writer = tsb.SummaryWriter(self.ckpt_path)
        self.warmup_scheduler = warmup_scheduler
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.weighted_sampler = weighted_sampler
        self.metric = metric
        self.history = 0 # Record for acc
        # Compile metric opt
        if 'loss' in metric:
            self.opt = np.less
            self.best = np.inf
        else:
            self.opt = np.greater
            self.best = -np.inf

        for loss_name in losses_name:
            self.losses[loss_name] = []

    def on_train_begin(self):
        tensor = torch.randn(1, self.args.n_colors, self.args.size, self.args.size).cuda()
        if self.args.n_gpus > 1:
            module = self.model.module
        else:
            module = self.model
        self.writer.add_graph(module, tensor)

    def on_train_end(self):
        print('saving last model....')
        if self.args.n_gpus > 1:
            save_module = self.model.module.state_dict()
        else:
            save_module = self.model.state_dict()
        torch.save(save_module, os.path.join(self.model_path,
                                             '{}_last.pt'.format(self.args.model)))


    def on_epoch_begin(self):
        self.history = 0
        print('=> Epoch {} <='.format(str(self.epoch)))
        self.bar = TorchBar(target=self.step, width=30)
        self.batch_num = 1
        if self.weighted_sampler is not None:
            self.weighted_sampler.step()

            # learning rate update
        if self.warmup_scheduler is not None and not self.warmup_scheduler.is_finish():
            if self.epoch >= 1:
                curr_lr = self.optimizer.param_groups[0]['lr']
                print('Learning rate warmup to {}'.format(curr_lr))
        elif self.lr_scheduler is not None:
            if self.epoch % 10 == 0:
                curr_lr = self.optimizer.param_groups[0]['lr']
                print('Current learning rate is {}'.format(curr_lr))

    def on_batch_begin(self):

        if self.warmup_scheduler and not self.warmup_scheduler.is_finish():
            self.warmup_scheduler.step()
        elif self.lr_scheduler:
            self.lr_scheduler.step()

    def on_batch_end(self, losses: dict):
        values = []
        for loss_name in losses.keys():
            loss = losses[loss_name]
            if not isinstance(loss, list):
                self.losses[loss_name].append(loss)
                values.append((loss_name, loss))
            else:
                self.losses[loss_name].append(loss[0])
                self.history += loss[1]
                smooth_loss = (self.history) / (self.batch_size * self.batch_num)

                values.append((loss_name, smooth_loss))
        self.bar.update(self.batch_num, values=values)
        self.batch_num += 1

    def on_epoch_end(self, val_metric: dict):
        del(self.bar)
        for loss_name in self.losses:
            loss = np.mean(self.losses[loss_name])
            self.writer.add_scalar(loss_name, loss,
                                   self.epoch)
            self.losses[loss_name] = []
        print('Validation result =>')
        for metric_name in val_metric.keys():
            self.writer.add_scalar(metric_name , val_metric[metric_name],
                                   self.epoch)
            print('{}: {}'.format(metric_name , val_metric[metric_name]))
            if metric_name == self.metric:
                if self.opt(val_metric[metric_name], self.best):
                    print('{} improved from {} to {}'.format(metric_name,
                                                             self.best, val_metric[metric_name]))
                    print('saving model....')
                    self.best = val_metric[metric_name]
                    if self.args.n_gpus > 1:
                        save_module = self.model.module.state_dict()
                    else:
                        save_module = self.model.state_dict()
                    torch.save(save_module, os.path.join(self.model_path,
                                                   '{}_best.pt'.format(self.args.model)))
        self.epoch += 1


class TorchBar():
    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None, unit_name='step'):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty')
                                  and sys.stdout.isatty())
                                 or 'ipykernel' in sys.modules
                                 or 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        Arguments:
                current: Index of current step.
                values: List of tuples:
                        `(name, value_for_last_step)`.
                        If `name` is in `stateful_metrics`,
                        `value_for_last_step` will be displayed as-is.
                        Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval
                    and self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is not None and current >= self.target:
                numdigits = int(np.log10(self.target)) + 1
                count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

class Pbar(object):
    """ Progress bar with title and timer
    Arguments:
    name: the bars name.
    target: Total number of steps expected.
    width: Progress bar width on screen.
    Usage example
    ```
    import kpbar
    import time
    pbar = kpbar.Pbar('loading and processing dataset', 10)
    for i in range(10):
        time.sleep(0.1)
        pbar.update(i)
    ```
    ```output
    loading and processing dataset
    10/10  [==============================] - 1.0s
    ```
    """

    def __init__(self, name, target, width=30):
        self.name = name
        self.target = target
        self.start = time()
        self.numdigits = int(np.log10(self.target)) + 1
        self.width = width
        print(self.name)

    def update(self, step):

        bar = ('%' + str(self.numdigits) + 'd/%d ') % (step + 1, self.target)

        status = ""

        if step < 0:
            step = 0
            status = "negtive?...\r\n"

        stop = time()

        status = '- {:.1f}s'.format((stop - self.start))

        progress = float(step + 1) / self.target

        # prog
        prog_width = int(self.width * progress)
        prog = ''
        if prog_width > 0:
            prog += ('=' * (prog_width - 1))
            if step + 1 < self.target:
                prog += '>'
            else:
                prog += '='
        prog += ('.' * (self.width - prog_width))

        # text = "\r{0} {1} [{2}] {3:.0f}% {4}".format(self.name, bar, prog, pregress, status)

        text = "\r{0} [{1}] {2}".format(bar, prog, status)
        sys.stdout.write(text)
        if step + 1 == self.target:
            sys.stdout.write('\n')
        sys.stdout.flush()


class WarmupLRScheduler:
    def __init__(self, optimizer, warmup_batch, initial_lr):
        self.step_num = 1
        self.optimizer = optimizer
        self.warmup_batch = warmup_batch
        self.initial_lr = initial_lr

    def step(self):
        if self.step_num <= self.warmup_batch:
            self.step_num += 1
            curr_lr = (self.step_num / self.warmup_batch) * self.initial_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = curr_lr

    def is_finish(self):
        return self.step_num > self.warmup_batch
