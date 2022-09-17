import torch
import time

class TorchTimer(object):
    class NamedTimer(object):
        def __init__(self, name, print_freq):
            self.name = name
            self.time_cost = 0
            self.begin_time = 0
            self.exe_counter = 0
            self.print_freq = print_freq
        
        def reset(self):
            self.time_cost = 0
            self.exe_counter = 0

        def __enter__(self):
            if self.print_freq > 0:
                torch.cuda.synchronize()
                self.begin_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.print_freq == -1:
                return
            torch.cuda.synchronize()
            self.time_cost += time.time() - self.begin_time
            self.exe_counter += 1
            if self.exe_counter % self.print_freq == 0:
                print('Average time cost of {}: {:.4}ms'. format(self.name, self.time_cost / self.exe_counter * 1000))
                self.reset()

    def __init__(self, print_freq = 10):
        self.timer_dict = {}
        self.print_freq = print_freq
    
    def timing(self, name, freq = None):
        print_freq = freq if freq is not None else self.print_freq
        if name not in self.timer_dict:
            self.timer_dict[name] = TorchTimer.NamedTimer(name, print_freq)
        return self.timer_dict[name]


if __name__ == '__main__':
    a = torch.rand((1000, 1000), device='cuda')
    b = torch.rand((1000, 1000), device='cuda')

    timer = TorchTimer(10)
    for i in range(100):
        with timer.timing('first'):
            c = a @ b
        
        with timer.timing('second'):
            d = a * b