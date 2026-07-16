import sys

class ProgressBar:
    def __init__(self, min_value=0, max_value=1, length=40, prefix='', suffix=''):
        self.min_value = min_value
        self.max_value = max_value
        self.length = length
        self.prefix = prefix
        self.suffix = suffix
        self.finished = False

    def update(self, position):
        if self.finished:
            return
        span = self.max_value - self.min_value
        frac = 0.0 if span == 0 else (position - self.min_value) / span
        frac = min(max(frac, 0.0), 1.0)
        filled = int(self.length * frac)
        if filled >= self.length:
            bar = '[' + '=' * self.length + ']'
        else:
            bar = '[' + '=' * filled + '>' + '.' * (self.length - filled - 1) + ']'
        percent = frac * 100
        sys.stdout.write(f'\r{self.prefix} {bar} {percent:6.2f}% {self.suffix}')
        sys.stdout.flush()
        if frac >= 1.0:
            sys.stdout.write('\n')
            self.finished = True


def scalar_multiply_list(list, x):
    '''returns a list where every element is multiplied by x.'''

    return [x * i for i in list]