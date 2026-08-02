import sys

import numpy as np

class ProgressBar:
    r'''Implements an updatable progress bar'''
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

def combine_npy_files(file1_path, file2_path, output_path):
    """
    Combines two .npy files into a single .npy file.
    
    Parameters:
    -----------
    file1_path : str
        Path to the first .npy file
    file2_path : str
        Path to the second .npy file
    output_path : str
        Path where the combined .npy file will be saved
    """
    data1 = np.load(file1_path, allow_pickle=True)
    data2 = np.load(file2_path, allow_pickle=True)
    
    # Combine the arrays
    combined_data = [data1]
    for i in data2:
        combined_data.append(i)

    entries = [np.asarray(r).tolist() for r in combined_data]
    final = np.empty(len(entries), dtype=object)
    for idx, entry in enumerate(entries):
        final[idx] = entry
    
    # Save the combined data
    np.save(output_path, final, allow_pickle= True)
    print(f"Combined data saved to {output_path}")
    print(f"Total entries: {len(final)}")