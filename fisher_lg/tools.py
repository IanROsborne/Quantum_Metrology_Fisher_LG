import sys
import pickle

import numpy as np
from tenpy.networks.mps import MPS

from fisher_lg.State import State_Evo

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
 
 
def save_state(psi, filename):
    """
    Save everything needed to reconstruct `psi` later.
 
    Parameters
    ----------
    psi : tenpy.networks.mps.MPS
        The ground-state MPS from your DMRG run (what you called psi_temp).
    model_params : dict
        Parameters used to build the model.
    extra_params : dict
        Any additional parameters you want to keep alongside the state.
    energy : float
        Ground-state energy (or whatever scalar you were attaching to
        psi.energy).
    filename : str
        Path to write the pickle file to, e.g. "ground_state.pkl".
    """
    # Force _B / _S into real lists of independent tensor copies. 
    Bs = [B.copy() for B in psi._B]
    Ss = [S.copy() if S is not None else None for S in psi._S]
 
    data = {
        "sites": psi.sites,
        "Bs": Bs,
        "Ss": Ss,
        "bc": psi.bc,
        "form": psi.form,
        "norm": psi.norm,
        "unit_cell_width": getattr(psi, "unit_cell_width", None),
        "model_params": psi.model_params,
        "extra_params": psi.extra_params,
        "energy": psi.energy,
    }
 
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
 
    print(f"Saved DMRG state to {filename}")
 
 
def load_state(filename, **kwargs):
    """
    Load a previously-saved DMRG ground state.
 
    Returns
    -------
    psi : tenpy.networks.mps.MPS
        Fully reconstructed MPS, ready to use directly for expectation
        values, correlation functions, etc. `psi.model_params`,
        `psi.extra_params`, and `psi.energy` are attached as attributes,
        same as in your original code.
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
 
    psi = MPS(
        data["sites"],
        data["Bs"],
        data["Ss"],
        bc=data["bc"],
        form=data["form"],
        norm=data["norm"],
    )
 
    if data.get("unit_cell_width") is not None:
        psi.unit_cell_width = data["unit_cell_width"]
 
    model_params = data["model_params"]
    for key in kwargs:
        model_params[key] = kwargs[key]
    extra_params = data["extra_params"]
    energy = data["energy"]

    psi = State_Evo.from_MPS(psi, model_params = model_params, extra_params = extra_params)
    psi.set_energy(energy)
    
    return psi
 
 
 