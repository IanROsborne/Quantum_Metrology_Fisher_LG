import numpy as np

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import dmrg, tebd
from tenpy.networks.mpo import MPO


class State(MPS):

    def __init__(self,
                 sites,
                 Bs,
                 SVs,
                 bc='finite',
                 form='B',
                 norm=1.0,
                 unit_cell_width=None,
                 understood_shift_symmetry=False):

        super().__init__(
            sites,
            Bs,
            SVs,
            bc,
            form,
            norm,
            unit_cell_width,
            understood_shift_symmetry=understood_shift_symmetry,
        )

        self.model = None
        self.energy = None
        self.params = None

    @classmethod
    def from_dmrg(cls, model_params, dmrg_params=None):

        if model_params is None:
            raise ValueError("Provide model_params")
        model_params = dict(model_params)
        model_type = model_params.get("model_type", "Ising")
        if model_type == "Ising":
            model = TFIChain(model_params)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        if dmrg_params is None:
            dmrg_params = {
                "mixer": None,
                "max_E_err": 1.e-10,
                "trunc_params": {"chi_max": 100, "svd_min": 1.e-10},
                "verbose": False,
            }

        #run dmrg
        psi = MPS.from_lat_product_state(model.lat, [['up']])
        eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
        energy, psi = eng.run()

        obj = cls(
            psi.sites,
            [B.copy() for B in psi._B],
            [S.copy() if S is not None else None for S in psi._S],
            psi.bc,
            psi.form,
            psi.norm,
            psi.unit_cell_width,
        )

        obj.energy = energy
        obj.model = model
        obj.params = dict(model_params)

        return obj
    
    def QFI(self, op, op_sum = True):
        """
        Calculate the Quantum Fisher Information (QFI) for a given state and operator.
        
        Parameters:
        psi : MPS
            The matrix product state representing the quantum state.
        op : str
            The operator for which to calculate the QFI ('Sigmax' or 'Sigmaz').
            
        Returns:
        float
            The calculated QFI value.
        """
        L_half = self.L//2
        if op_sum: 
            # Calculate the expectation value of the operator
            expe = np.mean(self.expectation_value(op))

            # Calculate the correlation function
            corr_q2 = self.correlation_function(op, op, sites1=None, sites2=None)
            # Calculate the expectation value of the squared operator
            expectation_q2 = np.mean(corr_q2)
        else:
            expe = self.expectation_value(op)[L_half]
            expectation_q2 = self.correlation_function(op, op, sites1=[L_half], sites2=[L_half])
        
        # Calculate QFI using the formula: QFI = 4 * (expectation_q2 - expe^2)
        qfi = 4 * (expectation_q2 - expe**2).flatten()[0]
        
        return qfi