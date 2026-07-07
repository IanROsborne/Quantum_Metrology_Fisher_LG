import numpy as np
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfSite, SpinHalfFermionSite, set_common_charges, GroupedSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg, tebd
from tenpy.networks.mpo import MPO



class KondoModel(CouplingMPOModel, NearestNeighborModel):
    r"""1D Kondo lattice model with conduction electrons and localized spins.
    
    H = -t \sum_{<ij>\sigma} c^\dagger_{i\sigma} c_{j\sigma} + h.c.
        + J_1 \sum_{i \sigma\sigma'} c^\dagger_{i\sigma} \vec{S}_{\sigma\sigma'} c_{i\sigma'} \cdot \vec{s}_i
        + J_2 \sum_{i} \vec{s}_i \cdot \vec{s}_{i+1}
    """

    def init_sites(self):
        ferm_site = SpinHalfFermionSite(cons_N='N', cons_Sz= None )
        spin_site = SpinHalfSite(conserve= None)
        set_common_charges([ferm_site, spin_site], new_charges='independent')
        
        site = GroupedSite([ferm_site, spin_site], labels = ['e', 'i'] , charges = 'independent')
        return site
   
    def init_lattice(self, model_params):
        L = model_params['L']
        bc = model_params.get("bc", "open")
        site = self.init_sites()
        lat = Chain(L, site, bc=bc, bc_MPS='finite')
        return lat
    
    def init_terms(self, model_params):
        t = model_params.get('t', 1.0)
        J1 = model_params.get('J1', 1.0)
        J2 = model_params.get('J2', 0.5)
        lat = self.init_lattice(model_params)
        
        # Electron hopping
        for u1, u2, dx in lat.pairs['nearest_neighbors']:
            self.add_coupling(-1.0 * t, u1, 'Cdue', u2, 'Cue', dx, plus_hc=True)
            self.add_coupling(-1.0 * t, u1, 'Cdde', u2, 'Cde', dx, plus_hc=True)
        # J1: Kondo coupling on each site
        # S_electron · s_impurity = S^z_e S^z_i + (S^+_e S^-_i + S^-_e S^+_i)/2
        for u in range(len(lat.unit_cell)):
            # Sz Sz coupling
            self.add_onsite_term(4.0 * J1, u, "Szi Sze")
            # S+ S- coupling
            self.add_onsite_term(2.0 * J1, u, "Spi Sme", plus_hc = False)  
            self.add_onsite_term(2.0 * J1, u, "Smi Spe", plus_hc = False)     
        
        # J2: Impurity-impurity Heisenberg coupling
        for u1, u2, dx in lat.pairs['nearest_neighbors']:
            self.add_coupling(4.0 * J2, u1, 'Szi', u2, 'Szi', dx)
            self.add_coupling(2.0 * J2, u1, 'Spi', u2, 'Smi', dx)
            self.add_coupling(2.0 * J2, u1, 'Smi', u2, 'Spi', dx)

    def build_Q_MPO(self, model_params, Q_op = 'Szi_tot'):
        """
        builds an MPO corresponding to Q
        """
        L = model_params['L']
        lat = self.init_lattice(model_params)

        Id = lat.unit_cell[0].get_op("Id")
        if Q_op == 'Szi_tot':
            Q = lat.unit_cell[0].get_op("Sigmazi")
        else:
            raise ValueError("Must pick a preconfigured operator")
        
        W = []

        for i in range(L):
            Wi = np.empty((2, 2), dtype=object)

            Wi[0, 0] = Id
            Wi[1, 1] = Id

            Wi[0, 1] = Q / L
            Wi[1, 0] = None

            W.append(Wi)
        return MPO.from_grids(lat.mps_sites(), W, IdL=0, IdR=1)