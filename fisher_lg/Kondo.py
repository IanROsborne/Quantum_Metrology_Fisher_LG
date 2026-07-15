import numpy as np

from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfSite, SpinHalfFermionSite, set_common_charges, GroupedSite
from tenpy.algorithms import dmrg



__all__ = ['KondoModel', 'KondoChain']


class KondoModel(CouplingMPOModel):
    r"""1D Kondo lattice model with conduction electrons and localized spins.
    
    H = -t \sum_{<ij>\sigma} c^\dagger_{i\sigma} c_{j\sigma} + h.c.
         J_1 \sum_{i \sigma\sigma'} c^\dagger_{i\sigma} \vec{S}_{\sigma\sigma'} c_{i\sigma'} \cdot \vec{s}_i
         J_2 \sum_{i} \vec{s}_i \cdot \vec{s}_{i+1}
    """
    couplings = {'t' : 'hopping', 'J_ii' : 'impurity-impurity Heisenberg coupling', 'J_ei' : 'conduction-impurity Heisenberg coupling'}

    def __init__(self, model_params):
        super().__init__(model_params)
        self.model_params = model_params

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', None)
        
        ferm_site = SpinHalfFermionSite(cons_N='N', cons_Sz= conserve )
        spin_site = SpinHalfSite(conserve= conserve)
        set_common_charges([ferm_site, spin_site], new_charges='same')
        
        site = GroupedSite([ferm_site, spin_site], labels = ['e', 'i'] , charges = 'same')
        return site

    
    def init_terms(self, model_params):
        t = model_params.get('t', 1.0)
        J1 = model_params.get('J_ei', 1.0)
        J2 = model_params.get('J_ii', 0.5)
        h = model_params.get('symmetry_breaking_field', 0)

        # Electron hopping
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-1.0 * t, u1, 'Cdue', u2, 'Cue', dx, plus_hc=True)
            self.add_coupling(-1.0 * t, u1, 'Cdde', u2, 'Cde', dx, plus_hc=True)
        # J1: Kondo coupling on each site
        # S_electron · s_impurity = S^z_e S^z_i + (S^+_e S^-_i + S^-_e S^+_i)/2
        for u in range(len(self.lat.unit_cell)):
            # Sz Sz coupling
            self.add_onsite(4.0 * J1, u, "Szi Sze")
            # S+ S- coupling
            self.add_onsite(2.0 * J1, u, "Spi Sme", plus_hc = False)  
            self.add_onsite(2.0 * J1, u, "Smi Spe", plus_hc = False)     
        
        # J2: Impurity-impurity Heisenberg coupling
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(4.0 * J2, u1, 'Szi', u2, 'Szi', dx)
            self.add_coupling(2.0 * J2, u1, 'Spi', u2, 'Smi', dx)
            self.add_coupling(2.0 * J2, u1, 'Smi', u2, 'Spi', dx)

        # h: symmetry breaking Zeeman field
        if h != 0:
            for u in range(len(self.lat.unit_cell)):
                # Sz Sz coupling
                self.add_onsite(-1.0 * h, u, "Szi")
                self.add_onsite(-1.0 * h, u, "Sze")


class KondoChain(KondoModel, NearestNeighborModel):
    """The :class:`KondoModel` on a Chain, suitable for TEBD.

    See the :class:`KondoModel` for the documentation of parameters.
    """

    default_lattice = Chain
    #force_default_lattice = True

    def init_lattice(self, model_params):
        """Initialize a 1D lattice"""
        L = model_params['L']
        conserve = model_params.get('conserve', 'best')
        bc_MPS = model_params.get('bc', 'finite')
        bc = 'periodic' if bc_MPS in ['infinite', 'segment'] else 'open'

        site = self.init_sites(model_params= model_params)
        lattice = Chain(L, site, bc=bc, bc_MPS=bc_MPS)
        
        return lattice

class AndersonImpurityModel(CouplingMPOModel):
    """1D single-site Anderson impurity model """

    couplings = {'t' : 'hopping', 'mu' : 'impurity potential', 'U' : 'impurity Hubbard interaction'}

    def __init__(self, model_params):
        super().__init__(model_params)
        self.model_params = model_params

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N')
        return SpinHalfFermionSite(cons_N=conserve)

    def init_lattice(self, model_params):
        L = model_params['L']
        site = self.init_sites(model_params)
        return Chain(L, site, bc='open')

    def init_terms(self, model_params):

        t = model_params['t']
        U = model_params['U']
        mu = model_params['mu']

        L = self.lat.N_sites
        imp = L // 2
        print(self.lat.unit_cell[0].opnames)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-1.0 * t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-1.0 * t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)

        # impurity onsite energy
        self.add_onsite_term(mu, imp, 'Ntot')

        # impurity Hubbard interaction
        self.add_onsite_term(U, imp, 'NuNd')

