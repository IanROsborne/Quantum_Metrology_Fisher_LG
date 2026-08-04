import numpy as np

from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfSite, SpinHalfFermionSite, set_common_charges, GroupedSite
from tenpy.networks.mps import MPS


__all__ = ['KondoModel', 'KondoChain', 'AndersonImpurityModel_Lanczos', 'AndersonImpurityModel', 'TFIsingChain']


class KondoModel(CouplingMPOModel):
    r"""1D Kondo lattice model with conduction electrons and localized spins.

    Parameters:

    model_params : dict 

    Hamiltonian:    
    H = -t \sum_{<ij>\sigma} c^\dagger_{i\sigma} c_{j\sigma} + h.c.
         J_1 \sum_{i \sigma\sigma'} c^\dagger_{i\sigma} \vec{S}_{\sigma\sigma'} c_{i\sigma'} \cdot \vec{s}_i
         J_2 \sum_{i} \vec{s}_i \cdot \vec{s}_{i+1}
    """
    couplings = {'t' : 'hopping',
                'J_ii' : 'impurity-impurity Heisenberg coupling', 
                'J_ei' : 'conduction-impurity Heisenberg coupling', 
                'h' : 'symmetry-breaking Zeeman field'}

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
        h = model_params.get('h', 0)

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
                self.add_onsite(-1.0 * h, u, "Szi")
                self.add_onsite(-1.0 * h, u, "Sze")


class KondoChain(KondoModel, NearestNeighborModel):
    r"""The :class:`KondoModel` on a Chain, suitable for TEBD.

    See the :class:`KondoModel` for the documentation of parameters.
    """

    default_lattice = Chain
    #force_default_lattice = True

    def init_lattice(self, model_params):
        """Initialize a 1D lattice"""
        L = model_params['L']
        bc_MPS = model_params.get('bc', 'finite')
        bc = 'periodic' if bc_MPS in ['infinite', 'segment'] else 'open'

        site = self.init_sites(model_params= model_params)
        lat = Chain(L, site, bc=bc, bc_MPS=bc_MPS)
        
        return lat

    def _set_initial_state(self, target_Sz = None):
            """
            Construct a half-filled product state with total Sz = target_Sz.
            """
            chain = self.lat.mps_sites()
            L = len(chain)
            if target_Sz is None:
                target_Sz = L
    
            # reference state has total Sz = 0
            state = ["up_e down_i"] * L
            if target_Sz > 0:
                if target_Sz > L:
                    raise ValueError("Requested Sz = " + str(target_Sz) +" sector impossible.")
    
                for i in range(target_Sz):
                    state[i] = "up_e up_i"
    
            elif target_Sz < 0:
                if -target_Sz > L:
                    raise ValueError("Requested Sz sector impossible.")
    
                for i in range(-target_Sz):
                    state[i] = "down_e down_i"
            psi = MPS.from_product_state(chain, state)
            return psi

    

class AndersonImpurityModel(CouplingMPOModel, NearestNeighborModel):
    r"""1D single-site Anderson impurity model 
    
    Paramters:

    model_params : dict

    Hamiltonian:
    $H = \sum_\sigma [ \sum_k 
    {\epsillon_k n_{k, \sigma} 
    + t (c^\dag _{k, \sigma}  c_{k + 1, \sigma} + c^\dag _{k, \sigma}  c_{k + 1, \sigma}  }
    + V (c^\dag_{0, \sigma}  c_{imp, \sigma}   + c^\dag_{imp, \sigma}  c_{0, \sigma}  
    + \mu n_{d, \sigma} ] 
    + U n_{d, \uparrow} n_{d, \downarrow}  $
    
    """

    couplings = {'t': 'hopping',   #Lanczos off diagonals type: list, int
                'mu' : 'conduction chemical potential',  #Lanczos diagonals type: list, int
                'mu_d': 'impurity potential',    #on-site impurity potential
                'U' : 'impurity Hubbard interaction',   #impurity Hubbard    
                'V' : 'exchange coupling',      #impurity-bath hopping
                'h' : 'Zeeman coupling'}  #breaks the SU(2) spin symmetry      

    def __init__(self, model_params):
        super().__init__(model_params)
        self.model_params = model_params

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N')
        cons_Sz = model_params.get('cons_Sz', None)
        return SpinHalfFermionSite(cons_N= conserve , cons_Sz= cons_Sz)

    def init_lattice(self, model_params):
        L = model_params['L']
        site = self.init_sites(model_params)
        return Chain(L + 1, site, bc='open')

    def init_terms(self, model_params):

        t = model_params['t']
        mu = model_params.get('mu', 0)
        U = model_params['U']
        mu_d = model_params['mu_d']
        V = model_params['V']
        h = model_params.get('h', 0)
        L = self.lat.N_sites
        imp = 0    #location of impurity in Lanczos chain is the 0th element
                   #note that this is hard coded in some places

        if type(t) in [float , int]:
            t = [float(t)] * L  
        else:
            t = list(t) * (L // len(t))
        t = [V* np.sqrt(2)] + t

        # hopping
        for i in range(L-1):
            self.add_coupling_term(-1.0 * t[i], i, i+1, 'Cdu', 'Cu', plus_hc= True, op_string = 'JW')
            self.add_coupling_term(-1.0 * t[i], i, i+1, 'Cdd', 'Cd', plus_hc= True, op_string = 'JW')

        if type(mu) in [float , int]:
            mu = [float(mu)] * L
        else:
            mu = list(mu) * (L // len(mu))
        mu = [mu_d] + mu


        # onsite potentials
        for i in range(L):
            self.add_onsite_term(mu[i], i, 'Ntot')

        # impurity Hubbard interaction
        self.add_onsite_term(U, imp, 'NuNd')

        # h: symmetry breaking Zeeman field
        if h != 0:
            # Sz Sz coupling
            for i in range(L):
                self.add_onsite_term(-1.0 * h, i, "Sz")


    def _set_initial_state(self, target_Sz = None):
        '''Returns an initial state suitable for the AIM_L which is compatible with DMRG'''

        model_params = self.model_params
        L = len(self.lat.mps_sites())

        if target_Sz is None:
            target_Sz = 0

        p_state = []
        i = 0
        t = 0
        site = self.init_sites(model_params)
        while i < L:
            while t < target_Sz:
                p_state.append('up') 
                i += 1
                t += 1
            if i != L-1:
                p_state.append('down') 
            else: 
                p_state.append('empty')
            i += 1
            if i != L:
                p_state.append('up') 
                i += 1
        return MPS.from_product_state(self.lat.mps_sites(), p_state, bc='finite')
             

class AndersonImpurityModel_Bulk(CouplingMPOModel, NearestNeighborModel):
    r"""1D single-site Anderson impurity model with impurity located in the bulk
    
    Paramters:

    model_params : dict

    Hamiltonian:
    $H = \sum_\sigma [ \sum_i 
    { t (c^\dag _{i, \sigma}  c_{i + 1, \sigma} + c^\dag _{i, \sigma}  c_{i + 1, \sigma}  }
    + V (c^\dag_{L//2, \sigma}  c_{imp, \sigma}   + c^\dag_{imp, \sigma}  c_{L//2, \sigma}  
    + V (c^\dag_{L//2 - 1, \sigma}  c_{imp, \sigma}   + c^\dag_{imp, \sigma}  c_{L//2 - 1, \sigma}  
    + \mu n_{d, \sigma} ] 
    + U n_{d, \uparrow} n_{d, \downarrow}  $
    
    """

    couplings = {'t': 'hopping',   #bath site hopping
                'mu_d': 'impurity potential',    #on-site impurity potential
                'U' : 'impurity Hubbard interaction',   #impurity Hubbard    
                'V' : 'exchange coupling',      #impurity-bath hopping
                'h' : 'Zeeman coupling',  #breaks the SU(2) spin symmetry     
                'mu': 'bath chemical potential'}   #chemical potential for conduction band 

    def __init__(self, model_params):
        super().__init__(model_params)
        self.model_params = model_params

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', None)
        cons_Sz = model_params.get('cons_Sz', None)
        #Unit cell must be doubled in order to accomadate the impurity site at L//2
        ferm_site1 = SpinHalfFermionSite(cons_N= conserve, cons_Sz= cons_Sz )
        ferm_site2 = SpinHalfFermionSite(cons_N= conserve, cons_Sz= cons_Sz )
        set_common_charges([ferm_site1, ferm_site2], new_charges='same')
        
        #left and right sites are labeled with 'l' 'r'
        site = GroupedSite([ferm_site1, ferm_site2], labels = ['l', 'r'] , charges = 'same') 
        return site

    def init_lattice(self, model_params):
        L = model_params['L'] // 2 #half the chain length because the unit cell was doubled
        site = self.init_sites(model_params)
        return Chain(L, site, bc='open')

    def init_terms(self, model_params):
        
        #see AndersonImpurityModel.couplings
        try:
            t = model_params['t']
            U = model_params['U']
            L = self.lat.N_sites
            mu_d = model_params['mu_d']
            mu = model_params.get('mu', 0)
            V = model_params['V']
            h = model_params.get('h', 0)
        except:
            KeyError("Make sure model_params contains values for (t, U, mu_d, V)")
        imp = L // 2    #location of impurity in lattice. impurity is on left side

        # conduction electron hopping
        for op_r, op_l in [['Cdur', 'Cul'], ['Cddr', 'Cdl']]:
            for i in range(L-1):
                if i+1 != imp:
                    self.add_coupling_term(-1.0 * t, i, i + 1, op_r, op_l, plus_hc= True)
                else:
                    self.add_coupling_term(-1.0 * V, i, i + 1, op_r, op_l, plus_hc= True)
                
        for i in range(L):
            if i != imp:
                self.add_onsite_term(-1.0 * t, i, 'Cdul Cur', plus_hc= True)
                self.add_onsite_term(-1.0 * t, i, 'Cddl Cdr', plus_hc= True)
            else:
                # bath electrons hop passed the impurity
                self.add_coupling_term(-1.0 * t, i-1, i , 'Cdur', 'Cur' , plus_hc= True )
                self.add_coupling_term(-1.0 * t, i-1, i , 'Cddr', 'Cdr' , plus_hc= True)

        # impurity/conduction onsite energy
        for i in range(L):
            if i == imp:
                self.add_onsite_term(mu_d, i, 'Ntotl')
                self.add_onsite_term(mu, i, 'Ntotr')
            else:
                self.add_onsite_term(mu, i, 'Ntotl')
                self.add_onsite_term(mu, i, 'Ntotr')

        # impurity Hubbard interaction
        self.add_onsite_term(U, imp, 'NuNdl')

        # h: symmetry breaking Zeeman field
        if h != 0:
            for u in range(len(self.lat.unit_cell)):
                # Sz Sz coupling
                self.add_onsite(-1.0 * h, u, "Szl")
                self.add_onsite(-1.0 * h, u, "Szr")

    def impurity_site(self):
        '''Returns the integer index of the impurity site'''

        L = self.lat.N_sites
        return L//2

    def _set_initial_state(self, target_Sz):
        r'''Returns a compatible initial state within the requested Sz sector'''

        L = len(self.lat.mps_sites())
        p_state = []
        model_params = self.model_params
        site = self.init_sites(model_params)
        target_Sz = target_Sz + L
        
        if target_Sz > 2 * L -1:
            raise ValueError('L')
        if target_Sz <= L-1:
            for i in range( L-1):
                if i < target_Sz:
                    p_state.append('up_l down_r') 
                else:
                    p_state.append('down_l down_r') 
            p_state.insert(L//2, 'down_l down_r')
        else: 
            for i in range( L):
                if i < target_Sz - L+1:
                    if i == L//2:
                        p_state.append('down_l up_r') 
                    else:
                        p_state.append('up_l up_r') 
                else:
                    if i == L//2:
                        p_state.append('down_l down_r') 
                    else:
                        p_state.append('up_l down_r') 
        return MPS.from_product_state(self.lat.mps_sites(), p_state, bc='finite')


class TFIsingChain(CouplingMPOModel, NearestNeighborModel):
    r"""1D single-site Transferse Field Ising Chain 
    
    Parameters:

    model_params : dict

    Hamiltonian: 
    $ H =  \sum_i [ J \sigma^z_i \sigma^z_{i + 1} - h \sigma^x_i - h_z \sigma^z_i ]
    $
    """

    couplings = {'J' : 'Ising Coupling',   #nearest-neighbor ZZ interaction
                'h ': 'transverse field',    #X magnetic field
                'hz' : 'longitudinal field'}  #breaks the Z2 spin symmetry      

    def __init__(self, model_params):
        super().__init__(model_params)
        self.model_params = model_params

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', None) 
        return SpinHalfSite(conserve= conserve)
    
    def init_lattice(self, model_params):
        L = model_params['L']
        site = self.init_sites(model_params)
        return Chain(L, site, bc='open')

    def init_terms(self, model_params):
        
        J = model_params['J']
        h = model_params['h']
        hz = model_params.get('hz', 0)

        # Ising interaction
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling( J, u1, 'Sigmaz', u2, 'Sigmaz', dx)

        # transvere field
        self.add_onsite(-1.0 * h, 0, 'Sigmax')

        # longitudinal - z field
        self.add_onsite(-1.0 * hz, 0, 'Sigmaz')

    def _set_initial_state(self, target_Sz = None):
            '''Returns an initial state suitable for the AIM_L which is compatible with DMRG'''
    
            model_params = self.model_params
            L = len(self.lat.mps_sites())
    
            if target_Sz is None:
                target_Sz = 0
    
            p_state = []
            site = self.init_sites(model_params)
            while i < L:
                while t < target_Sz:
                    p_state.append('up') 
                    i += 1
                    t += 1
                p_state.append('down') 
                i += 1
                if i != L:
                    p_state.append('up') 
                    i += 1
            return MPS.from_product_state(self.lat.mps_sites(), p_state, bc='finite')
                