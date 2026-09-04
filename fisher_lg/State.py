import itertools

import numpy as np

from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg, tebd
from tenpy.networks.mpo import MPO

from fisher_lg.Models import KondoChain, AndersonImpurityModel, TFIsingChain


class State_Evo(MPS):
    '''State_Evo is a class copied from tenpy.networks.mps.MPS which contains functions 
    useful for calculating the relevant features of the ground sates of quantum chains.
    '''

    model_dict = {
        "Ising" : TFIsingChain, 
        "Kondo" : KondoChain,  
        "AIM" : AndersonImpurityModel}

    def __init__(self,
                 sites,
                 Bs,
                 SVs,
                 bc='finite',
                 form='B',
                 norm=1.0,
                 unit_cell_width=None,
                 understood_shift_symmetry=False,
                 model_params = None,
                 extra_params = None,
                 energy = None,
                 initial_state = None):

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

        
    @classmethod
    def from_dmrg(cls, model_params, dmrg_params=None, Q_model = None, initial_state = None):
        '''Otputs State class as the ground state of a dmrg'''

        model_type = model_params.get("model_type", None)
        model_dict = cls.model_dict
        bc = model_params.get("bc", 'finite')
        
        if Q_model is None: 
            if model_params is None:
                raise ValueError("Provide model_params")
            model_params = dict(model_params)
            
            try:
                model = model_dict[model_type](model_params)
            except:
                raise KeyError(f"Unsupported model_type: {model_type}")
        else: 
            model = Q_model
            model_params = dict(model_params)

        if dmrg_params is None:
            dmrg_params = {'mixer': None,  # setting this to True helps to escape local minima
                'max_E_err': 1.e-10,
                'trunc_params': {
                    'chi_max': 100,
                    'svd_min': 1.e-10,
                },
                'verbose': False,
                'combine': True,
                'N_times' : 4,  #Only even N_times are used for K,
                'dt': 0.02,'N_steps': 2,
                'order': 2,
                'start_time' : 0  ,
                't_max' : 3,  #only an option for calculating the maximum of the LG correlator
                'check_after' : 5
            }

        #run dmrg
        if initial_state is None:
            psi_init = model._set_initial_state(target_Sz = model_params.get('target_Sz', None))
        else:
            psi_init = initial_state

        eng = dmrg.TwoSiteDMRGEngine(psi_init, model, dmrg_params)
        energy, psi_temp = eng.run()

        obj = cls(
            psi_temp.sites,
            [B.copy() for B in psi_temp._B],
            [S.copy() if S is not None else None for S in psi_temp._S],
            psi_temp.bc,
            psi_temp.form,
            psi_temp.norm,
            psi_temp.unit_cell_width,
        )

        obj.energy = energy
        obj.model = model
        obj.model_params = dict(model_params)
        obj.extra_params = dict(dmrg_params)
        obj.model_type = dict(model_params).get("model_type", None)
        obj.initial_state = initial_state
        obj.bc = bc
        obj.initial_state = psi_init

        return obj
    
    @classmethod
    def from_MPS(cls, psi : MPS, model_params = None, extra_params = None, energy = None):
        obj = cls(
        psi.sites,
        [B.copy() for B in psi._B],
        [S.copy() if S is not None else None for S in psi._S],
        psi.bc,
        psi.form,
        psi.norm,
        psi.unit_cell_width,
        )

        if model_params is not None:
            obj.set_model(model_params = model_params)
        if extra_params is not None:
            obj.extra_params = extra_params
        obj.initial_state = None
        psi.energy = energy        

        return obj
    
    @classmethod
    def from_data(cls, data):
        '''create MPS object from object array configured for from_data classmethod'''
        sites, Bs, Ss, bc, form, norm, unit_cell_width, model_params, extra_params, energy = data
        obj = cls(
        sites,
        Bs,
        Ss,
        bc,
        form,
        norm,
        unit_cell_width,
        )

        obj.set_model(model_params = dict(model_params))
        obj.extra_params = dict(extra_params)
        obj.initial_state = None
        obj.energy = energy

        return obj 
    
    @classmethod
    def from_dmrg_search_Sz(cls, model_params, dmrg_params=None, scan_Sz_with_Sp = False):
        '''When conserve is set to Sz it is important to search all possible 
        Sz sectors for the proper ground state. This is implemented here by 
        scanning through the even values of Sz, then searching the neighboring 
        odd values of the best even Szy'''

        model_type = model_params["model_type"]
        conserve = model_params.get('conserve'  , None )
        
        if model_type != "Kondo":
            raise ValueError("Sz search currently implemented only for Kondo.")
        if conserve is None:
            raise ValueError("conserve must be not None for specialized sector algorithm")
        
        model = cls.model_dict[model_type](dict(model_params))
        L = len(model.lat.mps_sites())

        best_energy = np.inf
        best_psi = None
        best_Sz = None
        psi_init = None

        for target_Sz in range(0, L + 1, 2):

            if (psi_init is None) or (not scan_Sz_with_Sp):   
                psi_init = model._set_initial_state( target_Sz)
                #psi_init.set_extra_params(extra_params = dmrg_params)
        
            else:
                psi_init = psi_init.apply_Q_MPO('Spi', op_sum = True)
                psi_init = psi_init.apply_Q_MPO('Spi', op_sum = True)
                psi_init.canonical_form()
                psi_init.norm = 1
        
            eng = dmrg.TwoSiteDMRGEngine( psi_init, model, dmrg_params )
            energy, psi_temp = eng.run()

            if energy < best_energy:
                best_energy = energy
                best_psi = psi_temp.copy()
                best_Sz = target_Sz
                best_init = State_Evo.from_MPS(psi_init)

        neighbor_list = [-1, 1] if best_Sz != L else [-1]
        best_Sz_temp = best_Sz

        for i in neighbor_list:
            psi_init = model._set_initial_state(target_Sz = best_Sz_temp + i)
            eng = dmrg.TwoSiteDMRGEngine( psi_init, model, dmrg_params )

            energy, psi_temp = eng.run()

            if energy < best_energy:
                best_energy = energy
                best_psi = psi_temp.copy()
                best_Sz = best_Sz_temp + i
                best_init = State_Evo.from_MPS(psi_init)
        
        
        obj = cls(
            best_psi.sites,
            [B.copy() for B in best_psi._B],
            [None if S is None else S.copy() for S in best_psi._S],
            best_psi.bc,
            best_psi.form,
            best_psi.norm,
            best_psi.unit_cell_width,
        )

        obj.energy = best_energy
        obj.model = model
        obj.model_params = dict(model_params)
        obj.extra_params = dict(dmrg_params)
        obj.model_type = model_type
        obj.best_Sz = best_Sz
        obj.initial_state = best_init

        return obj


    def copy(self):
        """Returns a copy of `self`.
        """
        # __init__ makes deep copies of B, S
        #MPS copy
        cp = super().copy()

        #State_Evo properties
        cp.model = self.model
        cp.model_params = self.model_params
        cp.extra_params = self.extra_params
        try:
            cp.energy = self.energy
        except ValueError:
            cp.energy = None

        cp.model_type = dict(self.model_params).get("model_type", None)
        cp.initial_state = self.initial_state
        cp.bc = self.bc

        return cp
    
    @property
    def available_models(self):
        """List of model types which have been included manually. 
        from_dmrg has the option of supplying a preconfigured model along with its
        parameters"""
        return self.__class__.model_dict

    
    def set_model(self, model_params = None):
        '''sets self.model and self.model_params and self.model_type and returns the dictionary'''
        
        model_dict = self.available_models
        if model_params is None:
            raise ValueError("Provide model_params")
        model_params = dict(model_params)
        model_type = model_params.get("model_type", "None")
        try:
            model = model_dict[model_type](model_params)
        except:
            raise KeyError(f"Unsupported model_type: {model_type}")
        
        self.model = model
        self.model_params = model_params
        self.model_type = model_type

        return model_params

    def set_extra_params(self, extra_params = None):
        '''Sets self.extra_params and returns the dictionary'''

        if extra_params is None:
            raise ValueError("Provide extra_params")
        self.extra_params = dict(extra_params)

        return dict(extra_params)

    def set_energy(self, E0 = None):
        '''sets and returns the GS energy assuming the model, model_params, and extra_params properties
            have already been specified'''
        if E0 is None:
            if self.model_params is None:
                raise ValueError ('Provide either E0 or use .set_model(model_params) to set the model')
            #run dmrg
            psi_temp = MPS.from_lat_product_state(self.model.lat, [['up']])
            eng = dmrg.TwoSiteDMRGEngine(psi_temp, self.model, self.extra_params)
            energy, psi_temp = eng.run()
            self.energy = energy
        elif type(E0) == float or np.float64:
            self.energy = E0
        else:
            raise ValueError("E0 must be float, np.float, or None instead of " + str(type(E0)))
        
        return self.energy     

    def get_state_labels(self):     
        """Returns the state labels of the model corresponding to model_type"""  
        
        return self.model.init_sites(self.model_params).state_labels    #for fiding the names of the initial states
            
    def get_opnames(self):     
        """Returns the state labels of the model corresponding to model_type"""  
        
        return self.model.init_sites(self.model_params).opnames  #for fiding the names of the initial states
            


    def QFI(self, op, op_sum = True, site = None):
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
        
        if op_sum: 
            # Calculate the expectation value of the operator
            expe = np.mean(self.expectation_value(op))

            # Calculate the correlation function
            corr_q2 = self.correlation_function(op, op, sites1=None, sites2=None)
            # Calculate the expectation value of the squared operator
            expectation_q2 = np.mean(corr_q2)
        else:
            op_site = self.L//2 if site is None else site
            expe = self.expectation_value(op)[op_site]
            expectation_q2 = self.correlation_function(op, op, sites1=[op_site], sites2=[op_site])
        
        # Calculate QFI using the formula: QFI = 4 * (expectation_q2 - expe^2)
        qfi = 4 * (expectation_q2 - expe**2).flatten()[0]
        
        return qfi
    
    def spin_spin_correlations(self, sites1 = None, sites2 = None):
        r'''Calculates spin-spin correlation function as an array < \\vec S_1 . \\vec S_2 >
        Parameters:
        self = State_Evo(MPS) state
        site1 = array of sites for S_1. If None, site1 = range(0,L)
        site2 = array of sites for S_2. If None, site2 = range(0,L)

        returns an np.array of the correlation matrix or a dictionary of correlation matrices if
        the unit cell has more than one site
        '''
        from tenpy.networks.site import GroupedSite

        if isinstance(self.model.lat.unit_cell[0], GroupedSite):
            labels = self.model.lat.unit_cell[0].labels
        else:
            labels = ['']
        SS = {}
        #calculate Sz Sz correlation
        for S_site1 in labels:
            for S_site2 in labels:
                SzSz = np.array(self.correlation_function(ops1 = 'Sz'+S_site1, ops2 = 'Sz'+S_site2, sites1 = sites1, sites2 = sites2))
                SmSp = np.array(self.correlation_function(ops1 = 'Sm'+S_site1, ops2 = 'Sp'+S_site2, sites1 = sites1, sites2 = sites2))
                SpSm = np.array(self.correlation_function(ops1 = 'Sp'+S_site1, ops2 = 'Sm'+S_site2, sites1 = sites1, sites2 = sites2))
                            
                SS[S_site1 + S_site2] = np.real(SzSz + (1./2.) * (SmSp + SpSm))

        if len(SS) == 1:
            return list(SS.values())[0]
        else:
            return SS


        
    def apply_Q_MPO(self, op, op_sum = True, site = None, new_chi_max = None):

        phi = self.copy()
        L = self.L
        model = self.model
        model_params = self.model_params
        extra_params = self.extra_params
        compression_method = extra_params.get('compression_method', 'SVD')
        chi_max = new_chi_max if new_chi_max is not None else extra_params['trunc_params']['chi_max']

        if site is None:
            site = L//2
        if op_sum:
            mpo = build_Q_MPO( op = op, op_sum = op_sum, model = model)
            mpo.apply(phi, options={'compression_method' : compression_method, 'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1e-9}})
        else:
            phi.apply_local_op(site, op)

        phi = State_Evo.from_MPS(phi, model_params)
        phi.set_energy(self.energy)
        phi.set_extra_params(extra_params = extra_params)
        
        return  phi
    

    def TEBD_initialize(self, tebd_params = None):
        """Initialize the TEBD engine with appropriate start time"""

        model = self.model
        extra_params = self.extra_params if tebd_params is None else tebd_params
        
        dt = extra_params.get('dt', 0.05)
        start_time = extra_params.get('start_time',0)

        phi_t = self.copy()
        engine = tebd.TEBDEngine(phi_t, model, extra_params)
        engine.evolved_time = 0

        if start_time > 0 and start_time < dt:
            engine.calc_U(order = 2, delta_t = start_time)
            engine.evolve( N_steps = 1,  dt = start_time)
        elif start_time > 0:
            initial_steps = int(start_time / dt)
            dt_prime = round(start_time/initial_steps, 4)
            engine.calc_U(order = 2, delta_t = dt_prime )
            engine.evolve(initial_steps,   dt = dt_prime)
        engine.calc_U(order = 2, delta_t = dt )

        return phi_t, engine


    def calculate_QtQ(self, options ):
        """ Calculates 1/2 < psi | {Q1^dag(t), Q2} | psi > 
            It is important to note the hermitian conjugate Q1^dag(t). This means the result is always
            real even when Q is non-hermitian, e.g. when calculating Sx from Sp and Sm. 

            scalar_product multiplies the y-axis data to scale for easy conversion of the spin operators to 
            Pauli operators. 
              """
        op1 = options['op1']
        op2 = options.get('op2', None)
        op_sum = options['op_sum']
        site = options.get('site', 0)
        calc_upto_2t = options.get('calc_upto_2t', False)
        scalar_product = options.get('scalar_product', 1.)
        verbose = options.get('verbose', False)
        
        extra_params = self.extra_params.copy()
        E0 = self.energy
        N_times = extra_params.get('N_times', 10)
        assert N_times %2 ==0

        #DMRG engine
        # apply phi = Q1 |psi > Must be careful here because some processes
        # in MPS and MPO by default normalize the output wavefunction
        phi = self.apply_Q_MPO( op1, op_sum = op_sum, site = site, new_chi_max = options.get('phi_chi', None))

        if op2 is None: #xi = Q2 |psi >
            xi = phi
        else:
            xi = self.apply_Q_MPO( op2, op_sum = op_sum, site = site)
        
        # TEBD engine
        phi_tebd_params = extra_params.copy()
        phi_t_chi_max = options.get('phi_t_chi', None)
        if phi_t_chi_max is not None:
            phi_tebd_params['trunc_params']['chi_max'] = phi_t_chi_max
        phi_t, engine = phi.TEBD_initialize()
        start_time = engine.evolved_time

        times = [engine.evolved_time]
        expec_U = [scalar_product * np.real(xi.overlap(phi_t)* np.exp(1j * E0 * start_time))]  # 1/2 <psi | {Q^dag (t), Q} | psi > = Re[< phi | U | phi > exp(i E_0 t)]

        if verbose != False:
            print("times", "QtQ data")
            print(times, expec_U)
        
        for _ in range(N_times):

            engine.run()
            t = engine.evolved_time
            times.append(round(t, 4))
            expec_U.append(scalar_product * np.real(xi.overlap(phi_t) * np.exp(1j * E0 * t)))
            if verbose != False:
                print(times, expec_U)
            
            
        if calc_upto_2t: #option used for calculate_LG_correlator which combines two legs 
                        #of time and expec_U for calculating 2 C(t) - C(2t) easily

            N_times = round( N_times /2)
            
            if extra_params.get('start_time', 0) != 0:
                raise ValueError("Needs array index to be proportional to time. start_time must be 0.")
            
            for _ in range(N_times):
                
                engine.run()
                engine.run()
                t= engine.evolved_time
                expec = scalar_product * np.real(xi.overlap(phi_t) * np.exp(1j * E0 * t))
                times += [-100, round(t, 4)]
                expec_U += [-100, expec] #so that if the implementation is wrong we will see it in the data
                if verbose != False:
                    print(times, expec_U)

        return times, expec_U


    def calculate_QQtQQ(self, options):
        """ Calculates 1/2 < psi |Q4^dag Q3^dag(t) Q2 Q1 + Q1^dag Q2^dag Q3(t) Q4| psi >  

            ops : list of list of 4 operator strings - will sum the contributions from each list of operators
            scalar_product : float - multiplies the y-axis data to scale for easy conversion of the spin operators to 
            Pauli operators. 
                """

        ops_list = options['op1']
        if isinstance(ops_list, str):
            ops_list = [ops_list] * 4
        op_sum = options['op_sum']
        site = options.get('site', 0)
        calc_upto_2t = options.get('calc_upto_2t', False)
        scalar_product = options.get('scalar_product', 1.)
        
        extra_params = self.extra_params.copy()
        N_times = extra_params.get('N_times', 10)
        step_size = extra_params['dt'] * extra_params['N_steps']
        list_len = 2 * N_times +1 if calc_upto_2t else N_times +1

        times = [round(i * step_size, 4) for i in range(N_times + 1)]
        if calc_upto_2t:
            times += [-100 if i % 2 == 0 else round(times[-1] +  step_size * (i +1), 4)  for i in range(N_times)]

        expec_tot = np.zeros(list_len)
        for op1, op2, op3, op4 in ops_list:

            #DMRG engine
            # apply phi = Q1 |psi > Must be careful here because some processes
            # in MPS and MPO by default normalize the output wavefunction
            phi1 = self.apply_Q_MPO( op1, op_sum = op_sum, site = site)
            phi21 = phi1.apply_Q_MPO( op2, op_sum = op_sum, site = site)
            phi4 = self.apply_Q_MPO( op4, op_sum = op_sum, site = site)
            
            # TEBD engines for first term
            phi21_t, engine_21 = phi21.TEBD_initialize()
            phi4_t, engine_4 = phi4.TEBD_initialize()

            #times = [engine_21.evolved_time]
            phi43_t = phi4_t.apply_Q_MPO(op3, op_sum = op_sum, site = site)
            expec_tot[0] += scalar_product * np.real(phi43_t.overlap(phi21_t))
            
            for n in range(N_times):

                engine_21.run()
                engine_4.run()
                phi43_t = phi4_t.apply_Q_MPO(op3, op_sum = op_sum, site = site)
                expec_tot[n+1] += scalar_product * np.real( phi43_t.overlap(phi21_t))
                
            if calc_upto_2t: #option used for calculate_LG_correlator which combines two legs 
                            #of time and expec_U for calculating 2 C(t) - C(2t) easily

                N_times_half = round( N_times /2)
                if extra_params.get('start_time', 0) != 0:
                    raise ValueError("Needs array index to be proportional to time. start_time must be 0.")
                
                for n in range(N_times_half):
                    
                    engine_21.run()
                    engine_21.run()
                    engine_4.run()
                    engine_4.run()
                    phi43_t = phi4_t.apply_Q_MPO(op3, op_sum = op_sum, site = site)
                    expec_tot[N_times + n * 2  + 1 ] = -100  #so that if the implementation is wrong we will see it in the data
                    expec_tot[N_times + n * 2  + 2 ] += scalar_product * np.real(phi43_t.overlap(phi21_t))
                                       
        return times, expec_tot
    
    def calculate_LG_correlator(self, options):
        """Returns  times, 2 [C(t) - C(0)] - [C(2t) - C(0)], 2C(t) - C(2t) """
        op = options['op1']
        op_sum = options['op_sum']
        site = options.get('site', 0)
        options['calc_upto_2t'] = True
        scalar_product = options.get('scalar_product', 1.)
        trichotomic = options.get('trichotomic', False)
        
        #Calculates C values from 0 to time_end
        if trichotomic:
            times, expec_U = self.calculate_QQtQQ(options)    #Calculates only the extra C values needed to calculate K
        else:
            times, expec_U = self.calculate_QtQ(options )    #Calculates only the extra C values needed to calculate K

        if self.model_params.get('start_time', 0) == 0:
            C0 = expec_U[0]  
        else:
            phi = self.apply_Q_MPO( op, op_sum = op_sum, site = site)
            C0 = scalar_product * np.real(phi.overlap(phi))

        half_len = len(times)//2 +1
        LG_B_lst = []
        K_lst = []
        # LG_B = 2C(t) - C(2t) - C(0)
        for i in range(half_len):
            LG_B_lst += [expec_U[i] * 2 - expec_U[2 * i] - C0 ]
            K_lst +=    [expec_U[i] * 2 - expec_U[2 * i] ]
            
        return times[: half_len] , LG_B_lst, K_lst 


    def calculate_LG_bound(self, options ):
        """Calculates the lower bound on the QFI set by the LG correlator QFI > 8 LG_B_Max
        using a "search a little after the first local maximum" algorithm"""

        op = options['op1']
        op_sum = options['op_sum']
        site = options.get('site', 0)
        options['calc_upto_2t'] = True
        scalar_product = options.get('scalar_product', 1.)
        verbose = options.get('verbose', False)
        return_QFI = options.get('return_QFI', False)

        extra_params = self.extra_params
        E0 = self.energy

        #Initialize model and phi through building the Q_MPO
        phi = self.apply_Q_MPO( op, op_sum = op_sum, site = site)

        #Get QFI
        qfi = None
        if return_QFI:
            qfi = self.QFI( op, op_sum = op_sum, site = site)
        
        # TEBD engine
        phi_t, engine = phi.TEBD_initialize()
        start_time = engine.evolved_time

        #set up the maximum number of steps 
        t_max = extra_params.get('t_max', 5)
        N_steps = extra_params['N_steps']
        dt = extra_params['dt']
        step_max = int(max((t_max - start_time)/(N_steps * dt ),1))
        
        LG_B_vals = []
        best_LG_B = -np.inf
        best_t = 0
        i = 0
        i_max = np.inf
        local_max = -1.0 * np.inf
        check_after = extra_params.get('check_after', 5)

        times = [engine.evolved_time]
        expec_U = [scalar_product * np.real(phi.overlap(phi_t)* np.exp(1j * E0 * start_time))]  # 1/2 <psi | {Q(t), Q} | psi > = Re[< phi | U | phi > exp(i E_0 t)]

        if start_time == 0:
            C0 = expec_U[0]  
        else:
            phi = self.apply_Q_MPO( op, op_sum = op_sum, site = site)
            C0 = scalar_product * np.real(phi.overlap(phi))

        while i < step_max:     #loop until the end
            #construct the time correlator list
            engine.run()
            t = engine.evolved_time
            expec_U.append(scalar_product * np.real(phi.overlap(phi_t) * np.exp(1j * E0 * t)))
            times.append(t)

            while 2 *i <= len(expec_U)-1:    #when the correlator list is long enough calculate...
                LG_B = 2* expec_U[i] - expec_U[2*i] #- C0 
                LG_B_vals.append(LG_B)
                
                if LG_B > best_LG_B:    #keep track of the largest LG_B value
                    best_LG_B = LG_B
                    if verbose:
                        print('Current peak value:', LG_B, " at time t =", t)

                    # detect first local maximum
                if len(LG_B_vals) >= 3:
                    if (LG_B_vals[-2] > LG_B_vals[-3]) and (LG_B_vals[-2] >= LG_B_vals[-1]) and (LG_B_vals[-2] >= best_LG_B):
                        local_max = LG_B_vals[-2]   #LG_B
                        i_max = i                   #index
                        best_t = times[i]           #time
                        if verbose:
                            print('Local max found: ', local_max, ' at time t=' , best_t)
                        
                if i_max + check_after < i:   #check after 'check_after' steps passed the local maximum
                    if best_LG_B >= local_max:   #if there is a better LG_B then reset
                        i_max = np.inf
                    else:                       #if nothing beats it then return the last local maximum
                        if verbose:
                            print('No local maxima found after t =', best_t, ', searched until t_f =', t, 'LG(t_f) =', LG_B) 
                        return best_t,   local_max #*  8, qfi 
                i += 1

        if verbose:
            print('Algorithm finished without finding global maximum past t = 0')       
        return best_t, local_max , qfi


    def find_degeneracy(self, op = None, eps = 1e-9):
        '''Returns the degeneracy of the model with input model_params. 
        Also returns a list of the degenerate State_Evo, and the expectation of an operator'''

        extra_params = self.extra_params
        psi_cp= self.copy()
        energy = energy1 = self.energy
        deg_list = []
        op_expe_list = []

        while np.abs(energy - energy1) < eps:
            
            deg_list += [psi_cp]
            if op is not None:
                op_expe_list += [round(np.mean(psi_cp.expectation_value(op)), 4)]
            eng = dmrg.TwoSiteDMRGEngine(self.copy(), self.model, extra_params)
            eng.init_env(orthogonal_to = deg_list)
            energy1, psi_cp = eng.run()
            
        return len(deg_list), deg_list, op_expe_list

    def integrated_spin_correlation(self, Sz_corr = False):
        r'''returns the integrated spin correlation defined as
        
        Sigma(x) = 1 + \sum_i^x (< \vec S_i \cdot \vec S_d >)/(< \vec S_d \cdot \vec S_d >)

        as an array of length L
        '''
        Sigma_x = [1.]
        if Sz_corr:
            S_corr = self.correlation_function('Sz', 'Sz', sites1= [0])[0]
        else:
            S_corr = self.spin_spin_correlations(sites1= [0])[0]

        for i in range(1, len(S_corr)):
            Sigma_x += [Sigma_x[i-1] + float(S_corr[i]/S_corr[0])]

        return Sigma_x

        


def build_Q_MPO( op = 'Sigmax', op_sum = True, model = None, model_params = {}):
    """
    builds an MPO corresponding to Q = 1/L sum_i op_i
    if apply_to_GS then it ftn will return op | GS > as a MPS
    """
    if model is None:
        try:
            model_type = model_params['model_type']
            model = State_Evo.model_dict[model_type](model_params)
        except: 
            raise ValueError("Model type not configured yet")
    L = model.lat.N_cells
    bc = model_params.get('bc', 'finite')
    Id = model.lat.unit_cell[0].get_op("Id")
    if op not in model.lat.unit_cell[0].opnames:
        raise ValueError("operator "+ op," is not allowed. Please choose in " + str(model.lat.unit_cell[0].opnames))
    Q = model.lat.unit_cell[0].get_op(op)

    W = []
    for i in range(L):
        if op_sum:
            Wi = np.empty((2, 2), dtype=object)
            Wi[0, 0] = Id
            Wi[1, 1] = Id
            Wi[0, 1] = Q / L
            Wi[1, 0] = None
        else:
            Wi = np.empty((1, 1), dtype=object)
            Wi[0, 0] = Q if i == L//2 else Id
    
        W.append(Wi)

    mpo = MPO.from_grids(model.lat.mps_sites(), W, IdL=0, IdR=1 if op_sum else 0, bc = bc)

    return mpo

