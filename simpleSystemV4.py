import os
import torch
from ase.io import read
from schnetpack.md.utils import MDUnits, compute_centroid, batch_inverse
from schnetpack import Properties
from schnetpack.md.neighbor_lists import SimpleNeighborList
from tqdm import tqdm
from schnetpack.datasets import MD17
from schnetpack.md.simulation_hooks import logging_hooks
import numpy as np
import matplotlib.pyplot as plt
from schnetpack.md.utils import HDF5Loader
from schnetpack.atomistic import Atomwise, AtomisticModel
#from ModelV2 import Atomwise, AtomisticModel


# work directory
md_workdir = "/home/alexander/Dokumente/Uni/Bachelor/Gastegger/SimpleSystemV4"


test_path = '/home/alexander/Dokumente/Uni/Bachelor/Gastegger/SchnetPack/tests/data'

'''
# Load model and structure
#model_path = os.path.join(test_path, 'test_md_model.model')
model_path = '/home/alexander/Dokumente/Uni/Bachelor/Gastegger/forcetut/best_model'
molecule_path = os.path.join(test_path, 'test_molecule.xyz')

'''

model_path = "/home/alexander/Dokumente/Uni/Bachelor/Gastegger/oxygen/best_model"


oxygen_data_set = '/home/alexander/Dokumente/Uni/Bachelor/Gastegger/oxygen/o2_morse_d6_a3_r1.2_1000K.db'

oxygen_data = MD17(oxygen_data_set)

oxygen_atoms = []




 
oxygen_atoms.append(oxygen_data.get_atoms(idx=0))


# Number of molecular replicas
n_replicas = 1


system_temperature = 300  # Kelvin

time_step = 0.5  # fs
time_step2 = time_step * MDUnits.fs2atu



# Check if a GPU is available and use a CPU otherwise
'''
if torch.cuda.is_available():
    md_device = "cuda"
else:
    md_device = "cpu"
'''

md_device = "cpu"

class SimpleSystem:

    """
    Container for all properties associated with the simulated molecular system
    (masses, positions, momenta, ...). Uses atomic units internally.
    In order to simulate multiple systems efficiently dynamics properties
    (positions, momenta, forces) are torch tensors with the following
    dimensions:
        n_replicas x n_molecules x n_atoms x 3

    Static properties (n_atoms, masses, atom_types and atom_masks) are stored in
    tensors of the shape:
        n_atoms : 1 x n_molecules (the same for all replicas)
        masses : 1 x n_molecules x n_atoms x 1 (the same for all replicas)
        atom_types : n_replicas x n_molecules x n_atoms x 1 (are brought to this
                     shape in order to avoid reshapes during every calculator
                     call)
        atom_masks : n_replicas x n_molecules x n_atoms x 1 (can change if
                     neighbor lists change for the replicas)
    n_atoms contains the number of atoms present in every molecule, masses
    and atom_types contain the molcular masses and nuclear charges.
    atom_masks is a binary array used to mask superfluous entries introduced
    by the zero-padding for differently sized molecules.
    Finally a dictionary properties stores the results of every calculator
    call for easy access of e.g. energies and dipole moments.
    Args:
        device (str): Computation device (default='cuda').

    """

    def __init__(
        self,
        n_replicas,
        device=md_device,
        neighborlist=SimpleNeighborList,
        initializer=None,
    ):

        # Specify device
        self.device = device

        # molecules
        self.molecules = None

        # number of molecules and vector with the number of
        # atoms in each molecule
        self.n_replicas = n_replicas
        self.n_molecules = None
        self.n_atoms = None
        self.max_n_atoms = None

        # General static molecular properties
        self.atom_types = None
        self.masses = None
        self.atom_masks = None

        # Dynamic properties updated during simulation
        self.positions = None
        self.momenta = None
        self.forces = None

        # Property dictionary, updated during simulation
        self.properties = {}

        # Initialize neighbor list for the calculator
        self.neighbor_list = neighborlist

        """
        Brauche ich eigenen Initaliser? Schentpack Botzlmann sollte tun, doer?
        """
        # Initialize initial conditions
        self.initializer = initializer

    def load_molecules_from_xyz(self, path_to_file):
        """
        Wrapper for loading molecules from .xyz file
        Args:
            path_to_file (str): path to data-file
        """
        self.molecules = read(path_to_file)
        if not type(self.molecules) == list:
            self.molecules = [self.molecules]
        self.load_molecules(molecules=self.molecules)

    def load_molecules(self, molecules):
        """
        Initializes all required variables and tensors based on a list of ASE
        atoms objects.
        Args:
            molecules list(ase.Atoms): List of ASE atoms objects containing
            molecular structures and chemical elements.
        """

        # 1) Get maximum number of molecules, number of replicas and number of
        #    overall systems
        self.n_molecules = len(molecules)

        # 2) Construct array with number of atoms in each molecule
        self.n_atoms = torch.zeros(
            self.n_molecules, dtype=torch.long, device=self.device
        )

        for i in range(self.n_molecules):
            self.n_atoms[i] = molecules[i].get_number_of_atoms()

        # 3) Determine the maximum number of atoms present (in case of
        #    differently sized molecules)
        self.max_n_atoms = int(torch.max(self.n_atoms))

        # 4) Construct basic properties and masks
        self.atom_types = torch.zeros(
            self.n_replicas, self.n_molecules, self.max_n_atoms, device=self.device
        ).long()
        self.atom_masks = torch.zeros(
            self.n_replicas, self.n_molecules, self.max_n_atoms, device=self.device
        )
        self.masses = torch.ones(self.n_molecules, self.max_n_atoms, device=self.device)

        # Relevant for dynamic properties: positions, momenta, forces
        self.positions = torch.zeros(
            self.n_replicas, self.n_molecules, self.max_n_atoms, 3, device=self.device
        )
        self.momenta = torch.zeros(
            self.n_replicas, self.n_molecules, self.max_n_atoms, 3, device=self.device
        )

        # 5) Populate arrays according to the data provided in molecules
        for i in range(self.n_molecules):
            # Static properties
            self.atom_types[:, i, : self.n_atoms[i]] = torch.from_numpy(
                molecules[i].get_atomic_numbers()
            )
            self.atom_masks[:, i, : self.n_atoms[i]] = 1.0
            self.masses[i, : self.n_atoms[i]] = torch.from_numpy(
                molecules[i].get_masses() * MDUnits.d2amu
            )

            # Dynamic properties
            self.positions[:, i, : self.n_atoms[i], :] = torch.from_numpy(
                molecules[i].positions * MDUnits.angs2bohr
            )

        # 6) Do proper broadcasting here for easier use in e.g. integrators and
        #    thermostats afterwards
        self.masses = self.masses[None, :, :, None]
        self.atom_masks = self.atom_masks[..., None]

        # 7) Build neighbor lists
        if self.neighbor_list is not None:
            self.neighbor_list = self.neighbor_list(self)

        # 8) Initialize Momenta
        if self.initializer:
            self.initializer.initialize_system(self)

    @property
    def center_of_mass(self):
        """
        Compute the center of mass for each replica and molecule
        Returns:
            torch.Tensor: n_replicas x n_molecules x 1 x 3 tensor holding the
                          center of mass.
        """
        # Mask mass array
        masses = self.masses * self.atom_masks
        # Compute center of mass
        center_of_mass = torch.sum(self.positions * masses, 2) / torch.sum(masses, 2)
        return center_of_mass

    def remove_com(self):
        """
        Move all structures to their respective center of mass.
        """
        # Mask to avoid offsets
        self.positions -= self.center_of_mass[:, :, None, :]
        # Apply atom masks to avoid artificial shifts
        self.positions *= self.atom_masks
        
    def remove_com_translation(self):
        """
        Remove all components in the current momenta associated with
        translational motion.
        """
        self.momenta -= (
            torch.sum(self.momenta, 2, keepdim=True)
            / self.n_atoms.float()[None, :, None, None]
        )
        # Apply atom masks to avoid artificial shifts
        self.momenta *= self.atom_masks

    def remove_com_rotation(self, detach=True):
        """
        Remove all components in the current momenta associated with rotational
        motion using Eckart conditons.
        Args:
            detach (bool): Whether computational graph should be detached in
                           order to accelerated the simulation (default=True).
        """
        # Compute the moment of inertia tensor
        moment_of_inertia = (
            torch.sum(self.positions ** 2, 3, keepdim=True)[..., None]
            * torch.eye(3, device=self.device)[None, None, None, :, :]
            - self.positions[..., :, None] * self.positions[..., None, :]
        )
        moment_of_inertia = torch.sum(moment_of_inertia * self.masses[..., None], 2)

        # Compute the angular momentum
        angular_momentum = torch.sum(torch.cross(self.positions, self.momenta, -1), 2)

        # Compute the angular velocities
        angular_velocities = torch.matmul(
            angular_momentum[:, :, None, :], batch_inverse(moment_of_inertia)
        )

        # Compute individual atomic contributions
        rotational_velocities = torch.cross(
            angular_velocities.repeat(1, 1, self.max_n_atoms, 1), self.positions, -1
        )

        if detach:
            rotational_velocities = rotational_velocities.detach()

        # Subtract rotation from overall motion (apply atom mask)
        self.momenta -= rotational_velocities * self.masses * self.atom_masks

    @property
    def velocities(self):
        """
        Convenience property to access molecular velocities instead of the
        momenta (e.g for power spectra)
        Returns:
            torch.Tensor: Velocity tensor with the same shape as the momenta.
        """
        return self.momenta / self.masses


    @property
    def kinetic_energy(self):
        """
        Convenience property for computing the kinetic energy associated with
        each replica and molecule.
        Returns:
            torch.Tensor: Tensor of the kinetic energies (in Hartree) with
                          the shape n_replicas x n_molecules
        """
        # Apply atom mask
        momenta = self.momenta * self.atom_masks
        kinetic_energy = 0.5 * torch.sum(
            torch.sum(momenta ** 2, 3) / self.masses[..., 0], 2
        )
        return kinetic_energy.detach()

    @property
    def temperature(self):
        """
        Convenience property for accessing the instantaneous temperatures of
        each replica and molecule.
        Returns:
            torch.Tensor: Tensor of the instantaneous temperatures (in
                          Kelvin) with the shape n_replicas x n_molecules
        """
        temperature = (
            2.0
            / (3.0 * MDUnits.kB * self.n_atoms.float()[None, :])
            * self.kinetic_energy
        )
        return temperature


# Initialize the system
md_system = SimpleSystem(n_replicas, device=md_device)

# Load the structure
#md_system.load_molecules_from_xyz(molecule_path)
md_system.load_molecules(oxygen_atoms)


"""
Initialize the momenta, by drawing from a random normal distribution and rescaling the momenta to the desired
temperature afterwards. In addition, the system is centered at its center of mass.
Args:
    system (object): System class containing all molecules and their replicas.
"""
# Move center of mass to origin
md_system.remove_com()

# Initialize velocities
velocities = torch.randn(md_system.momenta.shape, device=md_system.device)

# Set initial system momenta and apply atom masks
md_system.momenta = velocities * md_system.masses * md_system.atom_masks


# Scale velocities to desired temperature
scaling = torch.sqrt(system_temperature / md_system.temperature)
md_system.momenta *= scaling[:, :, None, None]


class MDCalculatorError(Exception):
    """
    Exception for MDCalculator base class.
    """


pass

'''
# Velocity Verlet functions
def half_step(system):
    """
        Half steps propagating the system momenta according to:
        ..math::
            p = p + \frac{1}{2} F \delta t
        Args:
            system (object): System class containing all molecules and their
                             replicas.
        """
    system.momenta = system.momenta + 0.5 * system.forces * time_step2


def _main_step(system):
    """
    Propagate the positions of the system according to:
    ..math::
        q = q + \frac{p}{m} \delta t
    Args:
        system (object): System class containing all molecules and their
                         replicas.
    """
    system.positions = system.positions + time_step2 * system.momenta / system.masses
    system.positions = system.positions.detach()

'''


'''
Integrator Object is used by logging_hooks
'''

class SimpleIntegrator:
    """
    Basic integrator class template. Uses the typical scheme of propagating
    system momenta in two half steps and system positions in one main step.
    The half steps are defined by default and only the _main_step function
    needs to be specified. Uses atomic time units internally.
    If required, the torch graphs generated by this routine can be detached
    every step via the detach flag.
    Args:
        time_step (float): Integration time step in femto seconds.
        detach (bool): If set to true, torch graphs of the propagation are
                       detached after every step (recommended, due to extreme
                       memory usage). This functionality could in theory be used
                       to formulate differentiable MD.
    """

    def __init__(self, time_step, detach=True, device=md_device):
        self.time_step = time_step * MDUnits.fs2atu
        self.detach = detach
        self.device = device

    def main_step(self, system):
        """
        Main integration step wrapper routine to make a default detach
        behavior possible. Calls upon _main_step to perform the actual
        propagation of the system.
        Args:
            system (object): System class containing all molecules and their
                             replicas.
        """
        self._main_step(system)
        if self.detach:
            system.positions = system.positions.detach()
            system.momenta = system.momenta.detach()

    def half_step(self, system):
        """
        Half steps propagating the system momenta according to:
        ..math::
            p = p + \frac{1}{2} F \delta t
        Args:
            system (object): System class containing all molecules and their
                             replicas.
        """
        system.momenta = system.momenta + 0.5 * system.forces * self.time_step
        if self.detach:
            system.momenta = system.momenta.detach()

    def _main_step(self, system):
        """
        Main integration step to be implemented in derived routines.
        Args:
            system (object): System class containing all molecules and their
                             replicas.
        """
        raise NotImplementedError


class VelocityVerlet(SimpleIntegrator):
    """
    Standard velocity Verlet integrator for non ring-polymer simulations.
    Args:
        time_step (float): Integration time step in femto seconds.
    """

    def __init__(self, time_step, device=md_device):
        super(VelocityVerlet, self).__init__(time_step, device=device)

    def _main_step(self, system):
        """
        Propagate the positions of the system according to:
        ..math::
            q = q + \frac{p}{m} \delta t
        Args:
            system (object): System class containing all molecules and their
                             replicas.
        """
        system.positions = (
            system.positions + self.time_step * system.momenta / system.masses
        )
        system.positions = system.positions.detach()

# Setup the integrator
md_integrator = VelocityVerlet(time_step)

class SchnetPackCalculator:
    """
    MD calculator for schnetpack models.
    Args:
        model (object): Loaded schnetpack model.
        required_properties (list): List of properties to be computed by the calculator
        force_handle (str): String indicating the entry corresponding to the molecular forces
        position_conversion (float): Unit conversion for the length used in the model computing all properties. E.g. if
                             the model needs Angstrom, one has to provide the conversion factor converting from the
                             atomic units used internally (Bohr) to Angstrom: 0.529177...
        force_conversion (float): Conversion factor converting the forces returned by the used model back to atomic
                                  units (Hartree/Bohr).
        property_conversion (dict(float)): Optional dictionary of conversion factors for other properties predicted by
                                           the model. Only changes the units used for logging the various outputs.
        detach (bool): Detach property computation graph after every calculator call. Enabled by default. Should only
                       be disabled if one wants to e.g. compute derivatives over short trajectory snippets.
    """

    def __init__(
        self,
        model,
        required_properties,
        force_handle,
        position_conversion=1.0 / MDUnits.angs2bohr,
        force_conversion=1.0 / MDUnits.auforces2aseforces,
        property_conversion={},
        detach=True,
    ):
        """
        super(SchnetPackCalculator, self).__init__(
            required_properties,
            force_handle,
            position_conversion,
            force_conversion,
            property_conversion,
            detach,
        )
        """

        self.results = {}
        self.force_handle = force_handle
        self.required_properties = required_properties

        # Perform automatic conversion of units
        self.position_conversion = MDUnits.parse_mdunit(position_conversion)
        self.force_conversion = MDUnits.parse_mdunit(force_conversion)
        self.property_conversion = {
            p: MDUnits.parse_mdunit(property_conversion[p]) for p in property_conversion
        }
        self._init_default_conversion()

        self.detach = detach

        self.model = model

    def calculate(self, system):
        """
        Main routine, generates a properly formatted input for the schnetpack model from the system, performs the
        computation and uses the results to update the system state.
        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        inputs = self._generate_input(system)
        self.results = self.model(inputs)
        #print("hullo",self.results['hullo'])
        #print(self.results)
        self._update_system(system)

    def _generate_input(self, system):
        """
        Function to extracts neighbor lists, atom_types, positions e.t.c. from the system and generate a properly
        formatted input for the schnetpack model.
        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        Returns:
            dict(torch.Tensor): Schnetpack inputs in dictionary format.
        """
        positions, atom_types, atom_masks = self._get_system_molecules(system)
        neighbors, neighbor_mask = self._get_system_neighbors(system)

        inputs = {
            Properties.R: positions,
            Properties.Z: atom_types,
            Properties.atom_mask: atom_masks,
            Properties.cell: None,
            Properties.cell_offset: None,
            Properties.neighbors: neighbors,
            Properties.neighbor_mask: neighbor_mask,
        }
        
        #inputs_save = os.path.join(md_workdir, 'inputs')
        #torch.save(inputs, inputs_save)

        return inputs

    def _init_default_conversion(self):
        """
        Auxiliary routine to initialize default conversion factors (1.0) if no alternatives are given in
        property_conversion upon initializing the calculator.
        """
        for p in self.required_properties:
            if p not in self.property_conversion:
                self.property_conversion[p] = 1.0

    def _update_system(self, system):
        """
        Routine, which looks in self.results for the properties defined in self.required_properties and uses them to
        update the forces and properties of the provided system. If required, reformatting is carried out here.
        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """

        # Collect all requested properties (including forces)
        for p in self.required_properties:
            if p not in self.results:
                raise MDCalculatorError(
                    "Requested property {:s} not in " "results".format(p)
                )
            else:
                # Detach properties if requested
                if self.detach:
                    self.results[p] = self.results[p].detach()

                dim = self.results[p].shape
                system.properties[p] = (
                    self.results[p].view(
                        system.n_replicas, system.n_molecules, *dim[1:]
                    )
                    * self.property_conversion[p]
                )

            # Set the forces for the system (at this point, already detached)
            self._set_system_forces(system)

    def _get_system_molecules(self, system):
        """
        Routine to extract positions, atom_types and atom_masks formatted in a manner suitable for schnetpack models
        from the system class. This is done by collapsing the replica and molecule dimension into one batch dimension.
        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        Returns:
            torch.FloatTensor: (n_replicas*n_molecules) x n_atoms x 3 tensor holding nuclear positions
            torch.FloatTensor: (n_replicas*n_molecules) x n_atoms tensor holding nuclear charges
            torch.FloatTensor: (n_replicas*n_molecules) x n_atoms binary tensor indicating padded atom dimensions
        """
        positions = (
            system.positions.view(-1, system.max_n_atoms, 3) * self.position_conversion
        )

        atom_types = system.atom_types.view(-1, system.max_n_atoms)
        atom_masks = system.atom_masks.view(-1, system.max_n_atoms)
        return positions, atom_types, atom_masks

    def _set_system_forces(self, system):
        """
        Function to reformat and update the forces of the system from the computed forces stored in self.results.
        The string contained in self.force_handle is used as an indicator. The single batch dimension is recast to the
        original replica x molecule dimensions used by the system.
        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        forces = self.results[self.force_handle]
        #print(forces)
        system.forces = (
            forces.view(system.n_replicas, system.n_molecules, system.max_n_atoms, 3)
            * self.force_conversion
        )
        #print("WOO", system.forces)
        
    def _get_system_neighbors(self, system):
        """
        Auxiliary function, which extracts neighbor lists formatted for schnetpack models from the system class.
        This is done by collapsing the replica and molecule dimension into one batch dimension.
        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        Returns:
            torch.LongTensor: (n_replicas*n_molecules) x n_atoms x (n_atoms-1) tensor holding the indices of all
                              neighbor atoms.
            torch.LongTensor: (n_replicas*n_molecules) x n_atoms x (n_atoms-1) binary tensor indicating padded
                              dimensions.
        """
        if system.neighbor_list is None:
            raise ValueError("System does not have neighbor list.")
        neighbor_list, neighbor_mask = system.neighbor_list.get_neighbors()
        # neighbor_list = system.neighbor_list.neighbor_list
        # neighbor_mask = system.neighbor_list.neighbor_mask

        neighbor_list = neighbor_list.view(
            -1, system.max_n_atoms, system.max_n_atoms - 1
        )
        neighbor_mask = neighbor_mask.view(
            -1, system.max_n_atoms, system.max_n_atoms - 1
        )
        return neighbor_list, neighbor_mask


# Load the stored model
#with torch.no_grad():
#md_model = torch.load(model_path, map_location="cpu").to(md_device)
md_model = torch.load(model_path).to(md_device)


# Generate the calculator
md_calculator = SchnetPackCalculator(
    md_model,
    #required_properties=[Properties.energy, Properties.forces, 'forces_der'],    
    required_properties=[Properties.energy, Properties.forces],
    force_handle=Properties.forces,
    position_conversion="A",
    force_conversion="kcal/mol/A",
)





n_steps = 100

# md_simulator.simulate(n_steps)









'''
Jetzt der Simualtor der alles verkapselt
'''


"""
All molecular dynamics in SchNetPack is performed using the :obj:`schnetpack.md.Simulator` class.
This class collects the atomistic system (:obj:`schnetpack.md.System`), calculators (:obj:`schnetpack.md.calculators`),
integrators (:obj:`schnetpack.md.integrators`) and various simulation hooks (:obj:`schnetpack.md.simulation_hooks`)
and performs the time integration.
"""


class Simulator:
    """
    Main driver of the molecular dynamics simulation. Uses an integrator to
    propagate the molecular system defined in the system class according to
    the forces yielded by a provided calculator.
    In addition, hooks can be applied at five different stages of each
    simulation step:
     - Start of the simulation (e.g. for initializing thermostats)
     - Before first integrator half step (e.g. thermostats)
     - After computation of the forces and before main integrator step (e.g.
      for accelerated MD)
     - After second integrator half step (e.g. thermostats, output routines)
     - At the end of the simulation (e.g. general wrap up of file writes, etc.)
    This routine has a state dict which can be used to restart a previous
    simulation.
    Args:
        system (object): Instance of the system class defined in
                         molecular_dynamics.system holding the structures,
                         masses, atom type, momenta, forces and properties of
                         all molecules and their replicas
        integrator (object): Integrator for propagating the molecular
                             dynamics simulation, defined in
                             schnetpack.md.integrators
        calculator (object): Calculator class used to compute molecular
                             forces for propagation and (if requested)
                             various other properties.
        simulator_hooks (list(object)): List of different hooks to be applied
                                        during simulations. Examples would be
                                        file loggers and thermostats.
        step (int): Index of the initial simulation step.
        restart (bool): Indicates, whether the simulation is restarted. E.g. if set to True, the simulator tries to
                        continue logging in the previously created dataset. (default=False)
                        This is set automatically by the restart_simulation function. Enabling it without the function
                        currently only makes sense if independent simulations should be written to the same file.
    """

    def __init__(
        self, system, integrator, calculator, simulator_hooks=[], step=0, restart=False
    ):

        self.system = system        
        self.integrator = integrator
        self.calculator = calculator
        self.simulator_hooks = simulator_hooks
        self.step = step
        self.n_steps = None
        self.restart = restart

        # Keep track of the actual simulation steps performed with simulate calls
        self.effective_steps = 0

    def simulate(self, n_steps=100):
        """
        Main simulation function. Propagates the system for a certain number
        of steps.
        Args:
            n_steps (int): Number of simulation steps to be performed (
                           default=10000)
        """

        self.n_steps = n_steps

        # Perform initial computation of forces
        if self.system.forces is None:
            self.calculator.calculate(self.system)

        
        # Call hooks at the simulation start
        for hook in self.simulator_hooks:
            hook.on_simulation_start(self)
            
        #print(md_model.output_modules[0].parameters)
    
        paramter_list =[]
        
        
        for _ in tqdm(range(n_steps), ncols=120):

            
            # Call hook berfore first half step
            for hook in self.simulator_hooks:
                hook.on_step_begin(self)
            
            
            # Do half step momenta
            self.integrator.half_step(self.system)
            
            #print(MD17.forces)
            #print(md_model.output_modules[0].derivative.forces)
            #print(md_model.output_modules[0].parameters())
            #print(self.calculator.results["forces"])
            #print(self.calculator.results)
            

            # Do propagation MD/PIMD
            self.integrator._main_step(self.system)

            # Compute new forces
            self.calculator.calculate(self.system)
            '''
            Hier wird model aufgerufen, hier gibt es neuen graidenten
            '''
            
            #print(self.calculator.results)
            #print(md_model.output_modules[0].derivative)
            #print(MD17.forces)
            #print(md_model.features[:1])
            
            '''
            for name, param in md_model.output_modules[0].named_parameters():
                #print(name)
                print(name, param.grad.data)
            ''' 
                
            '''    
            for f in md_model.output_modules[0].parameters():
                f.backward()
                paramter_list.append(f.grad.data)
            
        
            print(paramter_list[0])
            '''
            
            # Call hook after forces
            for hook in self.simulator_hooks:
                hook.on_step_middle(self)
            
            
            # Do half step momenta
            self.integrator.half_step(self.system)

            
            # Call hooks after second half step
            for hook in self.simulator_hooks:
                hook.on_step_end(self)
            
        
            self.step += 1
            self.effective_steps += 1

        
        # Call hooks at the simulation end
        for hook in self.simulator_hooks:
            hook.on_simulation_end(self)
        
        
    @property
    def state_dict(self):
        """
        State dict used to restart the simulation. Generates a dictionary with
        the following entries:
            - step: current simulation step
            - systems: state dict of the system holding current positions,
                       momenta, forces, etc...
            - simulator_hooks: dict of state dicts of the various hooks used
                               during simulation using their basic class
                               name as keys.
        Returns:
            dict: State dict containing the current step, the system
                  parameters (positions, momenta, etc.) and all
                  simulator_hook state dicts
        """
        state_dict = {
            "step": self.step,
            "system": self.system.state_dict,
            "simulator_hooks": {
                hook.__class__: hook.state_dict for hook in self.simulator_hooks
            },
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        """
        Set the current state dict of the simulator using a state dict
        defined in state_dict. This routine assumes, that the identity of all
        hooks has not changed and the order is preserved. A more general
        method to restart simulations is provided below.
        Args:
            state_dict (dict): state dict containing the entries 'step',
            'simulator_hooks' and 'system'.
        """
        self.step = state_dict["step"]
        self.system.state_dict = state_dict["system"]

        # Set state dicts of all hooks
        for hook in self.simulator_hooks:
            if hook.__class__ in state_dict["simulator_hooks"]:
                hook.state_dict = state_dict["simulator_hooks"][hook.__class__]

    def restart_simulation(self, state_dict, soft=False):
        """
        Routine for restarting a simulation. Reads the current step, as well
        as system state from the provided state dict. In case of the
        simulation hooks, only the states of the thermostat hooks are
        restored, as all other hooks do not depend on previous simulations.
        If the soft option is chosen, only restores states of thermostats if
        they are present in the current simulation and the state dict.
        Otherwise, all thermostats found in the state dict are required to be
        present in the current simulation.
        Args:
            state_dict (dict): State dict of the current simulation
            soft (bool): Flag to toggle hard/soft thermostat restarts (
                         default=False)
        """
        self.step = state_dict["step"]
        self.system.state_dict = state_dict["system"]

        if soft:
            # Do the same as in a basic state dict setting
            for hook in self.simulator_hooks:
                if hook.__class__ in state_dict["simulator_hooks"]:
                    hook.state_dict = state_dict["simulator_hooks"][hook.__class__]
        else:
            # Hard restart, require all thermostats to be there
            for hook in self.simulator_hooks:
                # Check if hook is thermostat
                if hasattr(hook, "temperature_bath"):
                    if hook.__class__ not in state_dict["simulator_hooks"]:
                        raise ValueError(
                            f"Could not find restart information for {hook.__class__} in state dict."
                        )
                    else:
                        hook.state_dict = state_dict["simulator_hooks"][hook.__class__]

        # In this case, set restart flag automatically
        self.restart = True

    def load_system_state(self, state_dict):
        """
        Routine for only loading the system state of previous simulations.
        This can e.g. be used for production runs, where an equilibrated system
        is loaded, but the thermostat is changed.
        Args:
            state_dict (dict): State dict of the current simulation
        """
        self.system.state_dict = state_dict["system"]




# Path to database
log_file = os.path.join(md_workdir, 'simulation.hdf5')

# Size of the buffer
buffer_size = 100

# Set up data streams to store positions, momenta and all properties
data_streams = [
    logging_hooks.MoleculeStream(),
    logging_hooks.PropertyStream(),
]

# Create the file logger
file_logger = logging_hooks.FileLogger(
    log_file,
    buffer_size,
    data_streams=data_streams
)

# Update the simulation hooks
simulation_hooks = []
simulation_hooks.append(file_logger)
md_simulator = Simulator(md_system, md_integrator,md_calculator, simulator_hooks=simulation_hooks)

#print("hi",type(md_simulator.system.positions))

n_steps = 20

md_simulator.simulate(n_steps)

data = HDF5Loader(log_file)

from schnetpack.md.utils import MDUnits

# Get potential energies and check the shape
energies = data.get_property(Properties.energy)
#print('Shape:', energies.shape)

# Get the time axis
time_axis = np.arange(data.entries)*data.time_step / MDUnits.fs2atu # in fs

# Plot the energies
plt.figure()
plt.plot(time_axis, energies)
plt.ylabel('E [kcal/mol]')
plt.xlabel('t [fs]')
plt.tight_layout()
plt.show()


def plot_temperature(data):

    # Read the temperature
    temperature = data.get_temperature()

    # Compute the cumulative mean
    temperature_mean = np.cumsum(temperature) / (np.arange(data.entries)+1)

    # Get the time axis
    time_axis = np.arange(data.entries)*data.time_step / MDUnits.fs2atu # in fs

    plt.figure(figsize=(8,4))
    plt.plot(time_axis, temperature, label='T')
    plt.plot(time_axis, temperature_mean, label='T (avg.)')
    plt.ylabel('T [K]')
    plt.xlabel('t [fs]')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_temperature(data)