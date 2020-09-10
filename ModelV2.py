from schnetpack.datasets import MD17
import os
import schnetpack as spk
from torch.optim import Adam
import torch
from ase.visualize import view
import numpy as np
import matplotlib.pyplot as plt
from torch import nn as nn

import schnetpack
from schnetpack import nn as L, Properties
from torch.autograd import grad
from torch.autograd.functional import hessian

import datetime



forcetut = '/home/alexander/Dokumente/Uni/Bachelor/Gastegger/oxygen'

oxygen_data_set = '/home/alexander/Dokumente/Uni/Bachelor/Gastegger/oxygen/o2_morse_d6_a3_r1.2_1000K.db'

oxygen_data = MD17(oxygen_data_set)

print(len(oxygen_data))

print("Hey:\n",oxygen_data.get_atoms(idx=0))

atoms, properties = oxygen_data.get_properties(0)

print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])
print('Positons:\n',properties["_positions"])
'''

forcetut = '/home/alexander/Dokumente/Uni/Bachelor/Gastegger/forcetut'

ethanol_data_set = '/home/alexander/Dokumente/Uni/Bachelor/Gastegger/forcetut/ethanol.db'

ethanol_data = MD17(ethanol_data_set)

atoms, properties = ethanol_data.get_properties(0)
'''

'''
As can be seen, energy and forces are included in the properties dictionary.
To have a look at the forces array and check whether it has 
the expected dimensions, we can call:
'''
#print('Forces:\n', properties[MD17.forces])
#print('Shape:\n', properties[MD17.forces].shape)

'''
The atoms object can e.g. be used to visualize the ethanol molecule:
'''


view(atoms, viewer='x3d')

'''
Next, the data is split into training (1000 points),
 test (500 points) and validation set (remainder) and 
 data loaders are created. This is done in the same way 
 as described in the QM9 tutorial.
'''


train, val, test = spk.train_test_split(
        data=oxygen_data,
        #data=ethanol_data,
        num_train=1000,
        num_val=500,
        split_file=os.path.join(forcetut, "split.npz"),
    )

train_loader = spk.AtomsLoader(train, batch_size=12, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=12)


'''
Once again, we want to use the mean and standardeviation of the energies 
in the training data to precondition our model. This only needs to be done
 for the energies, since the forces are obtained as derivatives and automatically 
 capture the scale of the data. Unlike in the case of QM9, the subtraction of atomic
 reference energies is not necessary, since only configurations of the same 
 molecule are loaded. All this can be done via the get_statistics function of
 the AtomsLoader class:
'''

means, stddevs = train_loader.get_statistics(
    spk.datasets.MD17.energy, divide_by_atoms=True
)

print('Mean atomization energy / atom:      {:12.4f} [kcal/mol]'.format(means[MD17.energy][0]))
print('Std. dev. atomization energy / atom: {:12.4f} [kcal/mol]'.format(stddevs[MD17.energy][0]))

'''
Building the model

After having prepared the data in the above way, we can now build and train the force model. This is done in the same two steps as described in QM9 tutorial:

    Building the representation
    Defining an output module

For the representation we can use the same SchNet layer as in the previous tutorial:
'''
n_features = 128

schnet = spk.representation.SchNet(
    n_atom_basis=n_features,
    n_filters=n_features,
    n_gaussians=25,
    n_interactions=3,
    cutoff=5.,
    cutoff_network=spk.nn.cutoff.CosineCutoff
)

'''
Since we want to model forces, the Atomwise output module needs to be adapted 
slightly. We will still use one module to predict the energy, preconditioning
 with the mean and standard deviation per atom of the energy.

However, since the forces should be described as the derivative of the energy, 
we have to indicate that the corresponding derviative of the model should 
be computed. This is done by specifying derivative=MD17.forces, which also a
ssigns the computed derivative to the property MD17.forces. Since the forces 
are the negative gradient, we also need to enable negative_dr=True, 
which simply multipiies the derviative with -1.
'''

class AtomwiseError(Exception):
    pass

class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.
    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: 1)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        property (str): name of the output property (default: "y")
        contributions (str or None): Name of property contributions in return dict.
            No contributions returned if None. (default: None)
        derivative (str or None): Name of property derivative. No derivative
            returned if None. (default: None)
        negative_dr (bool): Multiply the derivative with -1 if True. (default: False)
        stress (str or None): Name of stress property. Compute the derivative with
            respect to the cell parameters if not None. (default: None)
        create_graph (bool): If False, the graph used to compute the grad will be
            freed. Note that in nearly all cases setting this option to True is not nee
            ded and often can be worked around in a much more efficient way. Defaults to
            the value of create_graph. (default: False)
        mean (torch.Tensor or None): mean of property
        stddev (torch.Tensor or None): standard deviation of property (default: None)
        atomref (torch.Tensor or None): reference single-atom properties. Expects
            an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
            property of element Z. The value of atomref[0] must be zero, as this
            corresponds to the reference property for for "mask" atoms. (default: None)
        outnet (callable): Network used for atomistic outputs. Takes schnetpack input
            dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)
    Returns:
        tuple: prediction for property
        If contributions is not None additionally returns atom-wise contributions.
        If derivative is not None additionally returns derivative w.r.t. atom positions.
    """

    def __init__(
        self,
        n_in,
        n_out=1,
        aggregation_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=schnetpack.nn.activations.shifted_softplus,
        property="y",
        contributions=None,
        derivative=None,
        negative_dr=False,
        stress=None,
        create_graph=True,
        mean=None,
        stddev=None,
        atomref=None,
        outnet=None,
        second_der = None,
        forces_der = "",
    ):
        super(Atomwise, self).__init__()

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative = derivative
        self.forces_der = 'forces_der'
        self.negative_dr = negative_dr
        self.stress = stress
        self.second_der = second_der
        
        #print("HOO")
        
        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        # initialize single atom energies
        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(
                torch.from_numpy(atomref.astype(np.float32))
            )
        else:
            self.atomref = None

        # build output network
        if outnet is None:
            self.out_net = nn.Sequential(
                schnetpack.nn.base.GetItem("representation"),
                schnetpack.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation),
            )
        else:
            self.out_net = outnet

        # build standardization layer
        self.standardize = schnetpack.nn.base.ScaleShift(mean, stddev)

        # build aggregation layer
        if aggregation_mode == "sum":
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=False)
        elif aggregation_mode == "avg":
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=True)
        else:
            raise AtomwiseError(
                "{} is not a valid aggregation " "mode!".format(aggregation_mode)
            )

    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        yi = self.out_net(inputs)
        yi = self.standardize(yi)

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0

        y = self.atom_pool(yi, atom_mask)

        # collect results
        result = {self.property: y}

        if self.contributions is not None:
            result[self.contributions] = yi

        if self.derivative is not None:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                result[self.property],
                inputs[Properties.R],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                retain_graph=True,
            )[0]
            #print("dy",dy.shape)
            
            #print("dy",list(dy.size()))
            #print("R",list(inputs[Properties.R].size()))
            #print("property",list(torch.ones_like(result[self.property]).size()))
           
            #print(result[self.property])            
            
            result[self.derivative] = sign * dy

            
            '''
            Das hier ist für F nach delta x also die BAleitng der forces
            '''
            
            
         
            
        if self.second_der is not None:
            
            '''
            forces_deriv = grad(
                grads_acc,
                inputs[Properties.R],
                grad_outputs=torch.ones_like(grads_acc),
                create_graph=self.create_graph,
                retain_graph=True,
                allow_unused=True,
            )[0]
            '''
            
            
            '''
            forces_deriv = hessian(
                forward,
                inputs[Properties.R],
                create_graph=self.create_graph,
            )
            '''
            
            
            '''
            dfdx ist ein [batch x natoms x 3] tensor.
            Für den Hessian flattet man alles bis auf die erste dim, 
            macht die Gradienten und sammelt das Ganze.
            '''
            
            '''
            dfdx = -dy
            B, A, C = dfdx.shape
            dfdx = dfdx.view(B,-1)
            #print(dfdx.shape)
            
            
           # for i in range(dfdx.shape[1]):
                #print(dfdx[:,i].shape)
            
            
            d2fdx2 = torch.stack(
                [grad(dfdx[:,i], 
                      inputs[Properties.R], 
                      torch.ones_like(dfdx[:,i]), 
                      create_graph=True, 
                      retain_graph=True)[0] 
                for i in range(dfdx.shape[1])], 
                dim=1
            )
            '''
            
            
            #print(d2fdx2.shape)
            
            '''
            
            A, B, C, D = d2fdx2.shape

            d2fdx2_reshaped = torch.tensor(np.zeros((A, C, D)))
            
            
            for i in range(d2fdx2.shape[0]):
                for j in range(d2fdx2.shape[1]):
                    d2fdx2_reshaped[i] += d2fdx2[i,j]
                   
            print(d2fdx2_reshaped.shape)

            '''
            '''
            Das sollte dann ein [batch x natoms*3 x natoms x 3] Tensor sein, 
            denn man nur mehr auf die richtige shape bringen muss.
            '''
            
            #print("grads_acc",list(grads_acc.size()))
            #print("R",list(inputs[Properties.R].size()))
            #print("property",list(result[self.property].size()))
            #forces_deriv = grad(result[self.property],dy)
            
            
            '''
            das hier ist für die gradienten der weights
            '''
            '''
            #print(self.out_net.named_children)
            parameter_grad_list = []
            for name, param in self.out_net[1].named_parameters():
            #for name in self.out_net[1].named_parameters():
                #print(name[1].grad)
                #if (name == 'out_net.1.weight') or (name == 'out_net.0.weight'):
                if (name == 'out_net.1.weight'):
                    if param.grad != None:
                        #print("beginn",param.grad.shape)
                        parameter_grad_list.append(param.grad)
            '''  
            #if len(parameter_grad_list) != 0:
                #print(torch.stack(parameter_grad_list).shape)
            #print(len(parameter_grad_list))
            #print(parameter_grad_list[1::2])
            #print("beginn", self.out_net)
            '''
            for name, param in self.out_net[0].named_parameters():
                print(name, param)
           '''
            
            '''
            train_parameter_list = []
            for name, param in self.out_net[1].named_parameters(): 
                if (name == 'out_net.0.weight') or (name == 'out_net.1.weight'):
                    if param != None:
                        for i in range(param.shape[0]):
                            for j in range(param.shape[1]):
                                train_parameter_list.append(param[i][j])
            '''
            train_parameter_list = []
            for name, param in self.out_net[1].named_parameters(): 
                if (name == 'out_net.0.weight') or (name == 'out_net.1.weight'):
                    if param != None:
                        train_parameter_list.append(param)
            
            #print("train_parameter_list",train_parameter_list[1].shape)
            '''
            Berehcnung der baleitung der forces
            Loop über paramter
            F[:,i,j] = einzelne forces werte, [batchsize x n-atoms x 3(x,y,z)]
            p = parameter
            '''
            '''
            print("ANFANG")
            #print("dy", dy.shape)
            hessian_list = []
            for parameter in train_parameter_list:
                #print("Ham",parameter.shape)
                for i in range(dy.shape[1]):
                    for j in range(dy.shape[2]):
                        #print("force", dy[:,i,j].shape)
                        hessian_grad= grad(dy[:,i,j], 
                                                parameter, 
                                                torch.ones_like(dy[:,i,j]), 
                                                retain_graph=True, 
                                                create_graph=True,
                                                allow_unused=True)[0]
                        hessian_list.append(hessian_grad)
                        
            
            #print("beginn",torch.stack(hessian_list))
            print(torch.stack(hessian_list).shape)
            '''
            
            A, B, C = dy.shape
            
            f_tmp = dy.view(A, -1)
            
            all_derivs = []
            for p in train_parameter_list:
                D, E = p.shape
                derivs = []
                for i in range(A):
                    for idx in range(B*C):
                        #print("blah",f_tmp[i,idx].shape)
                        dfdp = grad(f_tmp[i,idx], 
                                    p, 
                                    torch.ones_like(f_tmp[i,idx]), 
                                    retain_graph=True, 
                                    create_graph=True)[0]
                        #print("dfdp",dfdp.shape)
                        derivs.append(dfdp[:,None,...])
                        #print("dervis",len(derivs))
                # -> B x A*3 x paramdim
                all_derivs.append(
                    #torch.stack(derivs, dim=1).view(B, C, D, E)
                    torch.stack(derivs, dim=1).view(A, B, C, D, E)
                    )
                #print(torch.stack(all_derivs).shape)
               
            
            'all dervic etnhält jetzt 2 tensorend er form n-taoms x 3 x param dim 1 x param dim 2'
            #print(len(all_derivs))
            #print(all_derivs[1].shape)
            #print("grads_acc",grads_acc)
            #result.update({'forces_der': dy})
            result[self.forces_der] = dy
            #print(result['forces_der'])
            #print(forces_deriv)
            #print("deriv",list(forces_deriv.size()))


        if self.stress is not None:
            cell = inputs[Properties.cell]
            # Compute derivative with respect to cell displacements
            stress = grad(
                result[self.property],
                inputs["displacement"],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                retain_graph=True,
            )[0]
            # Compute cell volume
            volume = torch.sum(
                cell[:, 0, :] * torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                dim=1,
                keepdim=True,
            )[..., None]
            # Finalize stress tensor
            result[self.stress] = stress / volume

            result.update({'hullo': 'Blajh'})
    
        return result

energy_model = Atomwise(
    n_in=n_features,
    property=MD17.energy,
    mean=means[MD17.energy],
    stddev=stddevs[MD17.energy],
    derivative=MD17.forces,
    negative_dr=True,
    contributions = 'Contri',
    second_der = 'Der',
    forces_der = ''
)



'''
Both modules are then combined to an AtomisticModel.
'''


__all__ = ["AtomisticModel"]

class ModelError(Exception):
    pass


class AtomisticModel(nn.Module):
    """
    Join a representation model with output modules.
    Args:
        representation (torch.nn.Module): Representation block of the model.
        output_modules (list or nn.ModuleList or spk.output_modules.Atomwise): Output
            block of the model. Needed for predicting properties.
    Returns:
         dict: property predictions
    """

    def __init__(self, representation, output_modules):
        super(AtomisticModel, self).__init__()
        self.representation = representation
        if type(output_modules) not in [list, nn.ModuleList]:
            output_modules = [output_modules]
        if type(output_modules) == list:
            output_modules = nn.ModuleList(output_modules)
        self.output_modules = output_modules
        # For gradients
        self.requires_dr = any([om.derivative for om in self.output_modules])
        # For stress tensor
        self.requires_stress = any([om.stress for om in self.output_modules])

    def forward(self, inputs):
        """
        Forward representation output through output modules.
        """
        if self.requires_dr:
            inputs[Properties.R].requires_grad_()
        if self.requires_stress:
            # Generate Cartesian displacement tensor
            displacement = torch.zeros_like(inputs[Properties.cell]).to(
                inputs[Properties.R].device
            )
            displacement.requires_grad = True
            inputs["displacement"] = displacement

            # Apply to coordinates and cell
            inputs[Properties.R] = inputs[Properties.R] + torch.matmul(
                inputs[Properties.R], displacement
            )
            inputs[Properties.cell] = inputs[Properties.cell] + torch.matmul(
                inputs[Properties.cell], displacement
            )

        inputs["representation"] = self.representation(inputs)
        
        outs = {}
        
        for output_model in self.output_modules:
            outs.update(output_model(inputs))

        return outs


model = AtomisticModel(representation=schnet, output_modules=energy_model)



input_path = '/home/alexander/Dokumente/Uni/Bachelor/Gastegger/SimpleSystemV4/inputs'


inputs2 = torch.load(input_path)

#inputs.update({'contributions': True})

'''
print("INPUTS",inputs2)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model.to(device)

result2 = model(inputs2)

print("MAPPP",result2)

'''




'''
Training the model

To train the model on energies and forces, we need to update the loss function
 to include the latter. This combined loss function is:
L(Eref,Fref,Epred,Fpred)=1ntrain∑n=1ntrain[ρ(Eref−Epred)2+(1−ρ)3Natoms∑αNatoms∥∥F(α)ref−F(α)pred∥∥2],

where we take the predicted forces to be:
F(α)pred=−∂Epred∂R(α).

We have introduced a parameter ρ
in order to control the tradeoff between energy and force loss. 
By varying this parameter, the accuracy on energies and forces can be tuned.
 Setting ρ=0 only forces are trained, while in the case of ρ=1 only energies 
 are learned. Using PyTorch, we can implement this loss function in the 
 following way:
'''
print(forcetut)

# tradeoff
rho_tradeoff = 0.1

# loss function
def loss(batch, result):
    # compute the mean squared error on the energies
    diff_energy = batch[MD17.energy]-result[MD17.energy]
    err_sq_energy = torch.mean(diff_energy ** 2)

    # compute the mean squared error on the forces
    diff_forces = batch[MD17.forces]-result[MD17.forces]
    err_sq_forces = torch.mean(diff_forces ** 2)

    # build the combined loss function
    err_sq = rho_tradeoff*err_sq_energy + (1-rho_tradeoff)*err_sq_forces

    return err_sq

'''
Next, we procede in the same manner as in the QM9 tutorial. 
First, we specify that the Adam optimizer from PyTorch should be used 
to train the model:
'''


# build optimizer
optimizer = Adam(model.parameters(), lr=5e-4)

'''
Then, we construct the trainer hooks to monitor the training process and
 anneal the learning rate. Since we also learn forces in addition to the 
 nergies, we include a corresponding metric into the logger.
'''

# before setting up the trainer, remove previous training checkpoints and logs
#%rm -rf ./forcetut/checkpoints
os.system("rm -rf ./forcetut/checkpoints")
os.system("rm -rf ./forcetut/log.csv")



import schnetpack.train as trn

# set up metrics
metrics = [
    spk.metrics.MeanAbsoluteError(MD17.energy),
    spk.metrics.MeanAbsoluteError(MD17.forces)
]

# construct hooks
hooks = [
    trn.CSVHook(log_path=forcetut, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=5, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    )
]

'''
Finally, we build the SchNetPack Trainer and pass the optimizer,
 loss function, hooks and data loaders.
'''
import sys

class Trainer:
    r"""Class to train a model.
    This contains an internal training loop which takes care of validation and can be
    extended with custom functionality using hooks.
    Args:
       model_path (str): path to the model directory.
       model (torch.Module): model to be trained.
       loss_fn (callable): training loss function.
       optimizer (torch.optim.optimizer.Optimizer): training optimizer.
       train_loader (torch.utils.data.DataLoader): data loader for training set.
       validation_loader (torch.utils.data.DataLoader): data loader for validation set.
       keep_n_checkpoints (int, optional): number of saved checkpoints.
       checkpoint_interval (int, optional): intervals after which checkpoints is saved.
       hooks (list, optional): hooks to customize training process.
       loss_is_normalized (bool, optional): if True, the loss per data point will be
           reported. Otherwise, the accumulated loss is reported.
   """

    def __init__(
        self,
        model_path,
        model,
        loss_fn,
        optimizer,
        train_loader,
        validation_loader,
        keep_n_checkpoints=3,
        checkpoint_interval=10,
        validation_interval=1,
        hooks=[],
        loss_is_normalized=True,
    ):
        self.model_path = model_path
        self.checkpoint_path = os.path.join(self.model_path, "checkpoints")
        self.best_model = os.path.join(self.model_path, "best_model")
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.validation_interval = validation_interval
        self.keep_n_checkpoints = keep_n_checkpoints
        self.hooks = hooks
        self.loss_is_normalized = loss_is_normalized

        self._model = model
        self._stop = False
        self.checkpoint_interval = checkpoint_interval

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if os.path.exists(self.checkpoint_path):
            self.restore_checkpoint()
        else:
            os.makedirs(self.checkpoint_path)
            self.epoch = 0
            self.step = 0
            self.best_loss = float("inf")
            self.store_checkpoint()

    def _check_is_parallel(self):
        return True if isinstance(self._model, torch.nn.DataParallel) else False

    def _load_model_state_dict(self, state_dict):
        if self._check_is_parallel():
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)

    def _optimizer_to(self, device):
        """
        Move the optimizer tensors to device before training.
        Solves restore issue:
        https://github.com/atomistic-machine-learning/schnetpack/issues/126
        https://github.com/pytorch/pytorch/issues/2830
        """
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    @property
    def state_dict(self):
        state_dict = {
            "epoch": self.epoch,
            "step": self.step,
            "best_loss": self.best_loss,
            "optimizer": self.optimizer.state_dict(),
            "hooks": [h.state_dict for h in self.hooks],
        }
        if self._check_is_parallel():
            state_dict["model"] = self._model.module.state_dict()
        else:
            state_dict["model"] = self._model.state_dict()
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        self.best_loss = state_dict["best_loss"]
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._load_model_state_dict(state_dict["model"])

        for h, s in zip(self.hooks, self.state_dict["hooks"]):
            h.state_dict = s

    def store_checkpoint(self):
        chkpt = os.path.join(
            self.checkpoint_path, "checkpoint-" + str(self.epoch) + ".pth.tar"
        )
        torch.save(self.state_dict, chkpt)

        chpts = [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pth.tar")]
        if len(chpts) > self.keep_n_checkpoints:
            chpt_epochs = [int(f.split(".")[0].split("-")[-1]) for f in chpts]
            sidx = np.argsort(chpt_epochs)
            for i in sidx[: -self.keep_n_checkpoints]:
                os.remove(os.path.join(self.checkpoint_path, chpts[i]))

    def restore_checkpoint(self, epoch=None):
        if epoch is None:
            epoch = max(
                [
                    int(f.split(".")[0].split("-")[-1])
                    for f in os.listdir(self.checkpoint_path)
                    if f.startswith("checkpoint")
                ]
            )

        chkpt = os.path.join(
            self.checkpoint_path, "checkpoint-" + str(epoch) + ".pth.tar"
        )
        self.state_dict = torch.load(chkpt)

    def train(self, device, n_epochs=sys.maxsize):
        """Train the model for the given number of epochs on a specified device.
        Args:
            device (torch.torch.Device): device on which training takes place.
            n_epochs (int): number of training epochs.
        Note: Depending on the `hooks`, training can stop earlier than `n_epochs`.
        """
        self._model.to(device)
        self._optimizer_to(device)
        self._stop = False

        for h in self.hooks:
            h.on_train_begin(self)

        try:
            for _ in range(n_epochs):
                # increase number of epochs by 1
                self.epoch += 1

                for h in self.hooks:
                    h.on_epoch_begin(self)

                if self._stop:
                    # decrease self.epoch if training is aborted on epoch begin
                    self.epoch -= 1
                    break

                # perform training epoch
                #                if progress:
                #                    train_iter = tqdm(self.train_loader)
                #                else:
                train_iter = self.train_loader

                for train_batch in train_iter:
                    self.optimizer.zero_grad()

                    for h in self.hooks:
                        h.on_batch_begin(self, train_batch)

                    # move input to gpu, if needed
                    train_batch = {k: v.to(device) for k, v in train_batch.items()}

                    result = self._model(train_batch)
                    #print(result['forces_der'])
                    loss = self.loss_fn(train_batch, result)
                    loss.backward()
                    self.optimizer.step()
                    self.step += 1

                    for h in self.hooks:
                        h.on_batch_end(self, train_batch, result, loss)

                    if self._stop:
                        break

                if self.epoch % self.checkpoint_interval == 0:
                    self.store_checkpoint()

                # validation
                if self.epoch % self.validation_interval == 0 or self._stop:
                    for h in self.hooks:
                        h.on_validation_begin(self)

                    val_loss = 0.0
                    n_val = 0
                    for val_batch in self.validation_loader:
                        # append batch_size
                        vsize = list(val_batch.values())[0].size(0)
                        n_val += vsize

                        for h in self.hooks:
                            h.on_validation_batch_begin(self)

                        # move input to gpu, if needed
                        val_batch = {k: v.to(device) for k, v in val_batch.items()}

                        val_result = self._model(val_batch)
                        #print(val_result['forces_der'])

                        val_batch_loss = (
                            self.loss_fn(val_batch, val_result).data.cpu().numpy()
                        )
                        if self.loss_is_normalized:
                            val_loss += val_batch_loss * vsize
                        else:
                            val_loss += val_batch_loss

                        for h in self.hooks:
                            h.on_validation_batch_end(self, val_batch, val_result)

                    # weighted average over batches
                    if self.loss_is_normalized:
                        val_loss /= n_val

                    if self.best_loss > val_loss:
                        self.best_loss = val_loss
                        torch.save(self._model, self.best_model)
                        #result = self._model(train_batch)
                        #print("meep", result['forces_der'])
                        print("HEy")
                        #print(self._model(inputs))

                    for h in self.hooks:
                        h.on_validation_end(self, val_loss)

                for h in self.hooks:
                    h.on_epoch_end(self)

                if self._stop:
                    break
            #
            # Training Ends
            #
            # run hooks & store checkpoint
            for h in self.hooks:
                h.on_train_ends(self)
            self.store_checkpoint()

        except Exception as e:
            for h in self.hooks:
                h.on_train_failed(self)

            raise e

trainer = Trainer(
    model_path=forcetut,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

'''
We then train our model for 300 epochs, which should take approximately 
10 minutes on a notebook GPU.
'''
# check if a GPU is available and use a CPU otherwise

'Die hessian Berehcnung nimmt zuviels peicher ein, auf cpu umchalten'

'''
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
'''
device = "cpu"

# determine number of epochs and train
n_epochs = 30000
trainer.train(device=device, n_epochs=n_epochs)

'''
for name, param in model.named_parameters():
    print("Hullo",name)
    #print(name, param.shape)
'''

'''
Training will produce several files in the model_path directory, 
which is forcetut in our case. The split is stored in split.npz. 
Checkpoints are written to checkpoints periodically, which can be 
used to restart training. A copy of the best model is stored in best_model,
 which can directly be accessed using the torch.load function. Since we 
 specified the CSV logger, the training progress is saved to log.csv.

Using the CSV file, the training progress can be vizualized. 
An example showing the evolution of the mean absolute errors (MAEs) during 
training is shown below. Besides the schnetpack.train.CSVHook, it is also 
possible to use the schnetpack.train.tensorboardHook. This makes it possible 
to monitor the training in real time with TensorBoard.
'''



# Load logged results
results = np.loadtxt(os.path.join(forcetut, 'log.csv'), skiprows=1, delimiter=',')

# Determine time axis
time = results[:,0]-results[0,0]

# Load the validation MAEs
energy_mae = results[:,4]
forces_mae = results[:,5]

# Get final validation errors
print('Validation MAE:')
print('    energy: {:10.3f} kcal/mol'.format(energy_mae[-1]))
print('    forces: {:10.3f} kcal/mol/\u212B'.format(forces_mae[-1]))

# Construct figure
plt.figure(figsize=(14,5))

# Plot energies
plt.subplot(1,2,1)
plt.plot(time, energy_mae)
plt.title('Energy')
plt.ylabel('MAE [kcal/mol]')
plt.xlabel('Time [s]')

# Plot forces
plt.subplot(1,2,2)
plt.plot(time, forces_mae)
plt.title('Forces')
plt.ylabel('MAE [kcal/mol/\u212B]')
plt.xlabel('Time [s]')

plt.show()



'''

dd_model = torch.load(os.path.join(forcetut, "best_model"))

print(inputs2)

result3 = dd_model(inputs2)

print("woop",result3)
'''

print(datetime.datetime.now())