#
#
from sgdml.predict import GDMLPredict
from sgdml.utils import io
from util_sgdml import *
from bg_session import session
import torch
from torch import nn, optim, distributions
from torch.nn import functional as F
import math
# from torchvision import transforms
# from torchvision.utils import save_image
#-------- directories -------- #


HIDDEN_DIM = 128
N_COUPLE_LAYERS = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


sm = nn.Softmax()

# --- defines the model and the optimizer ---- #
class CouplingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask, t = 1,layer_reg=None):
        super().__init__()
        self.s_fc1 = nn.Linear(input_dim, hid_dim)
        self.s_fc2 = nn.Linear(hid_dim, hid_dim)
        self.s_fc3 = nn.Linear(hid_dim, output_dim)
        self.t_fc1 = nn.Linear(input_dim, hid_dim)
        self.t_fc2 = nn.Linear(hid_dim, hid_dim)
        self.t_fc3 = nn.Linear(hid_dim, output_dim)
        self.mask = mask
        self.t =t
        self.output_dim = output_dim
        self.layer_reg = layer_reg

    def forward(self, x):
        x_m = x * self.mask
        s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x_m))))))/self.t
        t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(x_m)))))
        if self.layer_reg is not None:
            z = self.layer_reg*torch.normal(mean=torch.zeros(t_out.shape), std=torch.ones(t_out.shape)\
                             *np.sqrt(self.t)).to(device)
        else:
            z=1
        y = x_m + (1-self.mask)*(x*torch.exp(s_out)+t_out*z)
        log_det_jacobian = s_out.sum(dim=1)
        return y, log_det_jacobian

    def backward(self, y):

        y_m = y * self.mask
        s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(y_m))))))/self.t
        t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(y_m)))))
        if self.layer_reg is not None:
            z = self.layer_reg*torch.normal(mean=torch.zeros(t_out.shape), std=torch.ones(t_out.shape)\
                             *np.sqrt(self.t)).to(device)
        else:
            z=1
        x = y_m + (1-self.mask)*(y-t_out*z)*torch.exp(-s_out)
        log_det_jacobian = -s_out.sum(dim=1)
        return x,log_det_jacobian

class RealNVP(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask, n_layers = 6,t = 1,layer_reg = None, dropout = None):
        super().__init__()
        assert n_layers >= 2, 'num of coupling layers should be greater or equal to 2'
        self.n_layers = n_layers
        self.modules = []
        self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, mask,t = t,layer_reg=layer_reg))
        for _ in range(n_layers-2):
            mask = 1 - mask
            self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, mask,layer_reg=layer_reg))
        self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, 1 - mask))
        self.module_list = nn.ModuleList(self.modules)
        self.sm = nn.Softmax(dim=0)
        self.energylist = []
        self.layer_reg = layer_reg
        self.dropout = dropout
        self.alpha = nn.Parameter(torch.tensor(1.0e0))
        self.beta = nn.Parameter(torch.tensor(1.0e0))


    def forward(self, x,dropout = None):
        """
        X -> Z
        """
        ldj_sum = 0 # sum of log determinant of jacobian
        for i,module in enumerate(self.module_list):
            x, ldj= module(x)
            if self.dropout is not None:
                x = self.dropout(x)

            ldj_sum += ldj
        return x, ldj_sum

    def backward(self, z,dropout = None):
        """
        Z -> X
        """
        ldj_sum = 0 # sum of log determinant of jacobian
        # if self.z is not None:
        #     z = self.z
        #     self.z = None #reset
        for i,module in enumerate(reversed(self.module_list)):
            z,ldj = module.backward(z)
            if self.dropout is not None:
                z = self.dropout(z)

            ldj_sum += ldj
        return z, ldj_sum
    # def loss_kl(self,)
def sum_noti(e,i):
    return torch.sum(e) - e[i]
def dpdef(e,i,j,t,kB):
    if i ==j:
        e = torch.exp(e)
        z_ni = sum_noti(e,i)
        z = z_ni + e[i]
        top = -z*z_ni
        bottom = torch.pow(z_ni*e[i]+z_ni,2)*t*kB
    elif i !=j:
        top = torch.exp(e[j] + e[i])
        bottom = torch.pow(torch.exp(e).sum(),2)*t*kB
    return top/bottom

def dsdpf(p_i,i):
    return torch.log(p_i[i])+1
def dsdef(e,Temperature,kB):
    energy = e - torch.min(e)
    energy = energy/(Temperature)
    p_i = torch.exp(-energy/kB)
    Z = torch.sum(p_i)
    p_i /=Z
    dSdP = torch.stack([dsdpf(p_i,i) for i in range(len(energy))])
    dPdE = torch.stack([  torch.stack([dpdef(energy, i,j, Temperature,kB) for i in range(len(energy))]) for j in range(len(energy))])
    dSdP[dSdP==float('-Inf')]=-100
    return torch.matmul(dSdP, dPdE)
class PytorchEntropyWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,Temperature,kB):
        g = dsdef(input,Temperature,kB)
        S = entropyf(input, Temperature,sm)
        ctx.save_for_backward(S,g)
        return S
    @staticmethod
    def backward(ctx,grad_output):
        S,g = ctx.saved_tensors
        return g,None,None
PyEntropy = PytorchEntropyWrapper.apply
d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, num,temp, energy_model,device,OUTPUT_DIM,t_sample= '1'):#sample bg ##    z = torch.normal(mean=torch.zeros(OUTPUT_DIM*101), std=torch.ones(OUTPUT_DIM*101)*np.sqrt(t)).reshape(101,OUTPUT_DIM)
    model.eval()
    with torch.no_grad():
        z = torch.normal(mean=torch.zeros(OUTPUT_DIM*num),#*np.sqrt(temp)
                std=torch.ones(OUTPUT_DIM*num)*t_sample).reshape(num,OUTPUT_DIM).to(device)
        x, log_det_j_sum_zx = model.backward(z)
        z_energy = ((0.5/temp)*z**2).sum(1)
        e,_ = energy_model.energy(x.cpu().detach().numpy())
        e = torch.FloatTensor(e)
    return x,e, z_energy, log_det_j_sum_zx
# --- train and test --- #
def jacobian(outputs, inputs, create_graph=False):
    """Computes the jacobian of outputs with respect to inputs

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size() + inputs.size()
                            ).view((-1,) + inputs.size())
    for i, out in enumerate(outputs.view(-1)):
        col_i = torch.autograd.grad(out, inputs, retain_graph=True,
                              create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(outputs.size() + inputs.size())


def z_dynamics_n(t,  model, initial_structure,energy_model,OUTPUT_DIM ,steps = 100,StepSizeGrad = 1.0e-3,
        StepSizeNorm = 1.0e-1,SIM = ['N'],translate_energy = 0,MAX = 1000,sgdml=False,scale = 1):
    accept = 0
    total = 0
    traj = []
    bad = 0
    logRzxL = []
    zL = []
    initial_structure.requires_grad = True
    x0 = initial_structure
    z, logRxz = model(x0)
    while accept < steps:
        zprop = z
        if 'Z' in SIM:
            G_oi = torch.autograd.grad(x.sum(), z, retain_graph=True,allow_unused=True)[0]#output, input
            zprop += -G_oi*StepSizeGrad
        if 'N' in SIM:
            n = torch.normal(mean=torch.zeros(OUTPUT_DIM), std=torch.ones(OUTPUT_DIM)).to(device)#*np.sqrt(t)
            zprop +=  n*StepSizeNorm#np.sqrt(2*t)*
        x1, logRzx = model.backward(zprop)
        if sgdml:
            e1,_ = energy_model.energy(x1.detach().cpu().numpy())[0]
            e0,_ = energy_model.energy(x0.detach().cpu().numpy())[0]
        else:
            e1 = energy_model.energy(x1.detach().cpu().numpy())[0]
            e0 = energy_model.energy(x0.detach().cpu().numpy())[0]
        e1 = torch.FloatTensor(e1)
        e0 = torch.FloatTensor(e0)
#         e = torch.stack((e1,e0))
#         emin = torch.min(e)
#         e1 = e1 - emin
#         e0 = e0-emin
#         MIN = torch.min()
        #accept or not with prob min(1, exp(-deltaE)) --> deltaE = u(f(z)) - u(x) - logRzx + logRxz
        deltaE = -(((e1 )-(e0))/t )[0].data#- logRzx + logRxz
#         print(torch.exp(deltaE).data)
        min = torch.min(torch.FloatTensor([torch.exp(deltaE).data,1]))
        # print(torch.exp(deltaE), e1,'prop', e0,'old')

        rand = torch.rand(1)
        if rand < min:
            z = zprop
            x0 = x1
            traj.append(x1)
            logRzxL.append(logRzx)
            zL.append(zprop)
            accept +=1
        else:
            bad +=1
            continue
        total +=1
        if total > MAX or bad > 5000:
            return torch.stack(traj).permute(1,0,2)
    return torch.stack(traj).permute(1,0,2),torch.cat(logRzxL),torch.cat(zL)

def keep_conforms(e, e2, x,x2, t, logrzx=1, logrzx2=1,device = 'cuda',kB = 8.617333262145e-5):
    """
    e2 is proposal, e is current
    """
    # print(e2, 'e2')
    # print(e, 'e')
    ee = torch.exp(-(e2-e+logrzx-logrzx2)/(t*kB)).reshape(1,-1)
    # print(ee, 'ee')
    ones = torch.ones(ee.shape).to(device)
    cat = torch.cat((ones,ee),dim=0)
    #obtain probability of accepting new conformation
    prob_acc, ind = torch.min(cat,dim=0)
    #prob_acc = prob_acc[ind]
    rand = torch.rand(prob_acc.shape).to(device)

    e_gather = torch.where(rand <=prob_acc,e2, e)
    ind = torch.where(rand <=prob_acc)[0]
    ind_n = torch.where(rand > prob_acc)[0]
    x_gather_new = x2[ind,:]
    x_gather_old = x[ind_n,:]
    x_gather = torch.cat((x_gather_new, x_gather_old))

    return x_gather, e_gather
def train(model,bg,epoch,t,optimizer,PyEnergy,OUTPUT_DIM,explore=1,KL=1.0, ML = 1.0, R =False,S=None
        ,MAXepoch = 2000,energy_model = None,scale = 1,powers= [1, 1,1],P=2000,kB=1,HCF=None,
        entropynn = None,batch_gen=101,Ss=None,device = d,dropout=None,grad_reg=None, layer_reg=None,
        dropout_type=None,out=None, conform = None, t_sample = 't', manual_xtrain = None, conform_kb = 1, StepSize = 1.0e-3):#dropouttype is 'mc' or 'reg'

        #powers = [KL, ML]
    INPUT_DIM = OUTPUT_DIM
    model.train()
    loss = 0

    if manual_xtrain is not None:
        xtrain = manual_xtrain

    else:
        xtrain = torch.FloatTensor(bg.xtrain).to(device)

    if S =='decision':
        if epoch > 1:
            xtrain =bg.conform

        else:
            if manual_xtrain is not None:
                xtrain = manual_xtrain

            else:
                # bg.xtrain = torch.FloatTensor(bg.xtrain).to(device)
                # xtrain =  bg.xtrain
                #xtrain = None
                pass

    #xtrain.requires_grad = True
    train_loss = 0
    optimizer.zero_grad()
    z = torch.normal(mean=torch.zeros(OUTPUT_DIM*batch_gen), std=torch.ones(OUTPUT_DIM*batch_gen)*t_sample\
                ).reshape(batch_gen,OUTPUT_DIM).to(device)


    #reconstruct z distribution
    if xtrain is not None:
        zout, log_det_j_sum_xz = model(xtrain,dropout = dropout)

        # #mcmc -> if commented out, then no mcmc
        #z = zout + StepSize*z
    #reconstruct x distribution
    x, log_det_j_sum_zx = model.backward(z,dropout = dropout)



    #print(G_oi,'g',G_oi.shape)
    if energy_model is None:
        energy = PyEnergy(x,bg.sgdml_model)*scale
        if xtrain is not None:
            energy_c = PyEnergy(xtrain,bg.sgdml_model)


    else:
        energy = PyEnergy(x,energy_model)*scake
        if xtrain is not None:
            energy_c = PyEnergy(xtrain,energy_model)


    MINE = torch.min(energy)
    MINM = torch.max(energy)
    energy = energy - MINE
    loss_KL = 0

#---------------------------------------------------------------------------

    # # #forward
    logdet = log_det_j_sum_zx
    zm_zx= ((0.5/t)*z**2 ).sum(1)
    #zm_zx = zm_zx - torch.log(torch.exp(zm_zx).sum()) #normalize
    pe = energy
    pe_nn =explore*logdet# kB*t*(explore*logdet)# + zm_zx) #p_z = p_x*Rxz^{-1}
    #pe_nn = pe_nn - torch.log(torch.exp(pe_nn).sum()) #
    #loss_KL = pe.mean()#torch.pow((pe + pe_nn),2).mean()
    loss_KL = (energy/(t*kB)  -(model.alpha+logdet)).mean()#explore*logdet
#----------------------------------------------------------------------------80 datapoin
# no retrain
# /home/spike/guest/guest/code/extra/BGenerators/data/gdb7/1000/80/gdb7-m80-i1-c1-d1.json get_json list
# using potential energy
# (101, 45) xtrain shape
# loading sgdml model from /home/spike/guest/guest/code/bg_sgdml/bg/bg_pytorch/trained.80.npz
# using sgdml



    #loss_ML = -(log_det_j_sum_xz-((0.5/t)*zout**2).sum(1)).mean() #+OUTPUT_DIM * np.log(np.sqrt(t))
#---------------------------------------------------------------------------
    #backward
    if xtrain is not None:
        #backwardlog_det_j_sum_zx
        zm_xz = ((0.5/t)*zout**2 ).sum(1)#((0.5/t)*zout**2 ).sum(1)
        pe_nn_xz = (zm_xz )
        #pe_nn_xz = pe_nn_xz - torch.log(torch.exp(pe_nn_xz).sum())

        energyb = -log_det_j_sum_xz#+  (energyb - torch.min(energyb))/(t*kB)
        #energyb = energyb - torch.log(torch.exp(energyb).sum())

        loss_ML = ( model.beta + pe_nn_xz+ energyb  ).mean()  #+OUTPUT_DIM * np.log(np.sqrt(t)) # +   torch.normal(mean=torch.zeros(101), std=torch.ones(101)*np.sqrt(t)          )
        #    loss_ML = (  zm_xz -log_det_j_sum_xz).mean()
    else:
        loss_ML = 0
#---------------------------------------------------------------------------

    loss_KL = torch.abs(loss_KL)
    loss_ML = torch.abs(loss_ML)
    #loss_KL = torch.pow(loss_KL, powers[0])
    #loss_ML = torch.pow(loss_ML, powers[1])
    #### new

    if S == 'decision':
        if epoch == 1:
            bg.conform, bg.econform,bg.logrzx= x.clone().detach(), energy.clone().detach(),log_det_j_sum_zx.detach()#, , bg.logrzx =x, log_det_j_sum_zx.detach()
        else:
            #print(bg.econform, energy, bg.conform, x)
            #bg.conform, bg.econform = keep_conforms(bg.econform, energy, bg.conform, x, t ,kB = conform_kb)#, kB=1)#,\

            bg.conform, bg.econform = keep_conforms(bg.econform, energy, bg.conform, x, t, bg.logrzx,log_det_j_sum_zx ,kB = conform_kb)#, kB=1)#,\
            bg.conform = bg.conform.clone().detach()
            bg.econform = bg.econform.clone().detach()



    loss += loss_ML#*ML
    loss += loss_KL*KL

    loss.backward()
    optimizer.step()

    if epoch % P ==0 :
        if xtrain is not None:
            if HCF is not None:
                #print(energy, logdet)
                HC = HCF(energy.detach()/scale,t,kB, pr_i = logdet, device = 'cuda' )
            else:
                HC = analysis.CalcHeatCapacity( energy.detach().numpy()/scale,0, 0, log_det_j_sum_xz.detach().numpy(),
                                      T=t,kB=kB)[1][0] #sgdml is trained @eV

            out = [epoch,energy.mean().item(), loss_KL, loss_ML,[MINE.item()/scale,MINM.item()/scale],(HC),entropyRzx(log_det_j_sum_zx),logdet.mean(), entropyf(energy-torch.min(energy),t,sm),[]]

    if out is None:
        return None#,bg.conform
    else:
        return out#, bg.conform

def test(epoch):
    model.eval()
    test_loss = 0
    x_all = np.array([[]]).reshape(0,2)
    z_all = np.array([[]]).reshape(0,2)
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            z, log_det_j_sum = model(data)
            cur_loss = -(prior_z.log_prob(z)+log_det_j_sum).mean().item()
            test_loss += cur_loss
            x_all = np.concatenate((x_all,data.numpy()))
            z_all = np.concatenate((z_all,z.numpy()))

        subfig_plot(1, x_all, -2, 3, -1, 1.5,'Input: x ~ p(x)', 'b')
        subfig_plot(2, z_all, -3, 3, -3,3,'Output: z = f(x)', 'b')

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

def SampleAndHC(model, energy_model, num = 1000,T = 1 ):
    #SampleAndHC(model,bg.sgdml_model, num = 1000,T = 1 )
    x,e, z_energy, logRzx = sample(model, num, T, energy_model)
    HC = analysis.CalcHeatCapacity(e.data.numpy(), x.data.numpy(),
     z_energy.data.numpy(), logRzx.data.numpy(), Temperature ,kB = 1,translate_energy = 5348)
    return HC,e
def HC(x,e,T=1):
    #SampleAndHC(model,bg.sgdml_model, num = 1000,T = 1 )
    HC = analysis.CalcHeatCapacity(e.data.numpy(), x.data.numpy(), etraj_z=0,logRzx=0,T=T ,kB = 1,translate_energy = 5348)
    return HC

def z_dynamics(t, model, initial_structure,energy_model ,steps = 100,StepSizeGrad = 1.0e-3,
        StepSizeNorm = 1.0e-1,SIM = ['N'],translate_energy = 5348,MAX = 1000,sgdml=False,scale = 1.0e-3):
    accept = 0
    total = 0
    traj = []
    initial_structure.requires_grad = True
    x0 = initial_structure
    z, logRxz = model(x0)
    while accept < steps:
        zprop = z
        if 'Z' in SIM:
            G_oi = torch.autograd.grad(x.sum(), z, retain_graph=True,allow_unused=True)[0]#output, input
            zprop += -G_oi*StepSizeGrad
        if 'N' in SIM:
            n = torch.normal(mean=torch.zeros(OUTPUT_DIM), std=torch.ones(OUTPUT_DIM))#*np.sqrt(t)
            zprop +=  n*StepSizeNorm#np.sqrt(2*t)*
        x1, logRzx = model.backward(zprop)
        if sgdml:
            e1,_ = energy_model.energy(x1.detach().numpy())
            e0,_ = energy_model.energy(x0.detach().numpy())
        else:
            e1 = energy_model.energy(x1.detach().numpy())
            e0 = energy_model.energy(x0.detach().numpy())
        e1 = torch.FloatTensor(e1)
        e0 = torch.FloatTensor(e0)
        MIN = torch.min()
        #accept or not with prob min(1, exp(-deltaE)) --> deltaE = u(f(z)) - u(x) - logRzx + logRxz
        deltaE = -(((e1 )-(e0))/t )[0].data#- logRzx + logRxz
        min = torch.min(torch.FloatTensor([torch.exp(deltaE).data,1]))
        print(torch.exp(deltaE), e1,'prop', e0,'old')

        rand = torch.rand(1)
        if rand < min:
            z = zprop
            x0 = x1
            traj.append(x1)
            accept +=1
        else:
            continue
        total +=1
        if total > MAX:
            return torch.stack(traj).permute(1,0,2)
    return torch.stack(traj).permute(1,0,2)
def OutputTraj2txt(tensor, outfile,torch=False):
    f = open(outfile,'w')
    for i in tensor:
        for j in i:
            for k in j:
                if torch:
                    f.write('%s '%k.detach().numpy().tolist())
                else:
                    f.write('%s '%k)
            f.write('\n')
    f.close()
    print('output file to ',outfile)
    return None
def sample_openmm(model, num,temp, energy_model):
    model.eval()
    with torch.no_grad():
        z = torch.normal(mean=torch.zeros(OUTPUT_DIM*num),
                std=torch.ones(OUTPUT_DIM*num)*np.sqrt(temp)).reshape(num,OUTPUT_DIM)
        z_energy = ((0.5/temp)*z**2).sum(1)
        x, log_det_j_sum_zx = model.backward(z)
        e= energy_model.energy(x.detach().numpy())
        e = torch.FloatTensor(e)
    return x,e, z_energy, log_det_j_sum_zx #x may suppose to be x*10

def sample_mdSgdml(initial_structure, temperature, energy_model,numsteps=1000,deltaT = 1.0e0,zscale = 1.0e-5):
    trajL = []
    energyL = []
    x = torch.FloatTensor(initial_structure)
    px = x
    for i in range(numsteps):
        z = zscale*torch.normal(mean=torch.zeros(OUTPUT_DIM), std=torch.ones(OUTPUT_DIM)*np.sqrt(temperature)).reshape(OUTPUT_DIM)

        energy, force = energy_model.energy(x)
        energy = torch.FloatTensor(energy)
        force = torch.FloatTensor(force)
        tmp_x = x
        x = 2*x - px + force*deltaT**2#+z#/m
        #x = x - force*(deltaT**2)#+z#/m
        px = tmp_x
        trajL.append(x)
        energyL.append(energy)
    return torch.stack(trajL), torch.stack(energyL)
def sample_MCMCsGDML(t, initial_structure,energy_model ,steps = 100
        ,StepSize = 1.0e-1,translate_energy = 5348,MAX = 1000):
    trajL = []
    energyL = []
    x = torch.FloatTensor(initial_structure)
    accept = 0
    total = 0
    while accept < steps:

        z = StepSize*torch.normal(mean=torch.zeros(OUTPUT_DIM),
                        std=torch.ones(OUTPUT_DIM)*np.sqrt(t)).reshape(OUTPUT_DIM)
        e,_ = energy_model.energy(x)
        xtmp =x+ z
        etmp,_ = energy_model.energy(xtmp)
        e = e+ translate_energy
        etmp = etmp+ translate_energy
        #accept or not with prob min(1, exp(-deltaE)) --> deltaE = u(f(z)) - u(x) - logRzx + logRxz
        deltaE = -(((etmp )-(e))/t )#- logRzx + logRxz
        min = np.min([np.exp(deltaE),1])
        #print(torch.exp(deltaE), e1,'prop', e0,'old')

        rand = np.random.rand(1)
        if rand < min:
            x = xtmp
            trajL.append(x)
            energyL.append(etmp)
            accept +=1
        else:
            continue
        total +=1
        if total > MAX:
            return np.stack(trajL).permute(1,0,2), np.stack(energyL)
    return np.stack(trajL), np.stack(energyL)
# def train_openmm(epoch,t,explore=1,KL=True, ML = True,MAXepoch = 2000,energy_model):#logw = -energy_x + energy_z + Jzx
#
#     return

def InitializePytorchBG( t = 1,translate = 0,datapoint=None,smi_directory=None,database = None,\
        home=None,sgdml_directory=None, directory=None,test_npz = None,energy = 'total', retrain = True):
    datapoint = directory
    print(datapoint,'datapoin')
    class PytorchEnergyWrapper(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input,sgdml_model):
            e,f = sgdml_model.energy(input.cpu().detach().numpy())
            e,g = torch.FloatTensor(e).to(device), torch.FloatTensor(-f).to(device)
            ctx.save_for_backward(e,g)
            return e #+ translate
        @staticmethod
        def backward(ctx,grad_output):
            e,g = ctx.saved_tensors
            return g, None
    PyEnergy = PytorchEnergyWrapper.apply
    bg= session("BG",retain_data = True, smi_directory=smi_directory,
              sgdml_trained_directory=None,directory_database=database+str(directory),
              sgdml_directory = sgdml_directory, home = home)
    bg.CreatePytorchBG(retrain_sgdml=retrain, lr=1.0e-3, e=500,split_data=[0,800],numlayers=3,nl_hidden=3 ,nh_dim=128,
                      temperature = t,explore = 0, weight_ML=1,weight_KL = 0, weight_new =0.0e0,
                      use_sgdml=True, sgdml_input = home+'trained.%s.npz'%datapoint,std=t ,datapoint=datapoint,test_npz = test_npz,energy=energy)
    return bg, PyEnergy
def InitializePytorchBG_openmm( t = 1,translate = 0):
    """
        def _get_energy_from_openmm_state(self, state):
            energy_quantity = state.getPotentialEnergy()
            return self._reduce_units(energy_quantity)

        def _get_gradient_from_openmm_state(self, state):
            forces_quantity = state.getForces(asNumpy=True)
            return -1. * np.ravel(self._reduce_units(forces_quantity) * self._length_scale)

    """
    bg= session("BG",retain_data = True, smi_directory=smi_dir,
              sgdml_trained_directory=None,directory_database=database+str(directory),
              sgdml_directory = home + 'sGDML/', home = home)
    bg.CreatePytorchBG(retrain_sgdml=False, lr=1.0e-3, e=500,split_data=[0,400],numlayers=3,nl_hidden=3 ,nh_dim=128,
                      temperature = t,explore = 0, weight_ML=1,weight_KL = 0, weight_new =0.0e0,
                      use_sgdml=False, sgdml_input = home+'trained.npz',energy = 'potential',std=t,datapoint = datapoint)
    class PytorchEnergyWrapper(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input,energy_model):
            e,f = energy_model.pytorchEnergy(input.cpu().detach().numpy())#pytorchEnergy
            e,g = torch.FloatTensor(e).to(device), torch.FloatTensor(-f).to(device)
            ctx.save_for_backward(e,g)
            return e
        @staticmethod
        def backward(ctx,grad_output):
            e,g = ctx.saved_tensors
            return g, None
    PyEnergy = PytorchEnergyWrapper.apply

    return bg, PyEnergy
def train_openmm(epoch,t,explore=1,KL=True, ML = True,MAXepoch = 2000,energy_model = None):#logw = -energy_x + energy_z + Jzx
    model.train()
    xtrain = torch.FloatTensor(bg.xtrain)
    #xtrain.requires_grad = True
    train_loss = 0
    optimizer.zero_grad()
    z = torch.normal(mean=torch.zeros(OUTPUT_DIM*101), std=torch.ones(OUTPUT_DIM*101)*np.sqrt(t)).reshape(101,OUTPUT_DIM)
    #z.requires_grad = True
    zout, log_det_j_sum_xz = model(xtrain)
    x, log_det_j_sum_zx = model.backward(z)
    #G_oi = torch.autograd.grad(zout.sum(), xtrain, retain_graph=True,allow_unused=True)[0]
    #G_oi = torch.autograd.grad(x.sum(), z, retain_graph=True,allow_unused=True)[0]
    #print(G_oi,'g',G_oi.shape)
    energy = PyEnergy(x,energy_model)
    MINE = torch.min(energy)
    MINM = torch.max(energy)

    logdet = log_det_j_sum_zx
    loss_KL = (energy/t - explore*logdet).mean()
    loss_ML = -(log_det_j_sum_xz-((0.5/t)*zout**2).sum(1)).mean() #+OUTPUT_DIM * np.log(np.sqrt(t))
    # tf = ((epoch+1)/MAXepoch)*t*2
    # if tf > t:
    #     tf = t
    #loss_var = torch.pow(torch.var(energy)/torch.var(log_det_j_sum_zx) -tf ,2)/45
    #loss_var = torch.abs(torch.var(energy)/torch.var(log_det_j_sum_zx) -t)

    loss = 0
    if ML:
        loss += loss_ML
    if KL:
        loss += loss_KL
    # loss += loss_var
    loss.backward()
    optimizer.step()
    #print(energy.data.mean(), loss_KL.data, loss_ML.data,[MINE.data,MINM.data])
    if energy.data.mean() < 8:
        return
# --- etc. functions --- #
















#------------train with grad wrt input -----------------------#
def train_grad(model,bg,epoch,t,optimizer,PyEnergy,explore=1,KL=True, ML = True, R =True,S=True
        ,MAXepoch = 2000,energy_model = None,scale = 1,kB=1, HCF=None, P = 500):#logw = -energy_x + energy_z + Jzx
    model.train()
    xtrain = torch.FloatTensor(bg.xtrain)
    #xtrain.requires_grad = True
    train_loss = 0
    optimizer.zero_grad()
    z = torch.normal(mean=torch.zeros(OUTPUT_DIM*101), std=torch.ones(OUTPUT_DIM*101)*np.sqrt(t)).reshape(101,OUTPUT_DIM)
    #z.requires_grad = True
    zout, log_det_j_sum_xz = model(xtrain)
    x, log_det_j_sum_zx = model.backward(z)
    #G_oi = torch.autograd.grad(zout.sum(), xtrain, retain_graph=True,allow_unused=True)[0]
    #G_oi = torch.autograd.grad(x.sum(), z, retain_graph=True,allow_unused=True)[0]
    #print(G_oi,'g',G_oi.shape)
    if energy_model is None:
        energy = PyEnergy(x,bg.sgdml_model)
    else:
        energy = PyEnergy(x,energy_model)
        #energy *=scale
    energy = energy*scale
    MINE = torch.min(energy)
    MINM = torch.max(energy)

    logdet = log_det_j_sum_zx
    loss_KL = (energy/t - explore*logdet).mean()
    loss_ML = -(log_det_j_sum_xz-((0.5/t)*zout**2).sum(1)).mean() #+OUTPUT_DIM * np.log(np.sqrt(t))

    # tf = ((epoch+1)/MAXepoch)*t*2
    # if tf > t:
    #     tf = t
    #loss_var = torch.pow(torch.var(energy)/torch.var(log_det_j_sum_zx) -tf ,2)/45
    #loss_var = torch.abs(torch.var(energy)/torch.var(log_det_j_sum_zx) -t)

    loss = 0
    if ML:
        loss += loss_ML
    if KL:
        loss += loss_KL
    if R:
        loss_R = torch.abs(torch.var(logdet) -torch.var(energy)/t)
        loss += loss_R

    if S == 'S1':
        #dlogRzx/dz
        z_ = z.detach()
        z_.requires_grad = True
        x_,logzx_ = model.backward(z_)
        #print(z_.shape, logzx_.shape)
        dzds, =torch.autograd.grad(logzx_.sum(), z_,only_inputs=True, create_graph=True, retain_graph=True)
        #dx/dz
        dxdz, = torch.autograd.grad(x_.sum(), z_,only_inputs=True, create_graph=True, retain_graph=True)
        #dE/dx = -F
        x__ = x.detach()
        x__.requires_grad = True
        z__, logxz__ = model.forward(x__)
        dsdx, = torch.autograd.grad(logxz__.sum(),x__,only_inputs=True, create_graph=True, retain_graph=True)
        if energy_model is None:
            energy = PyEnergy(x_,bg.sgdml_model)
        else:
            energy = PyEnergy(x_,energy_model)

        dedx, = torch.autograd.grad(energy.sum(), x_,only_inputs=True, create_graph=True, retain_graph=False)
        #*(3 above) = Temperature
        loss_S = torch.abs(dedx*(1/dsdx)-torch.sqrt(torch.FloatTensor([t]))).mean()
        loss +=loss_S*1.0e-1
    if S == 'S2':
        #dlogRzx/dz
        z_ = z.detach()
        z_.requires_grad = True
        x_,logzx_ = model.backward(z_)
        #print(z_.shape, logzx_.shape)
        dzds, =torch.autograd.grad(logzx_.sum(), z_,only_inputs=True, create_graph=True, retain_graph=True)
        #dx/dz
        dxdz, = torch.autograd.grad(x_.sum(), z_,only_inputs=True, create_graph=True, retain_graph=True)
        #dE/dx = -F
        x__ = x.detach()
        x__.requires_grad = True
        z__, logxz__ = model.forward(x__)
        dsdx, = torch.autograd.grad(logxz__.sum(),x__,only_inputs=True, create_graph=True, retain_graph=False)
        if energy_model is None:
            energy = PyEnergy(x_,bg.sgdml_model)
        else:
            energy = PyEnergy(x_,energy_model)
        sm = nn.Softmax()
        emin = torch.min(energy)
        p = sm((energy-emin)/t)
        S = (p*torch.log(p)).sum()
        dsde, = torch.autograd.grad(S, energy,only_inputs=True, create_graph=True, retain_graph=True)
        #*(3 above) = Temperature
        dsde = dsde.mean()
        loss_S = torch.abs((1/dsde-t))
        loss +=loss_S
    if S == 'S3':
        #dlogRzx/dz
        z_ = z.detach()
        z_.requires_grad = True
        x_,logzx_ = model.backward(z_)
        #print(z_.shape, logzx_.shape)
        # dzds, =torch.autograd.grad(logzx_.sum(), z_,only_inputs=True, create_graph=True, retain_graph=True)
        #dx/dz
        #dE/dx = -F
        x__ = x.detach()
        x__.requires_grad = True
        z__, logxz__ = model.forward(x__)

        dsdx, = torch.autograd.grad(logxz__.mean(),x__,only_inputs=True, create_graph=True, retain_graph=True)
        if energy_model is None:
            energy = PyEnergy(x_,bg.sgdml_model)
        else:
            energy = PyEnergy(x_,energy_model)

        dedx, = torch.autograd.grad(energy.mean(), x_,only_inputs=True, create_graph=True, retain_graph=True)
        dxde = 1/dedx
        #*(3 above) = Temperature
        loss_S = torch.abs((dsdx*dxde)-t).mean()
        loss +=loss_S
    if S == 'S4':
        x__ = x.detach()
        x__.requires_grad = True
        z__, logxz__ = model.forward(x__)

        dsdx, = torch.autograd.grad(logxz__.mean(),x__,only_inputs=True, create_graph=True, retain_graph=True)
        if energy_model is None:
            energy = PyEnergy(x_,bg.sgdml_model)
        else:
            energy = PyEnergy(x_,energy_model)

        dedx, = torch.autograd.grad(energy.mean(), x_,only_inputs=True, create_graph=True, retain_graph=True)
        dxde = 1/dedx
        #*(3 above) = Temperature
        loss_S = torch.abs((dsdx*dxde)-t).mean()
        loss +=loss_S
    # loss += loss_var
    loss.backward()
    optimizer.step()
    if HCF is not None:
        HC = HCF(energy.detach(),t,kB)
    if epoch % P == 0:
        print(energy.data.mean().item(), loss_KL.item(), loss_ML.item(),loss_S.item(),[MINE.data.item(),MINM.item()],HC)
    if energy.data.mean() < 8:
        return
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))
def File2Movie(F,scale = 1, pdbtemplate='80.pdb2',sgdml=True,skip = 10,bg=False,n_atoms = 15):#bg not used anymore

    f = open(F)
    x = f.readlines()
    x = np.array([i.split() for i in x]).astype('double').reshape(-1,n_atoms,3)
    traj = pt.load(pdbtemplate)
    f0 = traj[0]
    scale = np.sqrt((x**2).sum())/(45*16)*scale
    if sgdml == False:
        for i in range(0,x.shape[0],skip):
            frame = pt.Frame()
            frame.append_xyz(x[i]/scale)
            traj.append(frame)
    else:#permute indexes to match pdb
        permute = [4,2,3,1,0,12,13,14,8,9,10,11,7,6,5]
        for i in range(0,x.shape[0],skip):
            frame = pt.Frame()

            px =x[i][permute]
            frame.append_xyz(px/scale)
            traj.append(frame)
    # # perform superimpose to 1st frame to remove translation and rotation
    # traj.superpose(mask='@CA', ref=0)
    view = nv.show_pytraj(traj)
    return view, int(x.shape[0]/skip)

def TransformEnergy(energy,MIN = False, VAR = False,STD=False):
    print(energy.dtype)
    if energy.dtype == 'float32' or energy.dtype == 'float64':#numpy float
        default='numpy'
    #print(energy.dtype)
    elif energy.dtype == torch.float32 or energy.dtype == torch.float64: #torch float
        energy = energy.detach().numpy()
        default = 'torch'
    if MIN == True:
        energy -= np.min(energy)
    if VAR == True:
        energy /= np.var(energy)
    if STD == True:
        energy/=np.std(energy)
    if default == 'torch':
        energy = torch.FloatTensor(energy)
    return energy

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return int(math.ceil(n * multiplier) / multiplier)
#------------train with grad wrt input -----------------------#
import warnings
warnings.filterwarnings('ignore')
def hcf(e,t,kB):
    return str((torch.var(e)/(kB*t**2)))+'eV/K'
sm = nn.Softmax()

def prob_i(distance, x,weights= None,threshold = 1):

    """
    batch = 100
    L = 15
    x = torch.rand(batch,L,3)
    y = torch.cdist(x,x)

    prob_i(y, y,threshold=1)
    """
    sum_w = len(distance)
    if weights is not None:
        sum_w =weights.sum()
    ii = []
    for k in range(len(x)):
        obs =0
        for p in range(len(distance)):
            dij = torch.sqrt((x[k]-distance[p]).sum()**2)
            if dij < threshold:
                if weights is None:
                    obs += 1
                else:
                    obs +=w[p]
        ii.append(obs/sum_w)
    ii = torch.FloatTensor(ii)
    sii = ii.sum()
    return ii/sii#,obs,c,sum_w
def hcf_w(e,t,kB,z=None, r=None,sm=sm,pr_i = None,units = 'eV',ne=False,device = 'cuda'): #default cuda
    MINe = torch.min(e)
    e = e-MINe
    E = -e/(t)

    if r is not None:
        r = r-torch.min(r)
        E +=r
    if z is not None:
        z = z -torch.min(z)
        E += z
    if ne is False:
        E = torch.exp(E)
    else:
        E = torch.ones(E.shape).to(device)
    if pr_i is not None:

        #print(pr_i, 'pr_i')
        pr_i = torch.exp(-e/(t))
        #print(pr_i, 'pr_i2')
        pr_i = pr_i/pr_i.sum()
        E = pr_i
    else:
        esum = E.sum()
        E = E/esum
    mean = torch.mean(e)
    e = ((((e-mean)**2)*E)).sum()#/esum #sum because w is normalized to 1
    if units == 'eV/K':
        return str((e/(kB*(t**2))))+"eV/K (W)"
    elif units == 'eV':
        return str((e/(kB*(t**2)))*t)+"eV (W)"

def hcf_wa(e,t,kb,z=None,r=None,sm=sm,pr_i=None,units='eV',device='cpu'):
    hc = []
    print( '------------------------------------------------------------------------------------------------------')
    hc.append([hcf_w(e,t,kb,pr_i=None,units=units,ne = True,device=device),'-'])

    hc.append([hcf_w(e,t,kb,pr_i=pr_i,units=units,ne = True,device=device),'p_i'])
    # hc.append([hcf_w(e,t,kb, pr_i=pr_i,units=units,device=device),'w'])
    # hc.append([hcf_w(e,t,kb,z=z,pr_i=pr_i,units=units,device=device),'wz'])
    # hc.append([hcf_w(e,t,kb,r=r,pr_i=pr_i,units=units,device=device),'wr'])
    # hc.append([hcf_w(e,t,kb,z=z,r=r,pr_i=pr_i,units=units,device=device),'wrz'])
    return hc

# # HC = hcf(e,Temperature,kB)
def entropyf(e, t,sm,kB = 8.617333262145e-5):
    e = e-torch.min(e)
    p = sm(-(e/(kB*t)))
    ent = -p*torch.log(p)
    ent[ent!=ent]=0
    out = torch.sum(ent)# # HCw = hcf_w(e,Temperature,kB)
    return out
def calc_lr2(lr, decay):
    return lr/(1 + decay)
