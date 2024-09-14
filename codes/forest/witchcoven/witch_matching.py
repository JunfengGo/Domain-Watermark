"""Main class, holding information about models and training/testing routines."""

import random
import torch
import random
import torchvision
from PIL import Image
from ..consts import BENCHMARK, NON_BLOCKING
from ..utils import bypass_last_layer
from forest.data import datasets
torch.backends.cudnn.benchmark = BENCHMARK
from .witch_base import _Witch

class WitchGradientMatching(_Witch):
    """Brew passenger poison with given arguments.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _define_objective(self, inputs, labels, criterion, sources, target_classes, true_classes):
        """Implement the closure here."""
        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            differentiable_params = [p for p in model.parameters() if p.requires_grad]
            outputs = model(inputs)

            poison_loss = criterion(outputs, labels)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True)

            passenger_loss = self._passenger_loss(poison_grad, source_grad, source_clean_grad, source_gnorm)
            # if self.args.centreg != 0:
            #     passenger_loss = passenger_loss + self.args.centreg * poison_loss
            passenger_loss.backward(retain_graph=self.retain)
            
            return passenger_loss.detach().cpu(), prediction.detach().cpu()
        
        return closure

    def _passenger_loss(self, poison_grad, source_grad, source_clean_grad, source_gnorm):
        """Compute the blind passenger loss term."""
        passenger_loss = 0
        poison_norm = 0


        SIM_TYPE = ['similarity', 'similarity-narrow', 'top5-similarity', 'top10-similarity', 'top20-similarity']
        # if self.args.loss == 'top10-similarity':
        #     _, indices = torch.topk(torch.stack([p.norm() for p in source_grad], dim=0), 10)
        # elif self.args.loss == 'top20-similarity':
        #     _, indices = torch.topk(torch.stack([p.norm() for p in source_grad], dim=0), 20)
        # elif self.args.loss == 'top5-similarity':
        #     _, indices = torch.topk(torch.stack([p.norm() for p in source_grad], dim=0), 5)
        # else:
        if self.args.loss == 'similarity':

            indices = torch.arange(len(source_grad))
        
        else:
            print("GGGGGGGGGGGGGGGGGGGGG")

        for i in indices:
            if self.args.loss in ['scalar_product', *SIM_TYPE]:
                passenger_loss -= (source_grad[i] * poison_grad[i]).sum()
      
            if self.args.loss in SIM_TYPE or self.args.normreg != 0:
                
                poison_norm += poison_grad[i].pow(2).sum()
             

        passenger_loss = passenger_loss / source_gnorm  # this is a constant

        if self.args.loss in SIM_TYPE:
            passenger_loss = 1 + passenger_loss / poison_norm.sqrt()


        return passenger_loss
    
    def _create_patch(self, patch_shape):
        temp_patch = 0.5 * torch.ones(3, patch_shape[1], patch_shape[2])
        patch = torch.bernoulli(temp_patch)
        return patch
        
    def patch_sources(self, kettle):

        if self.args.load_patch == '':

            patch = self._create_patch([3, int(self.args.patch_size), int(self.args.patch_size)])
        
        else:
            patch = Image.open(self.args.load_patch)
            totensor = torchvision.transforms.ToTensor()
            resize = torchvision.transforms.Resize(int(self.args.patch_size))
            patch = totensor(resize(patch))

        patch = (patch.to(**self.setup) - kettle.dm) / kettle.ds
        self.patch = patch.squeeze(0)

        

        # Add patch to sourceset
        if self.args.random_patch:
            print("Add patches to the source images randomely ...")
        else:
            print("Add patches to the source images on the bottom right ...")


        # for idx, (source_img, label, image_id) in enumerate(kettle.sourceset):
        #     source_img = source_img.to(**self.setup)

        #     # if self.args.random_patch:
        #     #     patch_x = random.randrange(0,source_img.shape[1] - self.patch.shape[1] + 1)
        #     #     patch_y = random.randrange(0,source_img.shape[2] - self.patch.shape[2] + 1)
        #     # else:
        #     #     patch_x = source_img.shape[1] - self.patch.shape[1]
        #     #     patch_y = source_img.shape[2] - self.patch.shape[2]

            # delta_slice = torch.zeros_like(source_img).squeeze(0)
            # diff_patch = self.patch - source_img[:, patch_x: patch_x + self.patch.shape[1], patch_y: patch_y + self.patch.shape[2]]
            # delta_slice[:, patch_x: patch_x + self.patch.shape[1], patch_y: patch_y + self.patch.shape[2]] = diff_patch
            # source_delta.append(delta_slice.cpu())

        sourceset = kettle.sourceset
        indices_val = sourceset.indices
        print("SAVE")
        torch.save(indices_val,"indices.pth")
      
        dw_val = [self.dw_val[index] for index in indices_val]

        kettle.sourceset = datasets.DWset(kettle.sourceset, dw_val)
        # kettle.sourceset = datasets.Deltaset(kettle.sourceset, source_delta)
        #torch.save(kettle.sourceset,"stl.pth")
        # Add patch to sourceset
        if self.args.random_patch:
            print("Add patches to the source train images randomely ...")
        else:
            print("Add patches to the source train images on the bottom right ...")
        
        source_trainset = kettle.source_trainset
        indices_train = source_trainset.indices
        #print(indices_train)
        dw_train = [self.dw_train[index] for index in indices_train]
 
        kettle.source_trainset = datasets.DWset(kettle.source_trainset, dw_train)
        #print(len(kettle.source_trainset))
        dw_train_negative = [[self.dw_train_negative[0][index] for index in indices_train],[self.dw_train_negative[1][index] for index in indices_train],[self.dw_train_negative[2][index] for index in indices_train]]

        kettle.source_trainset_negative = datasets.DWset2(kettle.source_trainset, dw_train_negative)

        dw_val_negative = [self.dw_val_negative[index] for index in indices_val]
        kettle.sourceset_negative = datasets.DWset(kettle.sourceset, dw_val_negative)
