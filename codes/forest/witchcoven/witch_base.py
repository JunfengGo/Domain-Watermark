"""Main class, holding information about models and training/testing routines."""

import torch
import warnings

from ..utils import cw_loss
from ..consts import NON_BLOCKING, BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from ..victims.victim_single import _VictimSingle
from ..victims.batched_attacks import construct_attack
from ..victims.training import _split_data

FINETUNING_LR_DROP = 0.001


class _Witch():
    """Brew poison with given arguments.

    Base class.

    This class implements _brew(), which is the main loop for iterative poisoning.
    New iterative poisoning methods overwrite the _define_objective method.

    Noniterative poison methods overwrite the _brew() method itself.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize a model with given specs..."""
        self.args, self.setup = args, setup
        self.retain = True if self.args.ensemble > 1 and self.args.local_rank is None else False
        self.stat_optimal_loss = None
        
        if self.args.dataset == 'TinyImageNet':

            self.dw_val = torch.load("path/tensor_val_1.pt")
            self.dw_train = torch.load("path/tensor_train_1.pt")
            self.dw_val_negative = torch.load("path/tensor_val_2.pt")
            self.dw_train_negative = [torch.load("path/tensor_train_3.pt"),torch.load("path/tensor_train_4.pt"),torch.load("path/tensor_train_5.pt")]
        
        elif self.args.dataset == 'STL':

            self.dw_val = torch.load("path/tensor_val_2_x.pt")
            self.dw_train = torch.load("path/tensor_train_2_x.pt")
            self.dw_val_negative = torch.load("path/tensor_val_3_x.pt")
            self.dw_train_negative = [torch.load("path/tensor_train_1_x.pt"),torch.load("path/tensor_train_4_x.pt"),torch.load("path/tensor_train_5_x.pt")]
        
        elif self.args.dataset == 'CIFAR10':
            self.dw_val = torch.load("path/tensor_val.pt")
            self.dw_train = torch.load("path/tensor_train.pt")
            self.dw_val_negative = torch.load("path/tensor_val_3.pt")
            self.dw_train_negative = [torch.load("path/tensor_train_5.pt"),torch.load("path/tensor_train_6.pt"),torch.load("path/negative_train.pt")]
    
        
        else:
            


            raise ValueError('No Watermark')
            
    """ BREWING RECIPES """

    def brew(self, victim, kettle):
        """Recipe interface."""
        if len(kettle.poisonset) > 0:
            if len(kettle.sourceset) > 0:
                if self.args.eps > 0:
                    if self.args.budget > 0:
                        poison_delta = self._brew(victim, kettle)
                    else:
                        poison_delta = kettle.initialize_poison(initializer='zero')
                        warnings.warn('No poison budget given. Nothing can be poisoned.')
                else:
                    poison_delta = kettle.initialize_poison(initializer='zero')
                    warnings.warn('Perturbation interval is empty. Nothing can be poisoned.')
            else:
                poison_delta = kettle.initialize_poison(initializer='zero')
                warnings.warn('Source set is empty. Nothing can be poisoned.')
        else:
            poison_delta = kettle.initialize_poison(initializer='zero')
            warnings.warn('Poison set is empty. Nothing can be poisoned.')

        return poison_delta


    def _brew(self, victim, kettle):
        """Run generalized iterative routine."""
        print(f'Starting cradting poisons ...')
        self._initialize_brew(victim, kettle)
        poisons, scores = [], torch.ones(self.args.restarts) * 10_000
        # restarts = 1
        for trial in range(self.args.restarts):
            poison_delta, source_losses = self._run_trial(victim, kettle)
            scores[trial] = source_losses
            poisons.append(poison_delta.detach())
            if self.args.dryrun:
                break

        optimal_score = torch.argmin(scores)
        self.stat_optimal_loss = scores[optimal_score].item()
        print(f'Poisons with minimal source loss {self.stat_optimal_loss:6.4e} selected.')
        poison_delta = poisons[optimal_score]

        return poison_delta


    def _initialize_brew(self, victim, kettle):
        """Implement common initialization operations for brewing."""
        victim.eval(dropout=True)
        # Compute source gradients
        self.sources = torch.stack([data[0] for data in kettle.sourceset], dim=0).to(**self.setup)
        self.target_classes = torch.tensor(kettle.poison_setup['target_class']).to(device=self.setup['device'], dtype=torch.long)
        self.true_classes = torch.tensor([data[1] for data in kettle.sourceset]).to(device=self.setup['device'], dtype=torch.long)
     
        self.sources_train = torch.stack([data[0] for data in kettle.source_trainset], dim=0).to(**self.setup)
        
        self.sources_train_target_classes = torch.tensor([kettle.poison_setup['target_class'][0]] * kettle.source_train_num).to(device=self.setup['device'], dtype=torch.long)
        self.sources_train_true_classes = torch.tensor([data[1] for data in kettle.source_trainset]).to(device=self.setup['device'], dtype=torch.long)


        # I add negative set here 

        self.sources_train_negative = [torch.stack([data[0] for data in kettle.source_trainset_negative], dim=0).to(**self.setup),torch.stack([data[1] for data in kettle.source_trainset_negative], dim=0).to(**self.setup),torch.stack([data[2] for data in kettle.source_trainset_negative], dim=0).to(**self.setup)]
        
        self.sources_val_negative = torch.stack([data[0] for data in kettle.sourceset_negative], dim=0).to(**self.setup)

        # Modify source grad for backdoor poisoning
        if self.args.backdoor_poisoning:
            _sources = self.sources_train
            _sources_negative = self.sources_train_negative
            _true_classes= self.sources_train_true_classes
            _target_classes = self.sources_train_target_classes
        else:
            print("XDDDDD")

        if self.args.source_criterion in ['xent', 'cross-entropy']:
            
            self.source_grad, self.source_gnorm = victim.gradient(_sources, _target_classes, selection=self.args.source_selection_strategy,images_negative=_sources_negative,clip=4,coe=self.args.coe)
            
            # I add source negative here 
            #self.source_grad_negative, _ = victim.gradient(_sources_negative, _target_classes,clip=1.06)
        
        else:
            
            raise ValueError('Invalid source criterion chosen ...')
        
        print(f'Source Grad Norm is {self.source_gnorm}')


        
            

        if self.args.repel != 0:
            
            self.source_clean_grad, _ = victim.gradient(_sources, _true_classes)
        
        else:
            
            self.source_clean_grad = None

        # The PGD tau that will actually be used:
        # This is not super-relevant for the adam variants
        # but the PGD variants are especially sensitive
        # E.G: 92% for PGD with rule 1 and 20% for rule 2
        if self.args.attackoptim in ['PGD', 'GD']:
            # Rule 1
            self.tau0 = self.args.eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
        elif self.args.attackoptim in ['momSGD', 'momPGD']:
            # Rule 1a
            self.tau0 = self.args.eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
            self.tau0 = self.tau0.mean()
        else:
            # Rule 2
            self.tau0 = self.args.tau * (self.args.pbatch / 512) / self.args.ensemble




    def _run_trial(self, victim, kettle):
        """Run a single trial."""
        poison_delta = kettle.initialize_poison()

        dataloader = kettle.poisonloader

        if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
            # poison_delta.requires_grad_()
            if self.args.attackoptim in ['Adam', 'signAdam']:

                att_optimizer = torch.optim.Adam([poison_delta], lr=self.tau0, weight_decay=0)
            
            else:

                att_optimizer = torch.optim.SGD([poison_delta], lr=self.tau0, momentum=0.9, weight_decay=0)
            
            if self.args.scheduling:
                
                scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[self.args.attackiter // 2.667, self.args.attackiter // 1.6,
                                                                                            self.args.attackiter // 1.142], gamma=0.1)
            poison_delta.grad = torch.zeros_like(poison_delta)
            
            dm, ds = kettle.dm.to(device=torch.device('cpu')), kettle.ds.to(device=torch.device('cpu'))
            
            poison_bounds = torch.zeros_like(poison_delta)
        
        else:
            
            poison_bounds = None
        
        # attackiter = 250 
        for step in range(self.args.attackiter):
            
            source_losses = 0
            poison_correct = 0
            for batch, example in enumerate(dataloader):
                loss, prediction = self._batched_step(poison_delta, poison_bounds, example, victim, kettle)
                source_losses += loss
                poison_correct += prediction

                if self.args.dryrun:
                    break

            # Note that these steps are handled batch-wise for PGD in _batched_step
            # For the momentum optimizers, we only accumulate gradients for all poisons
            # and then use optimizer.step() for the update. This is math. equivalent
            # and makes it easier to let pytorch track momentum.
            if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                if self.args.attackoptim in ['momPGD', 'signAdam']:
                    poison_delta.grad.sign_()
                att_optimizer.step()
                if self.args.scheduling:
                    scheduler.step()
                att_optimizer.zero_grad()
                with torch.no_grad():
                    # Projection Step
                    poison_delta.data = torch.max(torch.min(poison_delta, self.args.eps /
                                                            ds / 255), -self.args.eps / ds / 255)
                    poison_delta.data = torch.max(torch.min(poison_delta, (1 - dm) / ds -
                                                            poison_bounds), -dm / ds - poison_bounds)

            source_losses = source_losses / (batch + 1)
            poison_acc = poison_correct / len(dataloader.dataset)
            if step % (self.args.attackiter // 25) == 0 or step == (self.args.attackiter - 1):
                print(f'Iteration {step}: Source loss is {source_losses:2.4f}, '
                      f'Poison clean acc is {poison_acc * 100:2.2f}%')

            if self.args.step:
                if self.args.clean_grad:
                    victim.step(kettle, None, self.sources, self.true_classes)
                else:
                    victim.step(kettle, poison_delta, self.sources, self.true_classes)

            if self.args.dryrun:
                break

            if not self.args.retrain_scenario == None:
                if step % (self.args.retrain_iter) == 0 and step != 0 and step != (self.args.attackiter - 1):
                    print("Retrainig the base model at iteration {}".format(step))
                    poison_delta.detach()
                    
                    if self.args.retrain_scenario == 'from-scratch':
                        victim.initialize()
                        print('Model reinitialized to random seed.')
                    elif self.args.retrain_scenario == 'finetuning':
                        if self.args.load_feature_repr:
                            victim.load_feature_representation()
                        victim.reinitialize_last_layer(reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True)
                        print('Completely warmstart finetuning!')

                    victim._iterate(kettle, poison_delta=poison_delta, max_epoch=self.args.retrain_max_epoch)
                    print('Retraining done!')
                    self._initialize_brew(victim, kettle)

        return poison_delta, source_losses

    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle):
        """Take a step toward minmizing the current source loss."""
        inputs, labels, ids = example

        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        # Check adversarial pattern ids
        poison_slices, batch_positions = kettle.lookup_poison_indices(ids)

        # This is a no-op in single network brewing
        # In distributed brewing, this is a synchronization operation

        # la ji 
        inputs, labels, poison_slices, batch_positions, randgen = victim.distributed_control(
            inputs, labels, poison_slices, batch_positions)

        # If a poisoned id position is found, the corresponding pattern is added here:
        
        if len(batch_positions) > 0:
            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
            if self.args.clean_grad:
                delta_slice = torch.zeros_like(delta_slice)
            delta_slice.requires_grad_()  # TRACKING GRADIENTS FROM HERE
            poison_images = inputs[batch_positions]
            inputs[batch_positions] += delta_slice





  
            if self.args.source_criterion in ['cw', 'carlini-wagner']:
                loss_fn = cw_loss
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
            # Change loss function to include corrective terms if mixing with correction
            
            # update source loss here 
     
         
            
            criterion = loss_fn

            closure = self._define_objective(inputs, labels, criterion, self.sources, self.target_classes,
                                             self.true_classes)
            loss, prediction = victim.compute(closure, self.source_grad, self.source_clean_grad,self.source_gnorm)
            delta_slice = victim.sync_gradients(delta_slice)

            # if self.args.clean_grad:
            #     delta_slice.data = poison_delta[poison_slices].detach().to(**self.setup)

            # Update Step
            if self.args.attackoptim in ['PGD', 'GD']:
                delta_slice = self._pgd_step(delta_slice, poison_images, self.tau0, kettle.dm, kettle.ds)

                # Return slice to CPU:
                poison_delta[poison_slices] = delta_slice.detach().to(device=torch.device('cpu'))
            elif self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                poison_delta.grad[poison_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
                poison_bounds[poison_slices] = poison_images.detach().to(device=torch.device('cpu'))
            else:
                raise NotImplementedError('Unknown attack optimizer.')
        else:
            loss, prediction = torch.tensor(0), torch.tensor(0)

        return loss.item(), prediction.item()

    def _define_objective():
        """Implement the closure here."""
        def closure(model, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            raise NotImplementedError()
            return source_loss.item(), prediction.item()

    def _pgd_step(self, delta_slice, poison_imgs, tau, dm, ds):
        """PGD step."""
        with torch.no_grad():
            # Gradient Step
            if self.args.attackoptim == 'GD':
                delta_slice.data -= delta_slice.grad * tau
            else:
                delta_slice.data -= delta_slice.grad.sign() * tau

            # Projection Step
            delta_slice.data = torch.max(torch.min(delta_slice, self.args.eps /
                                                   ds / 255), -self.args.eps / ds / 255)
            delta_slice.data = torch.max(torch.min(delta_slice, (1 - dm) / ds -
                                                   poison_imgs), -dm / ds - poison_imgs)
        return delta_slice


    def patch_sources(self, kettle):
        """Backdoor trigger attacks need to patch kettle.sources."""
        pass
