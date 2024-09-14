"""Data class, holding information about dataloaders and poison ids."""

import torch
import numpy as np

import pickle

import datetime
import os
import random
import PIL

from .datasets import construct_datasets, Subset
from .cached_dataset import CachedDataset
from ..victims.context import GPUContext

from .diff_data_augmentation import RandomTransform, RandomGridShift, RandomTransformFixed, FlipLR
from .mixing_data_augmentations import Mixup, Cutout, Cutmix, Maxup

from ..consts import PIN_MEMORY, NORMALIZE, BENCHMARK, DISTRIBUTED_BACKEND, SHARING_STRATEGY, MAX_THREADING

from ..utils import cw_loss

torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

class _Kettle():
    """Brew poison with given arguments.

    Data class.
    Attributes:
    - trainloader
    - validloader
    - poisonloader
    - poison_ids
    - trainset/poisonset/sourceset

    Most notably .poison_lookup is a dictionary that maps image ids to their slice in the poison_delta tensor.

    Initializing this class will set up all necessary attributes.

    Other data-related methods of this class:
    - initialize_poison
    - export_poison

    """

    def __init__(self, args, batch_size, augmentations, mixing_method=dict(type=None, strength=0.0),
                 setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize with given specs..."""
        self.args, self.setup = args, setup
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.mixing_method = mixing_method

        self.trainset, self.validset = construct_datasets(self.args.dataset, self.args.data_path, NORMALIZE)
        if self.args.pretrain_dataset is not None:
            self.pretrain_trainset, self.pretrain_validset = construct_datasets(self.args.pretrain_dataset,
                                                                                self.args.data_path, NORMALIZE)
        self.prepare_diff_data_augmentations(normalize=NORMALIZE)

        num_workers = self.get_num_workers()

        if self.args.lmdb_path is not None:
            from .lmdb_datasets import LMDBDataset  # this also depends on py-lmdb
            self.trainset = LMDBDataset(self.trainset, self.args.lmdb_path, 'train')
            self.validset = LMDBDataset(self.validset, self.args.lmdb_path, 'val')

        if self.args.cache_dataset:
            self.trainset = CachedDataset(self.trainset, num_workers=num_workers)
            self.validset = CachedDataset(self.validset, num_workers=num_workers)
            num_workers = 0

        self.prepare_experiment()

        # Generate loaders:
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=min(self.batch_size, len(self.trainset)),
                                                       shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        self.validloader = torch.utils.data.DataLoader(self.validset, batch_size=min(self.batch_size, len(self.validset)),
                                                       shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        
        validated_batch_size = max(min(args.pbatch, len(self.poisonset)), 1)
        
        self.poisonloader = torch.utils.data.DataLoader(self.poisonset, batch_size=validated_batch_size,
                                                        shuffle=self.args.pshuffle, drop_last=False, num_workers=num_workers,
                                                        pin_memory=PIN_MEMORY)

        if self.args.pretrain_dataset is not None:
            
            self.pretrainloader = torch.utils.data.DataLoader(self.pretrain_trainset, batch_size=min(self.batch_size, len(self.trainset)),
                                                              shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
            
            self.prevalidloader = torch.utils.data.DataLoader(self.pretrain_validset, batch_size=min(self.batch_size, len(self.validset)),
                                                              shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

        # Ablation on a subset?
        if args.ablation < 1.0:
            self.sample = random.sample(range(len(self.trainset)), int(self.args.ablation * len(self.trainset)))
            self.partialset = Subset(self.trainset, self.sample)
            self.partialloader = torch.utils.data.DataLoader(self.partialset, batch_size=min(self.batch_size, len(self.partialset)),
                                                             shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        # Save clean ids for later:
        self.clean_ids = [idx for idx in range(len(self.trainset)) if self.poison_lookup.get(idx) is None]
        # Finally: status
        self.print_status()


    """ STATUS METHODS """

    def print_status(self):
        class_names = self.trainset.classes
        if self.args.poison_selection_strategy == None:
            print(
                f'Poisoning setup generated for threat model {self.args.threatmodel} and '
                f'budget of {self.args.budget * 100}% - {len(self.poisonset)} images:')
        else:
            print(
                f'Poisoning setup generated for threat model {self.args.threatmodel} and '
                f'raw poisons before selection are {len(self.poisonset)} images:')

        _target = [self.poison_setup['target_class'][0]] * self.source_train_num
        if len(self.source_train_ids) > 5:
            print(
                f'--Source images drawn from class {", ".join([class_names[self.source_trainset[i][1]] for i in range(len(self.source_trainset))][:5])}...'
                f' with ids {self.source_train_ids[:5]}...')
            print(f'--Source images assigned target class {", ".join([class_names[i] for i in _target[:5]])}...')
        else:
            print(
                f'--Source images drawn from class {", ".join([class_names[self.source_trainset[i][1]] for i in range(len(self.source_trainset))])}.'
                f' with ids {self.source_train_ids}.')
            print(f'--Source images assigned target class {", ".join([class_names[i] for i in _target])}.')

        if self.poison_setup["poison_class"] is not None:
            print(f'--Poison images drawn from class {class_names[self.poison_setup["poison_class"]]}.')
        else:
            print(f'--Poison images drawn from all classes.')

        if self.args.ablation < 1.0:
            print(f'--Partialset is {len(self.partialset)/len(self.trainset):2.2%} of full training set')
            num_p_poisons = len(np.intersect1d(self.poison_ids.cpu().numpy(), np.array(self.sample)))
            print(f'--Poisons in partialset are {num_p_poisons} ({num_p_poisons/len(self.poison_ids):2.2%})')

    def get_num_workers(self):
        """Check devices and set an appropriate number of workers."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            max_num_workers = 4 * num_gpus
        else:
            max_num_workers = 4
        if torch.get_num_threads() > 1 and MAX_THREADING > 0:
            worker_count = min(min(2 * torch.get_num_threads(), max_num_workers), MAX_THREADING)
        else:
            worker_count = 0
        # worker_count = 200
        print(f'Data is loaded with {worker_count} workers.')
        return worker_count

    """ CONSTRUCTION METHODS """

    def prepare_diff_data_augmentations(self, normalize=True):
        """Load differentiable data augmentations separately from usual torchvision.transforms."""
        trainset, validset = construct_datasets(self.args.dataset, self.args.data_path, normalize)


        # Prepare data mean and std for later:
        if normalize:
            self.dm = torch.tensor(trainset.data_mean)[None, :, None, None].to(**self.setup)
            self.ds = torch.tensor(trainset.data_std)[None, :, None, None].to(**self.setup)
        else:
            self.dm = torch.tensor(trainset.data_mean)[None, :, None, None].to(**self.setup).zero_()
            self.ds = torch.tensor(trainset.data_std)[None, :, None, None].to(**self.setup).fill_(1.0)


        # Train augmentations are handled separately as they possibly have to be backpropagated
        if self.augmentations is not None or self.args.paugment:
            if 'CIFAR' in self.args.dataset:
                params = dict(source_size=32, target_size=32, shift=8, fliplr=True)
            elif 'MNIST' in self.args.dataset:
                params = dict(source_size=28, sourcetarget_size_size=28, shift=4, fliplr=True)
            elif 'TinyImageNet' in self.args.dataset:
                params = dict(source_size=64, target_size=64, shift=64 // 4, fliplr=True)
            elif 'SubImageNet' in self.args.dataset:
                params = dict(source_size=224, target_size=224, shift=224 // 4, fliplr=True)
            
            elif 'STL' in self.args.dataset:
                params = dict(source_size=96, target_size=96, shift=96 // 4, fliplr=True)
            else:

                raise NotImplementedError("GGGG")

                
            if self.augmentations == 'default':
                self.augment = RandomTransform(**params, mode='bilinear')
            elif self.augmentations == 'default-align':
                self.augment = RandomTransform(**params, mode='bilinear', align=False)
            elif self.augmentations == 'default-nn':
                self.augment = RandomTransform(**params, mode='nearest')
            elif self.augmentations == 'default-nn-align':
                self.augment = RandomTransform(**params, mode='nearest', align=False)
            elif self.augmentations == 'grid-shift':
                self.augment = RandomGridShift(**params)
            elif self.augmentations == 'LR':
                self.augment = FlipLR(**params)
            elif self.augmentations == 'affine-trafo':
                self.augment = RandomTransformFixed(**params, mode='bilinear')
            elif self.augmentations == 'affine-trafo-nn':
                self.augment = RandomTransformFixed(**params, mode='nearest')
            elif self.augmentations == 'affine-trafo-align':
                self.augment = RandomTransformFixed(**params, mode='bilinear', align=False)
            elif self.augmentations == 'affine-trafo-nn-align':
                self.augment = RandomTransformFixed(**params, mode='nearest', align=False)
            elif self.augmentations == 'affine-trafo-no-flip':
                params['fliplr'] = False
                self.augment = RandomTransformFixed(**params, mode='bilinear')
            elif self.augmentations == 'affine-trafo-nn-no-flip':
                params['fliplr'] = False
                self.augment = RandomTransformFixed(**params, mode='nearest')
            elif not self.augmentations:
                print('Data augmentations are disabled.')
                self.augment = RandomTransform(**params, mode='bilinear')
            else:
                raise ValueError(f'Invalid diff. transformation given: {self.augmentations}.')

            if self.mixing_method['type'] != '' or self.args.pmix:
                if 'mixup' in self.mixing_method['type']:
                    nway = int(self.mixing_method['type'][0]) if 'way' in self.mixing_method['type'] else 2
                    self.mixer = Mixup(nway=nway, alpha=self.mixing_method['strength'])
                elif 'cutmix' in self.mixing_method['type']:
                    self.mixer = Cutmix(alpha=self.mixing_method['strength'])
                elif 'cutout' in self.mixing_method['type']:
                    self.mixer = Cutout(alpha=self.mixing_method['strength'])
                else:
                    raise ValueError(f'Invalid mixing data augmentation {self.mixing_method["type"]} given.')

                if 'maxup' in self.mixing_method['type']:
                    self.mixer = Maxup(self.mixer, ntrials=4)


        return trainset, validset


    def prepare_experiment(self):
        """Choose sources from some label which will be poisoned toward some other chosen label, by modifying some
        subset of the training data within some bounds."""
        raise NotImplementedError()

    """ Methods modifying and applying poisons. """

    def initialize_poison(self, initializer=None):
        """Initialize according to args.init.

        Propagate initialization in distributed settings.
        """
        if initializer is None:
            initializer = self.args.init

        # ds has to be placed on the default (cpu) device, not like self.ds
        ds = torch.tensor(self.trainset.data_std)[None, :, None, None]
        if initializer == 'zero':
            init = torch.zeros(len(self.poison_ids), *self.trainset[0][0].shape)
        elif initializer == 'rand':
            init = (torch.rand(len(self.poison_ids), *self.trainset[0][0].shape) - 0.5) * 2
            init *= self.args.eps / ds / 255
        elif initializer == 'randn':
            init = torch.randn(len(self.poison_ids), *self.trainset[0][0].shape)
            init *= self.args.eps / ds / 255
        elif initializer == 'normal':
            init = torch.randn(len(self.poison_ids), *self.trainset[0][0].shape)
        else:
            raise NotImplementedError()

        init.data = torch.max(torch.min(init, self.args.eps / ds / 255), -self.args.eps / ds / 255)

        # If distributed, sync poison initializations
        if self.args.local_rank is not None:
            if DISTRIBUTED_BACKEND == 'nccl':
                init = init.to(device=self.setup['device'])
                torch.distributed.broadcast(init, src=0)
                init.to(device=torch.device('cpu'))
            else:
                torch.distributed.broadcast(init, src=0)
        return init

    def reset_trainset(self, new_ids):
        num_workers = self.get_num_workers()
        self.trainset = Subset(self.trainset, indices=new_ids)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=min(self.batch_size, len(self.trainset)),
                                                       shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)


    def lookup_poison_indices(self, image_ids):
        """Given a list of ids, retrieve the appropriate poison perturbation from poison delta and apply it."""
        poison_slices, batch_positions = [], []
        for batch_id, image_id in enumerate(image_ids.tolist()):
            lookup = self.poison_lookup.get(image_id)
            if lookup is not None:
                poison_slices.append(lookup)
                batch_positions.append(batch_id)

        return poison_slices, batch_positions

    """ EXPORT METHODS """

    def export_poison(self, poison_delta, path=None, mode='automl'):
        """Export poisons in either packed mode (just ids and raw data) or in full export mode, exporting all images.

        In full export mode, export data into folder structure that can be read by a torchvision.datasets.ImageFolder

        In automl export mode, export data into a single folder and produce a csv file that can be uploaded to
        google storage.
        """
        if path is None:
            path = self.args.poison_path

        dm = torch.tensor(self.trainset.data_mean)[:, None, None]
        ds = torch.tensor(self.trainset.data_std)[:, None, None]

        def _torch_to_PIL(image_tensor):
            """Torch->PIL pipeline as in torchvision.utils.save_image."""
            image_denormalized = torch.clamp(image_tensor * ds + dm, 0, 1)
            image_torch_uint8 = image_denormalized.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
            image_PIL = PIL.Image.fromarray(image_torch_uint8.numpy())
            return image_PIL

        def _save_image(input, label, idx, location, train=True):
            """Save input image to given location, add poison_delta if necessary."""
            filename = os.path.join(location, str(idx) + '.png')

            lookup = self.poison_lookup.get(idx)
            if (lookup is not None) and train:
                input += poison_delta[lookup, :, :, :]
                _torch_to_PIL(input).save(filename)
            
            else:

                pass
        # Save either into packed mode, ImageDataSet Mode or google storage mode
        if mode == 'packed':
            data = dict()
            data['poison_setup'] = self.poison_setup
            data['poison_delta'] = poison_delta
            data['poison_ids'] = self.poison_ids
            data['source_images'] = [data for data in self.sourceset]
            name = f'{path}poisons_packed_{datetime.date.today()}.pth'
            torch.save([poison_delta, self.poison_ids], os.path.join(path, name))

        elif mode == 'limited':
            # Save training set
            names = self.trainset.classes
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'sources', name), exist_ok=True)
            for input, label, idx in self.trainset:
                lookup = self.poison_lookup.get(idx)
                if lookup is not None:
                    _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)
            print('Poisoned training images exported ...')

            # Save secret sources
            for enum, (source, _, idx) in enumerate(self.sourceset):
                target_class = self.poison_setup['target_class'][enum]
                _save_image(source, target_class, idx, location=os.path.join(path, 'sources', names[target_class]), train=False)
            print('Source images exported with target class labels ...')

        elif mode == 'full':
            # Save training set
            names = self.trainset.classes
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'test', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'sources', name), exist_ok=True)
            for input, label, idx in self.trainset:
                _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)
            print('Poisoned training images exported ...')

            for input, label, idx in self.validset:
                _save_image(input, label, idx, location=os.path.join(path, 'test', names[label]), train=False)
            print('Unaffected validation images exported ...')

            # Save secret sources
            for enum, (source, _, idx) in enumerate(self.sourceset):
                target_class = self.poison_setup['target_class'][enum]
                _save_image(source, target_class, idx, location=os.path.join(path, 'sources', names[target_class]), train=False)
            print('Source images exported with target class labels ...')

        elif mode in ['automl-upload', 'automl-all', 'automl-baseline']:
            from ..utils import automl_bridge
            sourceclass = self.sourceset[0][1]
            poisonclass = self.poison_setup["poison_class"]

            name_candidate = f'{self.args.name}_{self.args.dataset}T{sourceclass}P{poisonclass}'
            name = ''.join(e for e in name_candidate if e.isalnum())

            if mode == 'automl-upload':
                automl_phase = 'poison-upload'
            elif mode == 'automl-all':
                automl_phase = 'all'
            elif mode == 'automl-baseline':
                automl_phase = 'upload'
            automl_bridge(self, poison_delta, name, mode=automl_phase, dryrun=self.args.dryrun)

        elif mode == 'numpy':

            _, h, w = self.trainset[0][0].shape
            training_data = np.zeros([len(self.trainset), h, w, 3])
            labels = np.zeros(len(self.trainset))
            for input, label, idx in self.trainset:
                lookup = self.poison_lookup.get(idx)
                if lookup is not None:
                    input += poison_delta[lookup, :, :, :]
                training_data[idx] = np.asarray(_torch_to_PIL(input))
                labels[idx] = label

            np.save(os.path.join(path, 'poisoned_training_data.npy'), training_data)
            np.save(os.path.join(path, 'poisoned_training_labels.npy'), labels)
            # names = self.trainset.classes
            # for name in names:
            #     os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
            #     os.makedirs(os.path.join(path, 'test', name), exist_ok=True)
            #     os.makedirs(os.path.join(path, 'sources', name), exist_ok=True)
            
            # _, h, w = self.trainset[0][0].shape
            # training_data = np.zeros([len(poison_delta), h, w, 3])
            # labels = np.zeros(len(poison_delta))
            # i = 0
            # for input, label, idx in self.trainset:
                
            #     _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)               
            #     lookup = self.poison_lookup.get(idx)
            #     input_base = input.clone()
            #     if lookup is not None:

            #         input += poison_delta[lookup, :, :, :]
            #         indices = input!=input_base
            #         input = input[indices]    
            #         training_data[i:i+len(input)] = np.asarray(_torch_to_PIL(input))
            #         labels[i:i+len(input)] = label[indices]
            #         i+=len(input)
            # np.save(os.path.join(path, 'poisoned_training_data.npy'), training_data)
            # np.save(os.path.join(path, 'poisoned_training_labels.npy'), labels)

        elif mode == 'kettle-export':
            with open(f'kette_{self.args.dataset}{self.args.model}.pkl', 'wb') as file:
                pickle.dump([self, poison_delta], file, protocol=pickle.HIGHEST_PROTOCOL)

        elif mode == 'benchmark':
            foldername = f'{self.args.name}_{"_".join(self.args.net)}'
            sub_path = os.path.join(path, 'benchmark_results', foldername, str(self.args.benchmark_idx))
            os.makedirs(sub_path, exist_ok=True)

            # Poisons
            benchmark_poisons = []
            for lookup, key in enumerate(self.poison_lookup.keys()):  # This is a different order than we usually do for compatibility with the benchmark
                input, label, _ = self.trainset[key]
                input += poison_delta[lookup, :, :, :]
                benchmark_poisons.append((_torch_to_PIL(input), int(label)))

            with open(os.path.join(sub_path, 'poisons.pickle'), 'wb+') as file:
                pickle.dump(benchmark_poisons, file, protocol=pickle.HIGHEST_PROTOCOL)

            # Source
            source, source_label, _ = self.sourceset[0]
            with open(os.path.join(sub_path, 'source.pickle'), 'wb+') as file:
                pickle.dump((_torch_to_PIL(source), source_label), file, protocol=pickle.HIGHEST_PROTOCOL)

            # Indices
            with open(os.path.join(sub_path, 'base_indices.pickle'), 'wb+') as file:
                pickle.dump(self.poison_ids, file, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            raise NotImplementedError()

        print('Dataset fully exported.')


    def select_poisons(self, victim, selection):
        victim.eval(dropout=True)

        print('Selecting poisons ...')

        if self.args.source_criterion in ['cw', 'carlini-wagner']:
            criterion = cw_loss
        else:
            criterion = torch.nn.CrossEntropyLoss()
            
        
        if selection == 'max_gradient':
            print('Selections strategy is {}'.format(selection))
            images = torch.stack([data[0] for data in self.poisonset], dim=0).to(**self.setup)
            labels = torch.tensor([data[1] for data in self.poisonset]).to(device=self.setup['device'], dtype=torch.long)
            poison_ids = torch.tensor([data[2].item() for data in self.poisonset], dtype=torch.long)

            # single model
            if self.args.ensemble == 1:
                grad_norms = []
                model = victim.model
                differentiable_params = [p for p in model.parameters() if p.requires_grad]
                for image, label in zip(images, labels):
                    loss = criterion(model(image.unsqueeze(0)), label.unsqueeze(0))
                    gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                    grad_norm = 0
                    for grad in gradients:
                        grad_norm += grad.detach().pow(2).sum()
                    grad_norms.append(grad_norm.sqrt())
            # ensemble models
            else:
                grad_norms_list = [[] for _ in range(len(victim.models))] 
                for i, model in enumerate(victim.models):
                    with GPUContext(self.setup, model) as model:
                        differentiable_params = [p for p in model.parameters() if p.requires_grad]
                        for image, label in zip(images, labels):
                            loss = criterion(model(image.unsqueeze(0)), label.unsqueeze(0))
                            gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                            grad_norm = 0
                            for grad in gradients:
                                grad_norm += grad.detach().pow(2).sum()
                            grad_norms_list[i].append(grad_norm.sqrt())

                print(f'Taking average gradient norm of ensebmle of {len(victim.models)} models')
                grad_norms = [sum(col) / float(len(col)) for col in zip(*grad_norms_list)]
            
            poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
            indices = [i[0] for i in sorted(enumerate(grad_norms), key=lambda x:x[1])][-poison_num:]

        else:
            raise NotImplementedError('Poison selection {} strategy is not implemented yet!'.format(selection))


        images = images[indices]
        labels = labels[indices]
        poison_ids = poison_ids[indices]
        print('{} poisons with maximum gradients selected'.format(len(indices)))

        print('Updating Kettle poison related fields ...')
        self.poison_ids = poison_ids
        self.poisonset = Subset(self.trainset, indices=self.poison_ids)
        self.poison_lookup = dict(zip(self.poison_ids.tolist(), range(poison_num)))

        num_workers = self.get_num_workers()
        validated_batch_size = max(min(self.args.pbatch, len(self.poisonset)), 1)
        self.poisonloader = torch.utils.data.DataLoader(self.poisonset, batch_size=validated_batch_size,
                                                        shuffle=self.args.pshuffle, drop_last=False, num_workers=num_workers,
                                                        pin_memory=PIN_MEMORY)
        self.clean_ids = [idx for idx in range(len(self.trainset)) if self.poison_lookup.get(idx) is None]
