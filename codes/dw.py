"""General interface script to launch poisoning jobs."""

import torch

import datetime
import time

import forest

from forest.filtering_defenses import get_defense
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()

if args.deterministic:
    forest.utils.set_deterministic()


if __name__ == "__main__":

    for iter in range(0,1):

        # args.eps=4+4*iter
        setup = forest.utils.system_startup(args)

        model = forest.Victim(args, setup=setup)
        data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup)
        witch = forest.Witch(args, setup=setup)

        if args.backdoor_poisoning:
            witch.patch_sources(data)

        start_time = time.time()
        if args.pretrained_model:
            print('Loading pretrained model...')
            stats_clean = None
        elif args.skip_clean_training:
            print('Skipping clean training...')
            stats_clean = None
        else:
            stats_clean = model.train(data, max_epoch=args.max_epoch)
        train_time = time.time()

        if args.poison_selection_strategy != None:
            data.select_poisons(model, args.poison_selection_strategy)

        poison_delta = witch.brew(model, data)

    # save poison data 
    # torch.save(poison_delta, args.poison_path)
    
        craft_time = time.time()
    
        filter_stats = dict()

        if not args.pretrained_model and args.retrain_from_init:
            stats_rerun = model.retrain(data, poison_delta)
        else:
            stats_rerun = None

        if args.vnet is not None:  # Validate the transfer model given by args.vnet
            train_net = args.net
            args.net = args.vnet
            args.ensemble = len(args.vnet)
            if args.vruns > 0:
                model = forest.Victim(args, setup=setup)  # this instantiates a new model with a different architecture
                stats_results = model.validate(data, poison_delta, val_max_epoch=args.val_max_epoch)
            else:
                stats_results = None
            args.net = train_net

# Non Transferable
        else:  # Validate the main model
            if args.vruns > 0:
                stats_results = model.validate(data, poison_delta, val_max_epoch=args.val_max_epoch)
            else:
                stats_results = None
    
    #     torch.save(model.model,args.modelsave_path)
    # torch.save(model.model.state_dict(), "./models/vgg_16_state_dict.pth")
    
        test_time = time.time()

        timestamps = dict(train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(',', ''),
                      craft_time=str(datetime.timedelta(seconds=craft_time - train_time)).replace(',', ''),
                      test_time=str(datetime.timedelta(seconds=test_time - craft_time)).replace(',', ''))
    # Save run to table
        results = (stats_clean, stats_rerun, stats_results)
        #forest.utils.record_results(data, witch.stat_optimal_loss, results,
         #                       args, model.defs, model.model_init_seed, extra_stats={**filter_stats, **timestamps})

    # Export
        if True:
             data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

        print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
        print('---------------------------------------------------')
        print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
        print(f'--------------------------- craft time: {str(datetime.timedelta(seconds=craft_time - train_time))}')
        print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - craft_time))}')
        print('-------------Job finished.-------------------------')
