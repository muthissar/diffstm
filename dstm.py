from dstm.model.short_term import DSTM
from dstm.model.long_term import LTM
from dstm.model.module import Module
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from dstm.util.load_data import SessionPreprocessing
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import os
import pathlib
# class Tee(object):
#     def __init__(self, *files):
#         self.files = files
#     def write(self, obj):
#         for f in self.files:
#             f.write(obj)
#             f.flush() # If you want the output to be visible immediately
#     def flush(self) :
#         for f in self.files:
#             f.flush()
#     # we expect the bus_order to work

if __name__ == '__main__':
    # freeze_support()
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--n_train_workers', type=int, default=16, help="number of training data loader workers")
    parser.add_argument('--seq_max_length', type=int, default=float('inf'), help="truncate sequences at specified length")
    parser.add_argument('--checkpoint', type=str, help="initialize model with spefified checkpoint")
    #parser.add_argument('--dataset', type=str, choices=["essen", "nes", "session", "pop"], default="session", help="(experimental) use specified dataset")
    parser.add_argument('--dataset', type=str, choices=["session"], default="session", help="use specified dataset")
    parser.add_argument('--skip_train', action='store_true', default=False, help="skip training step")
    parser.add_argument('--skip_test', action='store_true', default=False, help="skip testing step")
    #parser.add_argument('--dataset_size', type=str, choices=["small", "medium", "large"], default="large")
    parser.add_argument('--loop_data', action='store_true', default=False, help="(experimental) looping data augmentation")
    parser.add_argument('--job_id', type=int, help="inject job id (used for naming processes)")
    parser.add_argument('--early_stopping', action='store_true', default=True, help="when set, training will stop after no optimization progress for 3 epochs")
    parser.add_argument('--save_last', action='store_true', default=False, help="force saving of last model checkpoint (oterwise best model checkpoint will be kept)")
    parser.add_argument("--no_log", action='store_true', default=False, help="Skip wandb logging.")
    #parser.add_argument('--output_folder', type=str, default="out")
    subparsers = parser.add_subparsers(dest="subparser_name")
    parser = Module.add_model_specific_args(parser, subparsers)
    parser_cstm = subparsers.add_parser(name='dstm', help="subcommand for Differential Short-Term Models (DSTMs) models.")
    DSTM.add_model_specific_args(parser_cstm)
    parser_ltm = subparsers.add_parser(name='ltm', help="subcommand for Long-Term Models (LTMs) ")
    LTM.add_model_specific_args(parser_ltm)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # #NOTE: reserve all gpus
    # reserve_gpus(args.gpus)

    #basedir = "out/{}/{}".format(args.dataset,args.dataset_size)
    basedir = "out/{}".format(args.dataset)
    
    
    #NOTE: get dataset
    if args.dataset == "session":
        dataPreprocessor = SessionPreprocessing(loop=args.loop_data, max_workers=args.n_train_workers if args.n_train_workers is not None else 1)
        dataPreprocessor.prepare_dataset()
    
    else:
        raise NotImplementedError("Dataset {} is not implimented".format(args.dataset))
    d = dataPreprocessor.d
    
    model_type = args.subparser_name
    if args.checkpoint is not None:
        if model_type  == "ltm":
            mc = LTM.load_from_checkpoint(args.checkpoint)
        else:
            mc = DSTM.load_from_checkpoint(args.checkpoint)
        
    else:
        if model_type  == "ltm":
            mc = LTM(d, args)
        else:
            mc = DSTM(d, args)
    
    if args.job_id is None:
        job_id = os.getpid()
    else:
        job_id = args.job_id
    # TODO: add time_stamp
    
    #print_logs
    # for o in ['out', 'err']:
    #     folder = '{}/{}'.format(basedir, o)
    #     pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    #     f = open('{}/{}-{}.{}'.format(folder, model_type, job_id, o), 'w')
    #     n = "std" + o
    #     vars(sys)[n] = Tee(vars(sys)[n], f)
        

    #logging
    if args.no_log:
        logger = False
    else:
        log_name = model_type
        logger = WandbLogger(project = 'DSTM')
        logger.log_hyperparams(args)
    
    #savemodel
    modeldir = "{}/model".format(basedir)
    pathlib.Path(modeldir).mkdir(parents=True, exist_ok=True)
    model_name = "{}-{}".format(model_type, job_id) + '-{epoch:02d}-{val_loss:.2f}-{val_precision:.2f}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=modeldir,
        filename=model_name,
        save_last=args.save_last
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{}-last".format(model_name)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # swa_callback = StochasticWeightAveraging(
    #     swa_epoch_start=0.,
    #     #annealing_epochs=0,

    # )
    callbacks = [
        checkpoint_callback,
    #    swa_callback,
        lr_monitor
    ]
    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss'))
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        default_root_dir=basedir,
        logger=logger,
        #plugins=DDPPlugin(find_unused_parameters=False)
    )
    train_loader = dataPreprocessor.get_data_loader('train',
        seq_max_length = args.seq_max_length,
        batch_size=args.batch_size,
        num_workers=args.n_train_workers,
        shuffle=True,
        pin_memory=True
    )
    if args.seq_max_length <  float('inf'):
            val_loader = dataPreprocessor.get_data_loader('valid',
            batch_size = 2,
            num_workers=0,
            pin_memory=True
        )
    else:
        val_loader = dataPreprocessor.get_data_loader('valid',
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True
        )
    
    
    if not args.skip_train:
        print("Training")
        trainer.fit(mc, train_loader, val_loader)
    if not args.skip_test:
        if args.seq_max_length <  float('inf'):
            test_loader = dataPreprocessor.get_data_loader('test',
                batch_size = 1,
                num_workers=0,
                pin_memory=True
            )
        else:
            test_loader = dataPreprocessor.get_data_loader('test',
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=True
            )
        print("Testing")
        trainer.validate(mc, test_loader)