
dataset_defaults = {
    'fmow': {
        'epochs': 12,
        'batch_size': 32,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 1e-4,
            
            'amsgrad': True,
            
        },
        'pretrain_iters': 24000,
        'meta_lr': 0.01,
        'meta_steps': 5,
        'selection_metric': 'acc_worst_region',
        'reload_inner_optim': True,
        'eval_iters': 500
    },
    'camelyon': {
        'epochs': 20,
        'batch_size': 32,
        'optimiser': 'SGD',
        'optimiser_args': {
            'momentum': 0.9,
            'lr': 1e-4,

        },
        'pretrain_iters': 10000,
        'meta_lr': 0.01,
        'meta_steps': 3,
        'selection_metric': 'acc_avg',
        'reload_inner_optim': True,
        'eval_iters': -1
    },
    'poverty': {
        'epochs': 200,
        'batch_size': 64,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 1e-3,
            
            'amsgrad': True,
            
        },
        'pretrain_iters': 5000,
        'meta_lr': 0.1,
        'meta_steps': 5,
        'selection_metric': 'r_wg',
        'reload_inner_optim': True,
        'eval_iters': -1,
        'scheduler': 'StepLR',
        'scheduler_kwargs': {'gamma': 0.96,'step_size': 1,},
    },
    'iwildcam': {
        'epochs': 9,
        'batch_size': 16,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 1e-4,
            'weight_decay': 0.0,
            'amsgrad': True,
            
        },
        'pretrain_iters': 24000,
        'meta_lr': 0.01,
        'meta_steps': 10,
        'selection_metric': 'F1-macro_all',
        'reload_inner_optim': True,
        'eval_iters': 1000
    },
    'civil': {
        'epochs': 5,
        'batch_size': 16,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 1e-5,
            'amsgrad': True,
        },
        'pretrain_iters': 20000,
        'meta_lr': 0.05,
        'meta_steps': 5,
        'selection_metric': 'acc_wg',
        'reload_inner_optim': True,
        'eval_iters': 500
    },
    'rxrx': {
        'epochs': 90,
        'batch_size': 72,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'amsgrad': True,
            'betas': (0.9, 0.999),
        },
        'pretrain_iters': 15000,
        'meta_lr': 0.01,
        'meta_steps': 10,
        'selection_metric': 'acc_avg',
        'reload_inner_optim': True,
        'eval_iters': 2000,
        'scheduler': 'cosine_schedule_with_warmup',
        'scheduler_kwargs': {'num_warmup_steps': 5415},
    },
}
