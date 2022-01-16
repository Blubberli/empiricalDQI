import os
from datetime import datetime


def reset_wandb_env():
    print('resetting wandb env')
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            print(k, v)
            del os.environ[k]
    # force no watching. seems to have been reset somehow. makes saving slower
    os.environ['WANDB_WATCH'] = 'false'


def get_name_with_hyperparams(data_args, training_args, model_args, fold_id):
    group = ((training_args.output_dir_prefix +
              f"-seqlen{data_args.max_seq_length}"
              f"-batchsize{training_args.per_device_train_batch_size}"
              f"-model{model_args.model_name_or_path}"
              f"f-quality{data_args.quality_dim}"
              f"-lr{training_args.learning_rate}"))
    run_name = group + (f"-fold{fold_id}" + f"-time{datetime.now().strftime('%Y%m%d%H%M%S')}")
    return group, run_name
