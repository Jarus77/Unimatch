import tempfile
from pathlib import Path
from functools import partial

from ray import tune as ray_tune
from ray import train as ray_train
from ray.train import Checkpoint, get_checkpoint

from hyperopt import hp

from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search import ConcurrencyLimiter

from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB

from fixmatch import *

globally_best_iou = 0

def fixmatch_trainer(args, cfg):
    global globally_best_iou

    model = init_model(args.nclass, cfg['backbone'])
    print(f"Param count: {count_params(model):.1f}M")
    optimizer = init_optimizer(model, cfg)

    class_weights_np = np.array(cfg['class_weights'])
    class_weights = torch.tensor(class_weights_np).cuda()

    if args.nclass == 1:
        # criterion_jaccard = JaccardLoss("binary")
        pass
    else:
        # criterion_jaccard = JaccardLoss("multiclass")
        # criterion_l = criterion_jaccard
        # criterion_u = criterion_jaccard

        criterion_l = nn.CrossEntropyLoss(class_weights)
        criterion_u = nn.CrossEntropyLoss(class_weights, reduction='none')


    trainloader_l, trainloader_u, valloader = load_data(args, cfg)

    total_iters = len(trainloader_l) * args.num_epochs

    locally_best_iou = 0
    epoch = -1

    if args.use_checkpoint and os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        locally_best_iou = checkpoint['locally_best_iou']

        print(f"Loading checkpoint from epoch: {epoch}")

    print("Starting Training...")
    for epoch in range(epoch + 1, args.num_epochs):
        print(f"Epoch [{epoch}/{args.num_epochs}]\t Previous Best IoU: {locally_best_iou}")
        
        # # late login to reduce log rate 
        # if epoch == 1:
        #     init_logging(args, cfg)

        logs = run_epoch(
            model, optimizer,
            criterion_l, criterion_u, 
            trainloader_l, trainloader_u, valloader,
            epoch, total_iters,
            args, cfg
        )

        loss_t = logs['epoch_train/loss']
        loss_v = logs['eval/loss']
        wIoU = logs['eval/wIoU']
        gl_weights = cfg['grand_loss_weights']
        gl_losses = np.array([loss_t, loss_v, 1 - wIoU])
        
        # gl_losses = np.log(gl_losses)
        grand_loss = sum(gl_weights * gl_losses / sum(gl_weights))

        is_locally_best = wIoU > locally_best_iou
        locally_best_iou = max(wIoU, locally_best_iou)

        logs['main/wIoU'] = locally_best_iou
        logs['main/grand_loss'] = grand_loss

        # log({
        #     'main/wIoU': locally_best_iou,
        #     'main/grand_loss': grand_loss
        # })

        is_globally_best = wIoU > globally_best_iou
        globally_best_iou = max(wIoU, globally_best_iou)

        checkpoint_data = {
            'cfg': cfg,
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if epoch > 10 and is_locally_best:
            checkpoint_data['locally_best_iou'] = locally_best_iou
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir) / "locally_best.pth"
                torch.save(checkpoint_data, checkpoint_path)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                ray_train.report(logs,checkpoint=checkpoint)
        else:
            ray_train.report(logs)

        if epoch > 10 and is_globally_best:
            checkpoint_data['globally_best_iou'] = globally_best_iou
            torch.save(checkpoint_data, os.path.join(args.save_path, 'globally_best.pth'))


    print("Training Completed!")


def main(prev_best_cfgs, param_space, gpus_per_trial):
    set_seed(42)

    args = get_args()

    os.makedirs(args.save_path, exist_ok=True)

    # bohb = TuneBOHB(
    #     points_to_evaluate=prev_best_cfgs
    # )

    # search_alg = ConcurrencyLimiter(bohb, max_concurrent=2)

    # scheduler = HyperBandForBOHB(
    #     max_t=args.num_epochs,
    #     reduction_factor=2,
    # )

    hyperopt = HyperOptSearch(
        param_space,
        metric="main/grand_loss",
        mode="min",
        points_to_evaluate=prev_best_cfgs,
    )

    search_alg = ConcurrencyLimiter(hyperopt, max_concurrent=2)

    scheduler = ASHAScheduler(
        max_t=args.num_epochs,
        grace_period=3,
        reduction_factor=2)
    
    tuner = ray_tune.Tuner(
        ray_tune.with_resources(
            ray_tune.with_parameters(partial(fixmatch_trainer, args)),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=ray_tune.TuneConfig(
            metric="main/grand_loss",
            mode="min",
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=args.num_samples,
        ),
        run_config=ray_train.RunConfig(
            storage_path=args.save_path,
            checkpoint_config=ray_train.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="main/grand_loss",
            )
        ),
        # param_space=param_space,
    )

    results = tuner.fit()

    best_result = results.get_best_result("main/grand_loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial grand loss: {}".format(best_result.metrics["main/grand_loss"]))
    print("Best trial final training loss: {}".format(best_result.metrics["epoch_train/loss"]))
    print("Best trial final validation loss: {}".format(best_result.metrics["eval/loss"]))
    print("Best trial final validation accuracy: {}".format(best_result.metrics["eval/wIoU"]))


if __name__ == "__main__":
    prev_best_cfgs = [
        {
            'lr': 0.0001481,
            'weight_decay': 1.583e-9,

            'conf_thresh': 0.7829,
            'p_jitter': 0.3186,
            'p_gray': 0.6534,
            'p_blur': 0.2515,
        },
        {
            'lr': 0.0003784,
            'weight_decay': 1.071e-7,

            'conf_thresh': 0.6786,
            'p_jitter': 0.01492,
            'p_gray': 0.07219,
            'p_blur': 0.5036,
        },
        {
            'lr': 0.000711,
            'weight_decay': 1.652e-8,

            'conf_thresh': 0.8653,
            'p_jitter': 0.006844,
            'p_gray': 0.6599,
            'p_blur': 0.4109,
        },
        {
            'lr': 0.000634,
            'weight_decay': 7.382e-7,

            'conf_thresh': 0.56,
            'p_jitter': 0.795,
            'p_gray': 0.6707,
            'p_blur': 0.01434,
        },
        {
            'lr': 0.0008521,
            'weight_decay': 1.897e-7,

            'conf_thresh': 0.5874,
            'p_jitter': 0.4585,
            'p_gray': 0.6214,
            'p_blur': 0.2818,
        },
    ]

    # param_space = {
    #     'grand_loss_weights': np.array([1.0, 2.0, 4.0]),
    #     'crop_size': 800,
    #     'batch_size': 2, 
    #     'unlabeled_ratio': 10,

    #     'backbone': 'efficientnet-b0',
        
    #     'class_weights': [0.008, 1.0, 0.048],
    #     'lr': ray_tune.loguniform(1e-5, 1e-3),
    #     'lr_multi': 10.0,
    #     'weight_decay': ray_tune.loguniform(1e-9, 1e-5),
    #     'scheduler': 'poly',

    #     'conf_thresh': ray_tune.qloguniform(0.5, 0.99, 0.01),
    #     'p_jitter': ray_tune.quniform(0.0, 0.8, 0.1),
    #     'p_gray': ray_tune.quniform(0.0, 0.8, 0.1),
    #     'p_blur': ray_tune.quniform(0.0, 0.8, 0.1),
    # }

    param_space = {
        'grand_loss_weights': np.array([1.0, 2.0, 4.0]),
        'crop_size': 800,
        'batch_size': 2, 
        'unlabeled_ratio': 10,

        'backbone': 'efficientnet-b0',
        
        'class_weights': [0.008, 1.0, 0.048],
        'lr': hp.loguniform('lr', 1e-5, 1e-3),
        'lr_multi': 10.0,
        'weight_decay': hp.loguniform('weight_decay', 1e-9, 1e-5),
        'scheduler': 'poly',

        'conf_thresh': hp.qloguniform('conf_thresh', 0.5, 0.99, 0.01),
        'p_jitter': hp.quniform('p_jitter', 0.0, 0.8, 0.1),
        'p_gray': hp.quniform('p_gray', 0.0, 0.8, 0.1),
        'p_blur': hp.quniform('p_blur', 0.0, 0.8, 0.1),
    }

    main(prev_best_cfgs, param_space, gpus_per_trial=0.5)