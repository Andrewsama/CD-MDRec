import sys
sys.path.append('..')
from util import generate_random_seed
from models.model import CD_MDRec
import argparse
from pyhealth.utils import set_seed
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.tasks import drug_recommendation_mimic3_fn
from trainer import Trainer

if __name__ == "__main__":

    # create ArgumentParser instance
    parser = argparse.ArgumentParser(description='CD-MDRec: Contrastive Deep Fusion-based Diffusion Multi-view Drug Recommendation')
    parser.add_argument('--steps', type=int, default=20, help='the step of diffusion')
    parser.add_argument('--ddi_rate', type=float, default=0.06, help='ddi_rate')
    parser.add_argument('--alpha', type=float,  default=0.1, help='hyperparameter for ddi loss')
    parser.add_argument('--beta', type=float, default=0.1, help='hyperparameter for contrast loss')
    parser.add_argument('--gamma', type=float, default=0.3, help='hyperparameter for diffusion loss')
    parser.add_argument('--heads', type=int, default=4, help='the number of multi-head')
    parser.add_argument('--num_layers', type=int, default=2, help='the number of Transformer layer')
    parser.add_argument('--embedding_dim', type=int, default=512, help='embedding dimension')
    parser.add_argument('--samp_step', type=int, default=5, help='p_sample of diffusion')

    parser.add_argument('--batch_size', type=int, default=256, help='batch size')

    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='epoch')


    parser.add_argument('--device', type=str, default="cuda:0",help='The device to run the model on, e.g., "cuda:0" for GPU')
    parser.add_argument('--dataset_path', type=str, default="data/mimic-iii", help='The dataset file path, which should contain the main csv files')
    parser.add_argument('--dev', type=bool, default=False, help='whether to enable dev mode (only use a small subset of the data).')
    parser.add_argument('--refresh_cache', type=bool, default=False, help='whether to refresh the cache')
    parser.add_argument('--seed', type=bool, default=2357077683, help='random seed')


    args = parser.parse_args()

    seed = args.seed
    set_seed(seed)

    # STEP 1: load data
    base_dataset = MIMIC3Dataset(
        root=args.dataset_path,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        dev=args.dev,
        refresh_cache=args.refresh_cache
    )
    sample_dataset = base_dataset.set_task(drug_recommendation_mimic3_fn)
    sample_dataset.stat()

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # STEP 3: define model
    model = CD_MDRec(sample_dataset,
                    embedding_dim=args.embedding_dim,
                    heads=args.heads,
                    num_layers=args.num_layers,
                    alpha=args.alpha,
                    beta=args.beta,
                    steps=args.steps,
                    ddi_rate=args.ddi_rate,
                    gamma=args.gamma,
                   samp_step=args.samp_step,
                    )

    # STEP 4: define trainer
    trainer = Trainer(
        model=model,
        metrics=["jaccard_samples", "pr_auc_samples", "f1_samples", "ddi_score", "roc_auc_samples", "avg_med"],
        device=args.device,
        seed=seed,
        info=[str(args.steps), str(args.ddi_rate), str(args.alpha), str(args.beta), str(args.gamma), str(args.heads), str(args.num_layers), str(args.embedding_dim),str(args.samp_step)],
    )

    # train & test
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        epochs=args.epochs,
        monitor="jaccard_samples",
        lr=args.lr,
    )
