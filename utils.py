import os
import torch

import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS



def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    mkdir(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, f"model_{'best' if best else 'latest'}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, model_path)
    return model_path


def initialize_model(args):
    target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16
    )

    for param in model.parameters():
        param.requires_grad = False

    unfrozen_count = 0
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear)
            and "vision_tower" not in name
            and any(name.endswith(suffix) for suffix in target_suffixes)
        ):
            for param in module.parameters():
                param.requires_grad = True
                unfrozen_count += param.numel()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"{trainable_params:,}  parameters out of {total_params:,} ")
    return model.to(args.device), tokenizer


def load_model_from_checkpoint(args, model_path):
    model, tokenizer = initialize_model(args)
    if not os.path.exists(model_path):
        return model, tokenizer

    checkpoint = torch.load(model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, tokenizer


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result


def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    elif args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters, lr=args.learning_rate, momentum=0.9
        )
    else:
        raise ValueError(f"misisng: {args.optimizer_type}")

    return optimizer


def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError