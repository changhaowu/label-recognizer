import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from wanda.lib.prune import (
    prune_wanda,
    prune_magnitude,
    prune_sparsegpt,
    prune_ablate,
    check_sparsity,
    find_layers,
)
from wanda.lib.eval import eval_ppl, eval_zero_shot
from torch.utils.data import DataLoader, Dataset
from moondream.moondream import Moondream, detect_device, LATEST_REVISION


print("torch", version("torch"))
print("transformers", version("transformers"))
print("accelerate", version("accelerate"))
print("# of gpus: ", torch.cuda.device_count())


# def get_llm(model_name, cache_dir="llm_weights"):
#     model = AutoModelForCausalLM.from_pretrained(
#         # model_name,
#         torch_dtype=torch.float16,
#         cache_dir=cache_dir,
#         low_cpu_mem_usage=True,
#         device_map="auto",
#     )

#     # model.seqlen = model.config.max_position_embeddings
#     model.text_model.seqlen = model.text_model.config.max_position_embeddings
#     return model


def get_llm(model_name, cache_dir="llm_weights"):

    DEVICE = "cuda"
    DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16

    model = Moondream.from_pretrained(
        pretrained_model_name_or_path=cache_dir,
        # revision=MD_REVISION,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
        torch_dtype=DTYPE,
        device_map={"": DEVICE},
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cache_dir,
        trust_remote_code=True,
    )
    # model.seqlen = model.config.max_position_embeddings
    model.to(DEVICE)
    # tokenizer.to(DEVICE)
    model.text_model.seqlen = model.text_model.config.max_position_embeddings

    return model, tokenizer


def stock_label_loader(
    model,
    tokenizer,
    args,
):
    """
    Returns dataloaders for training, validation, and testing.

    Args:
        model (object): The model object.
        tokenizer (object): The tokenizer object.
        args (object): The arguments object.

    Returns:
        dict: A dictionary containing dataloaders for training, validation, and testing.
    """

    from PIL import Image
    import json

    ANSWER_EOS = "<|endoftext|>"
    IMG_TOKENS = 729
    DEVICE = "cuda"

    PROMPT = """
        Analyze the text in the provided image and extract the product name, price, and unit. Ensure the product name is accurately read from the image and not assumed. Follow these instructions precisely:

        1. Identify the product name, typically a recognizable item name found in the image.
        2. Determine the unit of measurement, which could be "kg", "L", or "st". If "kg" or "L" occur with "st", prefer to read "kg" or "L" rather than "st".
        3. Detect the price, associated with the identified unit (kg, L, or st). If "kg" or "L" are present in the image, use the price closest to these units. If neither "kg" nor "L" are present, then search for "st" and its respective price.

        Respond exclusively in the JSON format below, with no additional text or explanations. Include your response within `{ }`, then conclude your response with "<|endoftext|>". 
        The output should only be one combination of the most likely product name, price, and unit. Example format:

        ```json
        {
            "name": "Ryggfilé Alaska Pollock",
            "avg_price": "133",
            "unit": "kg"
        }
        <|endoftext|>
        """

    class PriceTagDataset(Dataset):

        def __init__(self, base_dir=None, split="train"):
            super().__init__()
            data = []

            if base_dir is None:
                base_dir = "./"

            labels_dir = os.path.join(base_dir, f"data_{split}", "labels")
            images_dir = os.path.join(base_dir, f"data_{split}", "images")

            for filename in os.listdir(labels_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(labels_dir, filename)
                    filename_without_extension = os.path.splitext(filename)[0]
                    image_filename = filename_without_extension + ".png"
                    image_path = os.path.join(images_dir, image_filename)

                    with open(file_path, "r") as json_file:
                        json_data = json.load(json_file)  # Read JSON data
                        with Image.open(image_path) as image:
                            data.append(
                                {
                                    "image": image.convert("RGB"),
                                    # "image": image.convert("L"),
                                    "qa": [
                                        {
                                            "question": PROMPT,
                                            "answer": json_data,
                                        }
                                    ],
                                }
                            )
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    datasets = {
        "train": PriceTagDataset(split="train"),
        "val": PriceTagDataset(split="validation"),
        "test": PriceTagDataset(split="test"),
    }

    def collate_fn(batch):
        labels_acc = []
        tokens_acc = []

        for sample in batch:
            image = sample["image"]
            image = model.vision_encoder.preprocess(image)
            img_embs = model.vision_encoder(image.unsqueeze(0)).squeeze(0).to(DEVICE)

            # toks = [tokenizer.bos_token_id]
            toks = [
                model.text_model.get_input_embeddings()(
                    torch.tensor([tokenizer.bos_token_id], device=DEVICE)
                )
            ]
            # labs = [-100] * (IMG_TOKENS + 1)
            toks.append(img_embs)
            labs = [-100] * (
                img_embs.size(0) + 1
            )  # Adjusted to match image embedding size

            for qa in sample["qa"]:
                q_t = tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False,
                ).input_ids
                q_t = torch.tensor(q_t).to(DEVICE)
                q_t_embeds = model.text_model.get_input_embeddings()(q_t)
                toks.append(q_t_embeds)
                # labs.extend([-100] * len(q_t))
                labs.extend([-100] * q_t_embeds.size(0))

                a_t = tokenizer(
                    f" {qa['answer']}{ANSWER_EOS}", add_special_tokens=False
                ).input_ids
                a_t = torch.tensor(a_t).to(DEVICE)
                a_t_embeds = model.text_model.get_input_embeddings()(a_t)
                toks.append(a_t_embeds)
                labs.extend(a_t.tolist())

            # # Debug: Print the dimensions of each tensor in toks
            # for idx, t in enumerate(toks):
            #     print(f"toks[{idx}] shape: {t.shape}")

            toks = torch.cat(toks, dim=0)
            tokens_acc.append(toks)
            labels_acc.append(labs)

        max_len = -1
        for labels in labels_acc:
            max_len = max(max_len, len(labels))

        attn_mask_acc = []

        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i

            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i] = torch.cat(
                [
                    tokens_acc[i],
                    torch.zeros((pad_i, tokens_acc[i].size(1)), device=DEVICE),
                ],
                dim=0,
            )
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        return (
            torch.stack(tokens_acc),
            # torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        )

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.nsamples,
            shuffle=True,
            collate_fn=collate_fn,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.nsamples,
            collate_fn=collate_fn,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.nsamples,
            collate_fn=collate_fn,
        ),
    }

    return dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="LLaMA model")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for sampling the calibration data.",
    )
    parser.add_argument(
        # "--nsamples", type=int, default=128, help="Number of calibration samples."
        "--nsamples",
        type=int,
        default=16,
        help="Number of calibration samples.",
    )
    parser.add_argument(
        "--sparsity_ratio", type=float, default=0.5, help="Sparsity level"
    )
    parser.add_argument(
        "--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"]
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        choices=[
            "magnitude",
            "wanda",
            "sparsegpt",
            "ablate_mag_seq",
            "ablate_wanda_seq",
            "ablate_mag_iter",
            "ablate_wanda_iter",
            "search",
        ],
    )
    parser.add_argument(
        "--cache_dir", default="checkpoints/moondream-ft_lr_3e-06_epoch_10", type=str
    )
    parser.add_argument(
        "--use_variant",
        action="store_true",
        help="whether to use the wanda variant described in the appendix",
    )
    parser.add_argument("--save", type=str, default=None, help="Path to save results.")
    parser.add_argument(
        "--save_model",
        type=str,
        default="checkpoints/moondream-ft_lr_3e-06_epoch_10_pruned",
        help="Path to save the pruned model.",
    )

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert (
            args.sparsity_ratio == 0.5
        ), "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    # model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model, tokenizer = get_llm(args.model, args.cache_dir)
    model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    # if (
    #     "30b" in args.model or "65b" in args.model
    # ):  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
    #     device = model.hf_device_map["lm_head"]
    print("use device ", device)
    dataLoader = stock_label_loader(model, tokenizer, args)["train"]

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(
                args,
                model,
                tokenizer,
                device,
                prune_n=prune_n,
                prune_m=prune_m,
                dataloader=dataLoader,
            )
        elif args.prune_method == "magnitude":
            prune_magnitude(
                args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m
            )
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(
                args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m
            )
        elif "ablate" in args.prune_method:
            prune_ablate(
                args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m
            )

    ################################################################
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(
            f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}",
            file=f,
            flush=True,
        )

    if args.eval_zero_shot:
        accelerate = False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate = True

        task_list = [
            "boolq",
            "rte",
            "hellaswag",
            "winogrande",
            "arc_easy",
            "arc_challenge",
            "openbookqa",
        ]
        num_shot = 0
        results = eval_zero_shot(
            args.model, model, tokenizer, task_list, num_shot, accelerate
        )
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == "__main__":
    main()
