import math
import os
import sys
import json
from PIL import Image, ImageOps

import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel

import random
import argparse



class PriceTagDataset(Dataset):
    def __init__(self, split='train', use_processed_images=True):
        super().__init__()
        self.use_processed_images = use_processed_images
        self.data = []
        self.original_image_dir = f"./data_{split}/images"
        self.processed_image_dir = f"./processed_data_{split}/images"
        os.makedirs(self.processed_image_dir, exist_ok=True)

        for filename in os.listdir(f"./data_{split}/labels"):
            if filename.endswith('.json'):
                json_path = os.path.join(f"./data_{split}/labels", filename)
                with open(json_path, 'r') as json_file:
                    json_data = json.load(json_file)
                
                image_filename = os.path.splitext(filename)[0] + '.png'
                original_image_path = os.path.join(self.original_image_dir, image_filename)
                processed_image_path = os.path.join(self.processed_image_dir, image_filename)

                # Determine which image path to use based on the use_processed_images flag
                image_path = processed_image_path if self.use_processed_images else original_image_path

                # Process and save image only if using processed images and it doesn't exist
                if self.use_processed_images and not os.path.exists(processed_image_path):
                    with Image.open(original_image_path) as image:
                        processed_image = self.process_image(image)
                        processed_image.save(processed_image_path)
                        print(f"Processed and saved image at: {processed_image_path}")
                
                # Load the image from the determined path
                with Image.open(image_path) as image:
                    image = image.copy().convert('RGB')
                    self.data.append({
                        "image": image,
                        "qa": [
                            {
                                "question": "Find average price for goods in each labels in these images, respond with following json format: ```{\"name\": <good name, without brand name>, \"price\": <average price, per L(liter), kg(kilogram) or st(piece)>, \"unit\": <L, kg or st>}```",
                                "answer": json_data,
                            }
                        ]
                    })
                    
    def process_image(self, image):
        padding = (int(image.size[0]/2), int(image.size[1]/2))
        padded_image = ImageOps.expand(image, border=padding, fill='white')
        downscaled_image = padded_image.resize(padded_image.size, Image.LANCZOS)
        return downscaled_image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Model:
    def __init__(self, config):
        self.processed_flag = config

        self.tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2", revision=config.MD_REVISION, trust_remote_code=True)

        self.moondream = AutoModelForCausalLM.from_pretrained(
                # "./checkpoints/moondream-ft" if CONTINUE else "vikhyatk/moondream2",
                # "./checkpoints/moondream_raw",
                "./checkpoints/moondream_cache",
                revision=config.MD_REVISION,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if config.DEVICE == "cuda" else None,
                torch_dtype=config.DTYPE, device_map={"": config.DEVICE}
            )


class Model_Data:
    def __init__(self, config, tokenizer, moondream):
        self.tokenizer = tokenizer
        self.moondream = moondream
        
        self.datasets = {
            "train": PriceTagDataset("train", use_processed_images=config.processed_flag),
            "val": PriceTagDataset("validation", use_processed_images=config.processed_flag),
            "test": PriceTagDataset("test", use_processed_images=config.processed_flag),
        }
        
        self.dataloaders = {
            "train": DataLoader(
                self.datasets["train"],
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                collate_fn=self.collate_fn,
            ),
            "val": DataLoader(
                self.datasets["val"],
                batch_size=config.BATCH_SIZE,
                collate_fn=self.collate_fn,
            ),
            "test": DataLoader(
                self.datasets["test"],
                batch_size=config.BATCH_SIZE,
                collate_fn=self.collate_fn,
            ),
        }


    def collate_fn(self, batch):
        images = [sample['image'] for sample in batch]
        images = torch.stack(self.moondream.vision_encoder.preprocess(images))
        images = rearrange(images,
                        "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                        p1=14, p2=14)

        labels_acc = []
        tokens_acc = []

        for sample in batch:
            toks = [self.tokenizer.bos_token_id]
            labs = [-100] * (config.IMG_TOKENS + 1)

            for qa in sample['qa']:
                q_t = self.tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))

                a_t = self.tokenizer(
                    f" {qa['answer']}{config.ANSWER_EOS}",
                    add_special_tokens=False
                ).input_ids
                toks.extend(a_t)
                labs.extend(a_t)

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
            tokens_acc[i].extend([self.tokenizer.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        return (
            images.to(dtype=config.DTYPE),
            torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
            torch.stack([torch.tensor(a, dtype=torch.bool)
                        for a in attn_mask_acc]),
        )


def compute_loss(config, batch, moondream):
    images, tokens, labels, attn_mask = batch

    images = images.to(config.DEVICE)
    tokens = tokens.to(config.DEVICE)
    labels = labels.to(config.DEVICE)
    attn_mask = attn_mask.to(config.DEVICE)

    with torch.no_grad():
        img_embs = moondream.vision_encoder.encoder(images)
        img_embs = moondream.vision_encoder.projection(img_embs)

    tok_embs = moondream.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat(
        (tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

    outputs = moondream.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )

    return outputs.loss


def lr_schedule(config, step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * config.LR + 0.9 * config.LR * x / 0.1
    else:
        return 0.1 * config.LR + 0.9 * config.LR * (1 + math.cos(math.pi * (x - 0.1))) / 2



def train(config):
    
    model = Model(config=config)
    moondream = model.moondream
    tokenizer = model.tokenizer
    
    model_data = Model_Data(config=config, tokenizer=tokenizer, moondream=moondream)
    dataloaders = model_data.dataloaders
          
    moondream.text_model.train()
    moondream.text_model.transformer.gradient_checkpointing_enable()

    total_steps = config.EPOCHS * len(dataloaders["train"]) // config.GRAD_ACCUM_STEPS
    optimizer = torch.optim.Adam(
        [
            {"params": moondream.text_model.parameters()},
        ],
        lr=config.LR * 0.1,
        betas=(0.9, 0.95),
        eps=1e-6
    )

    if config.USE_WANDB:
        import wandb
        wandb.init(
            project="moondream-ft",
            config={
                "EPOCHS": config.EPOCHS,
                "BATCH_SIZE": config.BATCH_SIZE,
                "GRAD_ACCUM_STEPS": config.GRAD_ACCUM_STEPS,
                "LR": config.LR,
            }
        )

    i = 0
    for epoch in range(config.EPOCHS):
        for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{config.EPOCHS}"):
            i += 1

            loss = compute_loss(config, batch, moondream)
            loss.backward()

            if i % config.GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr = lr_schedule(config, i / config.GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if i % 10 == 0 and config.USE_WANDB:
                # Calculate validation loss
                val_loss = 0
                for val_batch in tqdm(dataloaders["val"], desc="Validation"):
                    with torch.no_grad():
                        val_loss += compute_loss(config, val_batch, moondream).item()
                val_loss /= len(dataloaders["val"])

            if config.USE_WANDB:
                wandb.log({
                    "loss/train": loss.item(),
                    "lr": optimizer.param_groups[0]['lr']
                } | ({"loss/val": val_loss} if i % 10 == 0 else {}))

    if config.USE_WANDB:
        wandb.finish()

    moondream.save_pretrained("checkpoints/moondream_cache")


def test(config):
    
    model = Model(config=config)
    moondream = model.moondream
    tokenizer = model.tokenizer
    
    model_data = Model_Data(config=config, tokenizer=tokenizer, moondream=moondream)
    dataloaders = model_data.dataloaders
    datasets = model_data.datasets
    
    random_indices = random.sample(range(len(datasets["train"])), 5)
    
    for idx in random_indices:
        train_data = datasets["train"][idx]
        enc_image = moondream.encode_image(train_data['image'])
        print(f"Data sample {idx}:")
        print("Keys in train_data:", train_data.keys())
        for key, value in train_data.items():
            print(f"{key}: {value}")
        print(moondream.answer_question(enc_image,
                                        "Find average price for goods in each labels in these images, respond with following json format: ```{\"name\": <good name, without brand name>, \"price\": <average price, per L(liter), kg(kilogram) or st(piece)>, \"unit\": <L, kg or st>}```", tokenizer))
        print("\n")
    
    
    for test_data in datasets["test"]:
        enc_image = moondream.encode_image(test_data['image'])
        # print("Label:", test_data['label'])
        print("Keys in test_data:", test_data['qa'])
        print(moondream.answer_question(enc_image,
                                        "Find average price for goods in each labels in these images, respond with following json format: ```{\"name\": <good name, without brand name>, \"price\": <average price, per L(liter), kg(kilogram) or st(piece)>, \"unit\": <L, kg or st>}```", tokenizer))



def parse_args():
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16

    CONTINUE = 1
    MD_REVISION = "2024-04-02"
    EPOCHS = 2
    # Number of samples to process in each batch. Set this to the highest value that doesn't cause an
    # out-of-memory error. Decrease it if you're running out of memory. Batch size 8 currently uses around
    # 15 GB of GPU memory during fine-tuning.
    BATCH_SIZE = 8
    # BATCH_SIZE = 1
    # Number of batches to process before updating the model. You can use this to simulate a higher batch
    # size than your GPU can handle. Set this to 1 to disable gradient accumulation.
    GRAD_ACCUM_STEPS = 1
    # Learning rate for the Adam optimizer. Needs to be tuned on a case-by-case basis. As a general rule
    # of thumb, increase it by 1.4 times each time you double the effective batch size.
    #
    # Source: https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    #
    # Note that we linearly warm the learning rate up from 0.1 * LR to LR over the first 10% of the
    # training run, and then decay it back to 0.1 * LR over the last 90% of the training run using a
    # cosine schedule.
    LR = 3e-5
    # Whether to use Weights and Biases for logging training metrics.
    USE_WANDB = True
    ANSWER_EOS = "<|endoftext|>"
    # Number of tokens used to represent each image.
    IMG_TOKENS = 729
    
    parser = argparse.ArgumentParser(description="Training configurations")
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--md_revision', type=str, default="2024-04-02", help='Model revision date for checkpoints')
    parser.add_argument('--continue_training', action='store_true', help='Continue training from last checkpoint')
    parser.add_argument('--answer_eos', type=str, default="", help='End of sequence token for answers')
    parser.add_argument('--img_tokens', type=int, default=729, help='Number of tokens used to represent each image')
    parser.add_argument('--processed_flag', action='store_true', help='Flag to use processed images')
    parser.add_argument('mode', choices=['train', 'test'], help='Mode to run the script in')
    
    args = parser.parse_args()
    return args


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32


class TrainConfig:
    def __init__(self, device, dtype, batch_size, epochs, lr, grad_accum_steps, use_wandb, md_revision, continue_training, answer_eos, img_tokens, processed_flag):
        self.DEVICE = device
        self.DTYPE = dtype
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LR = lr
        self.GRAD_ACCUM_STEPS = grad_accum_steps
        self.USE_WANDB = use_wandb
        self.MD_REVISION = md_revision
        self.CONTINUE_TRAINING = continue_training
        self.ANSWER_EOS = answer_eos
        self.IMG_TOKENS = img_tokens
        self.processed_flag = processed_flag


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig(
        device=get_device(),
        dtype=get_dtype(),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        grad_accum_steps=args.grad_accum_steps,
        use_wandb=args.use_wandb,
        md_revision=args.md_revision,
        continue_training=args.continue_training,
        answer_eos=args.answer_eos,
        img_tokens=args.img_tokens,
        processed_flag=args.processed_flag
    )

    if args.mode == "train":
        train(config)
    elif args.mode == "test":
        test(config)


# if __name__ == "__main__":
    # # call train or test according to the command line arguments
    # if len(sys.argv) > 1 and sys.argv[1] == "train":
    #     train()
    # elif len(sys.argv) > 1 and sys.argv[1] == "test":
    #     test()
