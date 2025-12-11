import torch
import torch.nn as nn
import contextlib
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from lavis.models import Blip2Qformer, load_model_and_preprocess
from paper_codes.Dataloader_custom import ImageMOSDataset
import random
import itertools
import torch.nn.functional as F
import pandas as pd


class GroundingQFormer(nn.Module):
    def __init__(self, device, outputs, value_dim=768):
        super().__init__()
        self.device = device
        self.model: Blip2Qformer = outputs[0]
        self.value_dim = value_dim

        train_params = ("Qformer", "query_tokens", "itm_head")

        # Set the model parameters to trainable based on the train_params
        for name, param in self.model.named_parameters():
            if name.startswith(train_params):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.q_former_average_proj = nn.Linear(value_dim, value_dim)
        self.mos_score_layer = nn.Linear(value_dim, 1)  # Linear layer to output MOS score

        # Ensure final layer for MOS score is trainable
        for param in self.mos_score_layer.parameters():
            param.requires_grad = True

    def extract_vision_features(self, images):
        return_values = {}

        with torch.no_grad():
            with self._maybe_autocast():
                image_embeds_frozen = self.model.ln_vision(self.model.visual_encoder(images))

            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)

        # Prepare inputs for QFormer
        query_tokens = self.model.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)

        query_output = self.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=True
        )

        vl_embeddings = query_output.last_hidden_state  # vl_embeddings

        # Pass vl_embeddings through a linear layer and take the mean of the 32 learned query tokens as the value
        vl_embeddings = self.q_former_average_proj(vl_embeddings)
        vl_embeddings_averaged = vl_embeddings.mean(dim=1, keepdim=True)
        values = vl_embeddings_averaged[:, 0, :]

        # Compute MOS score using the final linear layer
        mos_scores = self.mos_score_layer(values)

        # Store the results in a dictionary
        return_values['values'] = values
        return_values['SA'] = query_output.attentions
        return_values['CA'] = query_output.cross_attentions
        return_values['vision_encoder_outputs'] = image_embeds_frozen
        return_values['mos_scores'] = mos_scores

        return return_values

    def _maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(device_type="cuda", dtype=dtype)
        else:
            return contextlib.nullcontext()


# Linear CKA loss function
def linear_CKA_torch(X, Y):
    """Compute linear CKA between two feature matrices using PyTorch."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    dot_product_similarity = torch.trace((X @ Y.T) @ (Y @ X.T))
    norm_x = torch.norm(X @ X.T, p='fro')
    norm_y = torch.norm(Y @ Y.T, p='fro')
    return dot_product_similarity / (norm_x * norm_y)


# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images1, images2, target_scores, *_ in dataloader:
            images1, images2, target_scores = images1.to(device), images2.to(device), target_scores.to(device)
            scores1 = model.extract_vision_features(images1)['mos_scores']
            scores2 = model.extract_vision_features(images2)['mos_scores']
            predictions = torch.sigmoid(scores1 - scores2) > 0.5
            targets = (target_scores > 0.5).float().view(-1, 1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    print(f"Accuracy: {100 * correct / total:.2f}%")
    return correct / total


# Iteration-based training function with even/odd switching
def train_iteration_based(model, band_csv_paths, transform, optimizer, criterion, device, num_iterations, val_csv, val_transform, save_path, run_number):
    best_val_loss = float('inf')

    def get_cyclic_dataloader_for_all_bands():
        run_csv_path = f"join_setting2/content/pan/run{run_number}/_train{run_number}.csv"  # CSV path for the current run
        # print(f"Reading CSV for run {run_number} from {run_csv_path}")
        # Pass the combined CSV DataFrame to your dataset
        dataset = ImageMOSDataset(csv_path=run_csv_path, transform=transform)  # Assuming ImageMOSDataset can handle a DataFrame
        return itertools.cycle(DataLoader(dataset, batch_size=16, shuffle=False))

    # Cyclic data loader for odd iterations (bandwise sampling)
    def get_cyclic_dataloader_bandwise():
        band_data = {}
        for band in band_csv_paths:
            dataset = ImageMOSDataset(csv_path=band_csv_paths[band], transform=transform)
            band_data[band] = DataLoader(dataset, batch_size=8, shuffle=True)
        return band_data

    for step in range(1, num_iterations + 1):
        if not step % 4 == 0:  # Even iterations (use cyclic with all bands)
            dataloader = get_cyclic_dataloader_for_all_bands()
            loss_term = "bce"
            band_data = None  
        else:  # Odd iterations (use cyclic bandwise)
            band_data = get_cyclic_dataloader_bandwise()
            dataloader = None  # Not needed for odd steps
            loss_term = "cka"

        optimizer.zero_grad()

        if loss_term == "bce":
            images1, images2, scores, _, _ = next(iter(dataloader))
            images1, images2, scores = images1.to(device), images2.to(device), scores.to(device)

            scores1 = model.extract_vision_features(images1)['mos_scores']
            scores2 = model.extract_vision_features(images2)['mos_scores']

            bce_loss = criterion(scores1 - scores2, scores.unsqueeze(-1))
            print(f"BCE Loss at step {step}: {bce_loss.item():.4f}")
            bce_loss.backward()

        elif loss_term == "cka":
            # Randomly select two different bands
            band1, band2 = random.sample(list(band_data.keys()), 2)
            
            # Sample one batch from each band
            images1, images2, scores, _, _ = next(iter(band_data[band1]))  # Batch from band1
            images1, images2, scores = images1.to(device), images2.to(device), scores.to(device)
            values_a = model.extract_vision_features(images1)['values']
            values_b = model.extract_vision_features(images2)['values']

            # Now sample from the second band (band2)
            images1_band2, images2_band2, scores_band2, _, _ = next(iter(band_data[band2]))  # Batch from band2
            images1_band2, images2_band2, scores_band2 = images1_band2.to(device), images2_band2.to(device), scores_band2.to(device)
            values_a_band2 = model.extract_vision_features(images1_band2)['values']
            values_b_band2 = model.extract_vision_features(images2_band2)['values']
            n = linear_CKA_torch(values_a, values_b_band2)
            cka_loss = 1 -   n
            print(f"CKA at step {step}: {n}")
            cka_loss.backward()

        optimizer.step()
        print(f"Step [{step}/{num_iterations}] - Loss Term: {loss_term}")

        # Validation every 50 steps
        if step % 50 == 0:
            val_dataset = ImageMOSDataset(csv_path=val_csv, transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            val_acc = evaluate_model(model, val_loader, device)
            val_loss = 1 - val_acc
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"Saved new best model at step {step} with val loss {val_loss:.4f}")

# ---- Main Execution ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Setup optimizer and loss function
criterion = nn.BCEWithLogitsLoss()
band_csv_paths = {}
run_folders = ['1', '2', '3', '4', '5']
for run in run_folders:
    band_csv_paths[run] = {}  # Initialize as an empty dictionary
    for i in range(14,15):  # Loop over the band indices
        band_csv_paths[run][f"band{i}"] = f"data_setting2/content/new_data_for_benchmark/band{i}/run{run}/_train{run}.csv"
    band_csv_paths[run]["pan"] = f"data_setting2/content/new_data_for_benchmark/pan/run{run}/_train{run}.csv"

# Final part of the training loop and execution
for run in run_folders:
    # Load the model and set up the optimizer for each run
    outputs = load_model_and_preprocess("blip2_feature_extractor", model_type="pretrain", device=device)
    model = GroundingQFormer(device, outputs).to(device)
    model_path = f'Notokens_s2_3/run{run}/best_model.pth'
    model.load_state_dict(torch.load(model_path), strict=False)
    print(f'model loaded for run{run} with path {model_path}')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    val_csv = f"join_setting2/content/pan/run{run}/val{run}.csv"
    save_path = f"similarity_loss_hybrid/{run}_best_model.pth"

    # Start training for the current run
    train_iteration_based(
        model=model,
        band_csv_paths=band_csv_paths[run], 
        transform=transform,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_iterations=500,  # Set the number of iterations for training
        val_csv=val_csv,  # Validation data CSV
        val_transform=transform,  # Transformation for validation data
        save_path=save_path,  # Path to save the best model
        run_number = run
    )

