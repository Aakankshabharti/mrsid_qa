import torch
import torch.nn as nn
import contextlib
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from lavis.models import Blip2Qformer, load_model_and_preprocess
from Dataloader_custom import ImageMOSDataset

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


def train(model, dataloader, optimizer, criterion, device, num_epochs, save_path, val_dataloader, best_val_loss):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images1, images2, target_scores, paths1, paths2 in dataloader:
            images1, images2, target_scores = images1.to(device), images2.to(device), target_scores.float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs1 = model.extract_vision_features(images1)
            mos_scores1 = outputs1['mos_scores']
            outputs2 = model.extract_vision_features(images2)
            mos_scores2 = outputs2['mos_scores']
            mos_scores = (mos_scores1 - mos_scores2)
            target_scores = target_scores.view(-1, 1)
            # Compute the loss
            loss = criterion(mos_scores, target_scores)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        # Print the loss for the current epoch
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")
        val_loss = evaluate_model_loss(model, val_dataloader, criterion, device)
        print(f"Validation Loss after epoch {epoch + 1}: {val_loss:.4f}")

        # Save the model if the validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epoch_save_path = os.path.join(save_path, f"best_model.pth")
            torch.save(model.state_dict(), epoch_save_path)
            print(f"Best model saved at {epoch_save_path}.")
            global best_token
            global best_vis
    return best_val_loss
        

def evaluate_model_loss(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation
        for images1, images2, target_scores, paths1, paths2 in dataloader:
            images1, images2, target_scores = images1.to(device), images2.to(device), target_scores.float().to(device)

            # Forward pass
            outputs1 = model.extract_vision_features(images1)
            mos_scores1 = outputs1['mos_scores']
            outputs2 = model.extract_vision_features(images2)
            mos_scores2 = outputs2['mos_scores']
            mos_scores = (mos_scores1 - mos_scores2)

            target_scores = target_scores.view(-1, 1)

            # Compute the loss
            loss = criterion(mos_scores, target_scores)
            running_loss += loss.item()

    # Return average validation loss
    return running_loss / len(dataloader)



def evaluate_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images1, images2, target_scores, paths1, paths2 in dataloader:
            images1, images2, target_scores = images1.to(device), images2.to(device), target_scores.float().to(device)
            # Forward pass
            outputs1 = model.extract_vision_features(images1)
            mos_scores1 = outputs1['mos_scores']
            outputs2 = model.extract_vision_features(images2)
            mos_scores2 = outputs2['mos_scores']
            mos_scores_diff = (mos_scores1 - mos_scores2)
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(mos_scores_diff)
            
            # Convert probabilities to binary predictions with a threshold of 0.5
            binary_predictions = (probabilities > 0.5).float()
            
            # Ensure target scores are binary for comparison
            binary_targets = (target_scores > 0.5).float().view(-1, 1)  

            # Count correct predictions
            correct += (binary_predictions == binary_targets).sum().item()
            total += target_scores.size(0)

    # Avoid division by zero
    accuracy = (correct / total * 100) if total > 0 else 0  # Convert to percentage
    print(f"Correct: {correct}, Total: {total}, Accuracy: {accuracy:.2f}%")
    return accuracy


# Define device and load model and preprocessing outputs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_save_folder = "Notokens_s2_3"
os.makedirs(base_save_folder, exist_ok=True)
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
outputs = load_model_and_preprocess("blip2_feature_extractor", model_type="pretrain", device=device)

# Optimizer (only learnable parameters are trainable)
for run in range(1,6):
    custom_qformer = GroundingQFormer(device, outputs).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, custom_qformer.parameters()), lr=1e-5)
    overall_best_val_loss = float('inf')
    bands = ['all_bands']
    for band in bands: 
        print(f"Training for Band {band}...")

        # Define CSV and image folder paths for the current band
        train_csv_path = f'/media/ece/8TB_NithinC_v2/Aakansha/Notokens_s1/join_setting1/content/all_bands/run{run}/train{run}.csv'
        val_csv_path = f'/media/ece/8TB_NithinC_v2/Aakansha/Notokens_s1/join_setting1/content/all_bands/run{run}/val{run}.csv'
        criterion = nn.BCEWithLogitsLoss()

        # Initialize datasets and dataloaders
        train_dataset = ImageMOSDataset(csv_path=train_csv_path, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        val_dataset = ImageMOSDataset(csv_path=val_csv_path, transform=transform)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        band_save_folder = os.path.join(base_save_folder, f'run{run}')
        os.makedirs(band_save_folder, exist_ok=True)

        # Train the model and save the best checkpoint for the current band
        overall_best_val_loss = train(
            custom_qformer,
            train_dataloader,
            optimizer,
            criterion,
            device,
            num_epochs=10,
            save_path=band_save_folder,
            val_dataloader=val_dataloader,
            best_val_loss=overall_best_val_loss,
        )

print("Training complete for all bands.")
