import torch
import torch.nn as nn
import contextlib
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from lavis.models import Blip2Qformer, load_model_and_preprocess
from Dataloader_custom import ImageMOSDataset
import numpy as np

class GroundingQFormer(nn.Module):
    def __init__(self, device, outputs, value_dim=768):
        super().__init__()
        self.device = device
        self.model: Blip2Qformer = outputs[0]
        self.value_dim = value_dim

        train_params = ("Qformer", "query_tokens", "itm_head")

        for name, param in self.model.named_parameters():
            if name.startswith(train_params):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.q_former_average_proj = nn.Linear(value_dim, value_dim)
        self.mos_score_layer = nn.Linear(value_dim, 1)

        for param in self.mos_score_layer.parameters():
            param.requires_grad = True

    def extract_vision_features(self, images):
        return_values = {}
        with torch.no_grad():
            with self._maybe_autocast():
                image_embeds_frozen = self.model.ln_vision(self.model.visual_encoder(images))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)

        query_tokens = self.model.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)
        query_output = self.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=True
        )

        vl_embeddings = self.q_former_average_proj(query_output.last_hidden_state)
        values = vl_embeddings.mean(dim=1, keepdim=True)[:, 0, :]
        mos_scores = self.mos_score_layer(values)

        return_values['values'] = values
        return_values['SA'] = query_output.attentions
        return_values['CA'] = query_output.cross_attentions
        return_values['vision_encoder_outputs'] = image_embeds_frozen
        return_values['mos_scores'] = mos_scores

        return return_values

    def _maybe_autocast(self, dtype=torch.float16):
        return torch.amp.autocast(device_type="cuda", dtype=dtype) if self.device != torch.device("cpu") else contextlib.nullcontext()


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images1, images2, target_scores, _, _ in dataloader:
            images1, images2, target_scores = images1.to(device), images2.to(device), target_scores.to(device)
            outputs1 = model.extract_vision_features(images1)
            outputs2 = model.extract_vision_features(images2)
            mos_scores_diff = outputs1['mos_scores'] - outputs2['mos_scores']
            probabilities = torch.sigmoid(mos_scores_diff)
            binary_predictions = (probabilities > 0.5).float()
            binary_targets = (target_scores > 0.5).float().view(-1, 1)
            correct += (binary_predictions == binary_targets).sum().item()
            total += target_scores.size(0)

    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, total


def test_model_for_split(model_path, test_csv_path, device, transform):
    outputs = load_model_and_preprocess("blip2_feature_extractor", model_type="pretrain", device=device)
    model = GroundingQFormer(device, outputs).to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    print('loaded model from path', model_path)
    test_dataset = ImageMOSDataset(csv_path=test_csv_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return evaluate_model(model, test_dataloader, device)


def test_all_splits_for_band(band, model_save_dir, device):
    print(f"\n========== Testing for Band: {band} ==========")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    accuracies = []
    total_correct = 0
    total_samples = 0

    for split in range(1, 16):
        print(f"Testing Split {split}...")
        model_path = f"{model_save_dir}/run{split}/best_model.pth"
        test_csv_path = f"data_setting1/{band}/run{split}/test{split}.csv"
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        try:
            accuracy, num_samples = test_model_for_split(model_path, test_csv_path, device, transform)
        except Exception as e:
            print(f"Error in Split {split}: {e}")
            continue
        print(f"Split {split} Accuracy: {accuracy:.2f}%")
        accuracies.append(accuracy)
        total_correct += accuracy * num_samples / 100
        total_samples += num_samples

    weighted_avg = (total_correct / total_samples * 100) if total_samples > 0 else 0
    median_acc = np.median(accuracies) if accuracies else 0

    print(f"\nBand: {band} | Median Accuracy: {median_acc:.2f}% | Weighted Average Accuracy: {weighted_avg:.2f}%\n")
    return median_acc

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_save_dir = "Notokens_s1"
bands = ['pan', 'band1', 'band2', 'band3', 'band4']
all_medians = []

# Run for each band
for band in bands:
    median = test_all_splits_for_band(band, model_save_dir, device)
    all_medians.append((band, median))

print("\n========== Median Accuracies by Band ==========")
for band, median in all_medians:
    print(f"{band}: {median:.2f}%")
