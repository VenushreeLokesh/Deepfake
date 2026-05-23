import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

LABELS = {0: 'Real', 1: 'Fake'}

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def get_target_layer(model):
    return [model.features[-3]]

def run_gradcam(model, img_path, device, output_dir=None, true_label=None, threshold=0.6):
    preprocess  = get_transforms()
    target_layer = get_target_layer(model)

    pil_img = Image.open(img_path).convert('RGB').resize((224, 224))
    rgb_img = np.array(pil_img, dtype=np.float32) / 255.0
    input_t = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_t)
        probs  = torch.softmax(logits, dim=1)
        pred   = probs.argmax(dim=1).item()
        conf   = probs[0, pred].item()

    with EigenCAM(model=model, target_layers=target_layer) as cam:
        targets   = [ClassifierOutputTarget(pred)]
        grayscale = cam(input_tensor=input_t, targets=targets)[0]

    heatmap = show_cam_on_image(rgb_img, grayscale, use_rgb=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(pil_img);               axes[0].set_title('Original');        axes[0].axis('off')
        axes[1].imshow(grayscale, cmap='jet'); axes[1].set_title('EigenCAM Heatmap');axes[1].axis('off')
        axes[2].imshow(heatmap);               axes[2].set_title('Overlay');          axes[2].axis('off')

        label_str = f"Pred: {LABELS[pred]} ({conf*100:.1f}%)"
        if true_label:
            correct = true_label.lower() == LABELS[pred].lower()
            label_str += f"  |  True: {true_label.capitalize()}"
            fig.suptitle(label_str, fontsize=13, color='green' if correct else 'red')
        else:
            fig.suptitle(label_str, fontsize=13)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, os.path.basename(img_path)), dpi=150, bbox_inches='tight')
        plt.close()

    return LABELS[pred], conf, grayscale