import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models.gradcam import run_gradcam, get_transforms, get_target_layer, LABELS
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch

def run_segmentation(model, img_path, device, output_dir=None, true_label=None, threshold=0.6):
    preprocess   = get_transforms()
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

    # Binary mask
    if pred == 1:
        binary_mask = (grayscale >= threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)
        center_mask = np.zeros((224, 224), dtype=np.uint8)
        cv2.ellipse(center_mask, (112, 112), (90, 110), 0, 0, 360, 255, -1)
        binary_mask = cv2.bitwise_and(binary_mask, center_mask)

        weak_localisation = False
        if cv2.countNonZero(binary_mask) == 0:
            weak_localisation = True
            cv2.ellipse(binary_mask, (112, 112), (70, 90), 0, 0, 360, 255, -1)
            print(f"⚠️  {os.path.basename(img_path)}: Weak localisation — fallback region used")
    else:
        binary_mask       = np.zeros((224, 224), dtype=np.uint8)
        weak_localisation = False

    # Red overlay
    overlay = (rgb_img * 255).astype(np.uint8).copy()
    if pred == 1:
        red_layer = np.zeros_like(overlay)
        red_layer[:, :, 0] = 255
        mask_bool = binary_mask.astype(bool)
        overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * red_layer[mask_bool]).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(pil_img);                          axes[0].set_title('Original');     axes[0].axis('off')
        axes[1].imshow(grayscale, cmap='jet');            axes[1].set_title('EigenCAM');     axes[1].axis('off')
        axes[2].imshow(binary_mask, cmap='gray');         axes[2].set_title('Binary Mask');  axes[2].axis('off')
        axes[3].imshow(overlay);                          axes[3].set_title('Segmented');    axes[3].axis('off')

        label_str = f"Pred: {LABELS[pred]} ({conf*100:.1f}%)"
        if true_label:
            correct = true_label.lower() == LABELS[pred].lower()
            label_str += f"  |  True: {true_label.capitalize()}"
            fig.suptitle(label_str, fontsize=13, color='green' if correct else 'red')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, os.path.basename(img_path)), dpi=150, bbox_inches='tight')
        plt.close()

    return LABELS[pred], conf, binary_mask, weak_localisation