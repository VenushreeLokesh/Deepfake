import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from models.gradcam import get_transforms, get_target_layer, LABELS
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def run_visualize(model, img_path, device, output_dir=None, true_label=None, threshold=0.6):
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

    heatmap = show_cam_on_image(rgb_img, grayscale, use_rgb=True)

    # Mask
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
    else:
        binary_mask       = np.zeros((224, 224), dtype=np.uint8)
        weak_localisation = False

    overlay = (rgb_img * 255).astype(np.uint8).copy()
    if pred == 1:
        red_layer = np.zeros_like(overlay)
        red_layer[:, :, 0] = 255
        mask_bool = binary_mask.astype(bool)
        overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * red_layer[mask_bool]).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    loc_score = float(grayscale[binary_mask > 0].mean()) if cv2.countNonZero(binary_mask) > 0 else 0.0

    fig = plt.figure(figsize=(18, 5), facecolor='#0f0f0f')
    gs  = GridSpec(1, 4, figure=fig, wspace=0.05)

    panels = [pil_img, heatmap, binary_mask, overlay]
    titles = ['Original', 'EigenCAM Heatmap', 'Binary Mask', 'Localisation Output']
    cmaps  = [None, None, 'gray', None]

    for i, (img_data, title, cmap) in enumerate(zip(panels, titles, cmaps)):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img_data, cmap=cmap)
        ax.set_title(title, color='white', fontsize=11, pad=8, fontweight='bold')
        ax.axis('off')

    pred_color = '#ff4444' if pred == 1 else '#44ff88'
    pred_text  = 'DEEPFAKE DETECTED' if pred == 1 else 'AUTHENTIC'
    conf_text  = f"Confidence: {conf*100:.1f}%"
    loc_text   = f"Localisation Score: {loc_score*100:.1f}%"
    if weak_localisation:
        loc_text += "  ⚠ Weak — fallback region used"

    true_text = ""
    if true_label:
        match     = true_label.lower() == LABELS[pred].lower()
        true_text = f"  |  GT: {true_label.upper()}  {'✓' if match else '✗'}"

    fig.text(0.5, 1.01,
             f"{pred_text}   {conf_text}   {loc_text}{true_text}",
             ha='center', va='bottom', fontsize=12,
             color=pred_color, fontweight='bold',
             transform=fig.transFigure)

    fig.text(0.5, -0.02, os.path.basename(img_path),
             ha='center', fontsize=9, color='#888888',
             transform=fig.transFigure)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, os.path.basename(img_path)),
                    dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    plt.close()