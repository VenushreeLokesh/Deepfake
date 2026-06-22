import io
import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Config ────────────────────────────────────────────────────────────────
WEIGHTS_GAN   = 'models/deepfake_classifier.pth'  # GAN faces, 96%
WEIGHTS_GAN_FT = 'models/deepfake_classifier_finetuned.pth'
WEIGHTS_CELEB = 'models/deepfake_celebdf.pth'       # video swaps, 90%
LABELS        = {0: 'Real', 1: 'Fake'}
THRESHOLD     = 0.6
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load both models at startup ────────────────────────────────────────────
def _load(weights_path, strip_prefix=False):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    state = torch.load(weights_path, map_location=DEVICE)
    if strip_prefix:
        state = {k.replace('model.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

gan_model   = _load(WEIGHTS_GAN,   strip_prefix=True)
gan_model_ft  = _load(WEIGHTS_GAN_FT, strip_prefix=True)   # same strip_prefix, since it was fine-tuned from the same checkpoint
celeb_model = _load(WEIGHTS_CELEB, strip_prefix=False)
print(f"✅ Both models loaded on {DEVICE}")

# ── Transforms ────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Main inference pipeline ───────────────────────────────────────────────
def run_inference(image_bytes: bytes):
    # Load image
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    rgb_img = np.array(pil_img, dtype=np.float32) / 255.0
    input_t = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    # Ensemble prediction — average both models
    with torch.no_grad():
        probs_gan   = torch.softmax(gan_model(input_t),   dim=1)
        probs_gan_ft = torch.softmax(gan_model_ft(input_t), dim=1)
        probs_celeb = torch.softmax(celeb_model(input_t), dim=1)
        
        fake_prob = max(probs_gan[0,1].item(), probs_gan_ft[0,1].item(), probs_celeb[0,1].item())


        pred = 1 if fake_prob >= 0.5 else 0
        conf = fake_prob if pred == 1 else (1 - fake_prob)

        print(f"GAN model   — real: {probs_gan[0,0]:.3f} fake: {probs_gan[0,1]:.3f}")
        print(f"Celeb model — real: {probs_celeb[0,0]:.3f} fake: {probs_celeb[0,1]:.3f}")
        print(f"Ensemble (max) — fake: {fake_prob:.3f}")
        print(f"Prediction: {LABELS[pred]} ({conf*100:.1f}%)")

    # EigenCAM — use celeb model for localisation (better on real faces)
    target_layer = [celeb_model.features[-3]]
    with EigenCAM(model=celeb_model, target_layers=target_layer) as cam:
        grayscale = cam(input_tensor=input_t,
                        targets=[ClassifierOutputTarget(pred)])[0]

    heatmap = show_cam_on_image(rgb_img, grayscale, use_rgb=True)

    # Binary mask
    weak_localisation = False
    if pred == 1:
        binary_mask = (grayscale >= THRESHOLD).astype(np.uint8) * 255
        kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)
        center_mask = np.zeros((224, 224), dtype=np.uint8)
        cv2.ellipse(center_mask, (112, 112), (90, 110), 0, 0, 360, 255, -1)
        binary_mask = cv2.bitwise_and(binary_mask, center_mask)
        if cv2.countNonZero(binary_mask) == 0:
            weak_localisation = True
            cv2.ellipse(binary_mask, (112, 112), (70, 90), 0, 0, 360, 255, -1)
    else:
        binary_mask = np.zeros((224, 224), dtype=np.uint8)

    # Red overlay
    overlay = (rgb_img * 255).astype(np.uint8).copy()
    if pred == 1:
        red_layer          = np.zeros_like(overlay)
        red_layer[:, :, 0] = 255
        mask_bool          = binary_mask.astype(bool)
        overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * red_layer[mask_bool]).astype(np.uint8)
        contours, _        = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    # Localisation score
    loc_score = float(grayscale[binary_mask > 0].mean()) if cv2.countNonZero(binary_mask) > 0 else 0.0

    # Build output figure
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
        loc_text += "  ⚠ Weak"

    fig.text(0.5, 1.01,
             f"{pred_text}   {conf_text}   {loc_text}",
             ha='center', va='bottom', fontsize=12,
             color=pred_color, fontweight='bold',
             transform=fig.transFigure)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)

    return {
        'label':              LABELS[pred],
        'confidence':         round(conf * 100, 2),
        'localisation_score': round(loc_score * 100, 2),
        'weak_localisation':  weak_localisation,
        'result_image':       buf.read(),
    }