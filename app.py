import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import gradio as gr
from PIL import Image, ImageOps

# =========================
# 1) ĐỊNH NGHĨA MÔ HÌNH
# =========================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# =========================
# 2) NẠP MÔ HÌNH
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(device)

MODEL_PATH = "models/lenet_mnist_model.pth"
try:
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
except TypeError:
    # fallback cho PyTorch cũ không hỗ trợ weights_only
    state_dict = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(state_dict)
model.eval()


# =========================
# 3) HẰNG SỐ CHUẨN HÓA MNIST
# =========================
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
])


# =========================
# 4) FGSM ATTACK
# =========================
def fgsm_attack(image_01, epsilon, grad):
    # image_01: ảnh ở miền [0,1]
    # grad: gradient theo input
    perturbed = image_01 + epsilon * grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed


# =========================
# 5) TIỆN ÍCH XỬ LÝ ẢNH
# =========================
def shift_array(arr, shift_x, shift_y):
    """
    Dịch mảng 2D theo shift_x, shift_y.
    shift_x > 0: dịch sang phải
    shift_y > 0: dịch xuống dưới
    """
    h, w = arr.shape
    shifted = np.zeros_like(arr)

    src_x1 = max(0, -shift_x)
    src_x2 = min(w, w - shift_x)
    dst_x1 = max(0, shift_x)
    dst_x2 = min(w, w + shift_x)

    src_y1 = max(0, -shift_y)
    src_y2 = min(h, h - shift_y)
    dst_y1 = max(0, shift_y)
    dst_y2 = min(h, h + shift_y)

    if src_x1 < src_x2 and src_y1 < src_y2:
        shifted[dst_y1:dst_y2, dst_x1:dst_x2] = arr[src_y1:src_y2, src_x1:src_x2]

    return shifted


def extract_raw_grayscale(input_img):
    """
    Nhận đầu vào từ Gradio Sketchpad / upload ảnh / numpy array
    và trả về ảnh grayscale uint8.
    """
    if input_img is None:
        return None

    # Gradio có thể trả dict
    if isinstance(input_img, dict):
        if "composite" in input_img:
            input_img = input_img["composite"]
        elif "image" in input_img:
            input_img = input_img["image"]
        else:
            input_img = list(input_img.values())[0]

    arr = np.array(input_img)

    if arr.ndim == 3 and arr.shape[2] == 4:
        # RGBA
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]

        # Nếu alpha thể hiện nét vẽ rõ thì ưu tiên dùng alpha
        if alpha.max() > 0 and np.mean(alpha < 250) > 0.01:
            gray = alpha.astype(np.uint8)
        else:
            gray = np.array(Image.fromarray(rgb).convert("L"), dtype=np.uint8)

    elif arr.ndim == 3:
        gray = np.array(Image.fromarray(arr).convert("L"), dtype=np.uint8)

    else:
        gray = arr.astype(np.uint8)

    return gray


def preprocess_like_mnist(input_img):
    """
    Chuyển ảnh người dùng vẽ về dạng gần MNIST:
    - chữ trắng nền đen
    - bỏ nhiễu yếu
    - crop sát digit
    - resize digit vào box 20x20, giữ tỉ lệ
    - đặt vào canvas 28x28
    - center theo center-of-mass
    """
    gray = extract_raw_grayscale(input_img)

    if gray is None:
        blank = Image.new("L", (28, 28), 0)
        return blank, np.array(blank.resize((280, 280), Image.Resampling.NEAREST))

    # Nếu nền sáng thì invert -> chữ trắng nền đen
    if gray.mean() > 127:
        gray = 255 - gray

    # Kéo tương phản nhẹ
    gray = np.array(ImageOps.autocontrast(Image.fromarray(gray, mode="L")), dtype=np.uint8)

    # Bỏ nhiễu yếu nhưng vẫn giữ sắc độ
    max_val = int(gray.max())
    threshold = max(20, int(max_val * 0.25))
    gray = gray.copy()
    gray[gray < threshold] = 0

    # Tìm bounding box của vùng có nét
    ys, xs = np.nonzero(gray)
    if len(xs) == 0 or len(ys) == 0:
        blank = Image.new("L", (28, 28), 0)
        return blank, np.array(blank.resize((280, 280), Image.Resampling.NEAREST))

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    digit = Image.fromarray(gray[y1:y2, x1:x2], mode="L")

    # Resize digit vào box 20x20 theo chuẩn gần MNIST
    w, h = digit.size
    if w > h:
        new_w = 20
        new_h = max(1, round(h * 20 / w))
    else:
        new_h = 20
        new_w = max(1, round(w * 20 / h))

    digit = digit.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Paste vào canvas 28x28
    canvas = Image.new("L", (28, 28), 0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas.paste(digit, (left, top))

    # Center theo trọng tâm pixel
    arr = np.array(canvas, dtype=np.float32)
    total_mass = arr.sum()

    if total_mass > 0:
        yy, xx = np.indices(arr.shape)
        cx = (xx * arr).sum() / total_mass
        cy = (yy * arr).sum() / total_mass

        target_center = 13.5
        shift_x = int(round(target_center - cx))
        shift_y = int(round(target_center - cy))

        arr = shift_array(arr, shift_x, shift_y)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    processed = Image.fromarray(arr.astype(np.uint8), mode="L")
    preview_large = np.array(processed.resize((280, 280), Image.Resampling.NEAREST))

    return processed, preview_large


# =========================
# 6) HÀM CHÍNH
# =========================
def predict_and_attack(input_img, epsilon):
    if input_img is None:
        return (
            "<h3 style='color: red;'>Vui lòng vẽ ảnh</h3>", gr.update(value=None),
            "<h3>Nhiễu FGSM</h3>", gr.update(value=None),
            "<h3>Ảnh sau khi bị tấn công</h3>", gr.update(value=None)
        )

    # Preprocess kiểu MNIST
    img_pil, seen_by_ai_large = preprocess_like_mnist(input_img)

    # Tensor normalized
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)

    # Predict clean
    output = model(img_tensor)
    probs = torch.exp(output)
    conf_before, pred_before_tensor = probs.max(1)
    pred_before = pred_before_tensor.item()
    conf_before = conf_before.item()

    html_before = f"<h3 style='color: #3b82f6;'>Dự đoán ban đầu: {pred_before} (Tự tin: {conf_before:.1%})</h3>"

    # Nếu epsilon = 0 thì chỉ predict
    if epsilon <= 0:
        return (
            html_before, seen_by_ai_large,
            "<h3>Nhiễu FGSM (ε = 0)</h3>", None,
            f"<h3 style='color: #16a34a;'> Giữ nguyên dự đoán: {pred_before} (Không có nhiễu)</h3>", seen_by_ai_large
        )

    # Tính gradient cho FGSM
    loss = F.nll_loss(output, pred_before_tensor)
    model.zero_grad(set_to_none=True)
    loss.backward()
    data_grad = img_tensor.grad.detach()

    # Đưa ảnh từ normalized về [0,1]
    mean = torch.tensor(MNIST_MEAN, device=device).view(1, 1, 1, 1)
    std = torch.tensor(MNIST_STD, device=device).view(1, 1, 1, 1)

    img_denorm = img_tensor.detach() * std + mean

    # FGSM attack
    perturbed_img_denorm = fgsm_attack(img_denorm, epsilon, data_grad)

    # Normalize lại trước khi đưa vào model
    perturbed_img_norm = (perturbed_img_denorm - mean) / std

    with torch.no_grad():
        output_after = model(perturbed_img_norm)
        probs_after = torch.exp(output_after)
        conf_after, pred_after_tensor = probs_after.max(1)

    pred_after = pred_after_tensor.item()
    conf_after = conf_after.item()

    # Ảnh nhiễu trực quan
    noise = (perturbed_img_denorm - img_denorm).squeeze().detach().cpu().numpy()

    noise_visual = np.zeros((28, 28, 3), dtype=np.uint8)
    denom = max(epsilon, 1e-8)
    pos = np.clip(noise / denom, 0, 1)
    neg = np.clip(-noise / denom, 0, 1)

    # Đỏ = tăng pixel, Xanh = giảm pixel
    noise_visual[..., 0] = (pos * 255).astype(np.uint8)
    noise_visual[..., 2] = (neg * 255).astype(np.uint8)

    noise_visual_large = np.array(
        Image.fromarray(noise_visual).resize((280, 280), Image.Resampling.NEAREST)
    )

    # Ảnh sau tấn công để hiển thị
    perturbed_img_uint8 = (
        perturbed_img_denorm.squeeze().detach().cpu().numpy().clip(0, 1) * 255
    ).astype(np.uint8)

    perturbed_img_large = np.array(
        Image.fromarray(perturbed_img_uint8).resize((280, 280), Image.Resampling.NEAREST)
    )

    html_noise = f"<h3>Nhiễu FGSM (ε = {epsilon:.2f})</h3>"

    if pred_after == pred_before:
        html_after = f"<h3 style='color: #16a34a;'> Vẫn dự đoán đúng: {pred_after} (Tự tin: {conf_after:.1%})</h3>"
    else:
        html_after = f"<h3 style='color: #ef4444;'> Bị lừa thành: {pred_after} (Tự tin: {conf_after:.1%})</h3>"

    return (
        html_before, seen_by_ai_large,
        html_noise, noise_visual_large,
        html_after, perturbed_img_large
    )


# =========================
# 7) GIAO DIỆN GRADIO
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Demo Tấn công FGSM vào Mô hình Nhận diện Chữ số")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Sketchpad(
                type="numpy",
                label="Vẽ chữ số ở đây",
                height=320
            )
            epsilon_slider = gr.Slider(
                minimum=0.0,
                maximum=0.5,
                step=0.05,
                value=0.15,
                label="Mức nhiễu Epsilon (ε)"
            )
            submit_btn = gr.Button("Dự đoán và Tấn công", variant="primary")

        with gr.Column(scale=1):
            status_before = gr.HTML("<h3>Ảnh mô hình thực sự 'nhìn' thấy</h3>")
            seen_img = gr.Image(
                type="numpy",
                label="",
                show_label=False,
                height=320
            )

    with gr.Row():
        with gr.Column():
            status_noise = gr.HTML("<h3>Nhiễu FGSM</h3>")
            noise_img = gr.Image(
                type="numpy",
                label="",
                show_label=False,
                height=320
            )
        with gr.Column():
            status_after = gr.HTML("<h3>Ảnh sau khi bị tấn công</h3>")
            perturbed_img = gr.Image(
                type="numpy",
                label="",
                show_label=False,
                height=320
            )

    submit_btn.click(
        fn=predict_and_attack,
        inputs=[input_img, epsilon_slider],
        outputs=[status_before, seen_img, status_noise, noise_img, status_after, perturbed_img]
    )


# =========================
# 8) CHẠY APP
# =========================
if __name__ == "__main__":
    demo.launch()