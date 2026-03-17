# vt_siglip/train_utils_linked.py
import torch
from .siglip_model_token_4 import VisionTextSigLIP
from .data_linked import ImageBatchMulti, TextBatchSingle, build_targets_imgmulti_textsingle


def train_step_linked(
    model: VisionTextSigLIP,
    token_model,
    batch_img: ImageBatchMulti,
    batch_txt: TextBatchSingle,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Training step with Token (vision) and VT (text + projection) fully separated.
    - Token: extract image_feats using forward_test (no_grad + frozen)
    - VT: train contrastive / ASL using image_feats + text
    """
    model.train()
    device = next(model.parameters()).device

    # Move image tensor to the Token device
    images = batch_img.images.to(device, non_blocking=True)

    # Extract features using the Token model without training or gradients
    with torch.no_grad():
        image_feats = token_model.forward_test(images)
    # Build multi-label targets with shape (B, M)
    targets = build_targets_imgmulti_textsingle(batch_img.label_sets, batch_txt.labels).to(
        device
    )

    # Text tensors are moved to the actual device inside LLMTextEncoder,
    # so CPU tensors can be passed directly here.
    input_ids = batch_txt.input_ids
    attention_mask = batch_txt.attention_mask

    optimizer.zero_grad(set_to_none=True)

    use_amp = (scaler is not None) and (device.type == "cuda")
    if use_amp:
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            out = model(image_feats, input_ids, attention_mask, targets=targets)
            loss = out["loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        out = model(image_feats, input_ids, attention_mask, targets=targets)
        loss = out["loss"]
        loss.backward()
        optimizer.step()

    # Support both "temperature" and "temp" keys
    temp_tensor = out.get("temperature", out.get("temp", None))
    if temp_tensor is None:
        temp_tensor = model.log_temp.exp()

    return {
        "loss": float(loss.detach().cpu().item()),
        "temp": float(temp_tensor.detach().cpu().item()),
    }
