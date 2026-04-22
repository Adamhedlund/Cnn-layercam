from torchvision.models import resnet18, ResNet18_Weights
from torchcam.methods import LayerCAM
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    return model, weights

def load_preprocess(image_path, weights):
    pil_image = Image.open(image_path)
    pil_image = pil_image.convert("RGB")

    preprocess = weights.transforms()
    
    input_tensor = preprocess(pil_image)
    input_tensor = input_tensor.unsqueeze(0)

    return pil_image, input_tensor

def predict(model, input_tensor, weights, top_k=5):
    with torch.no_grad():
        outputs = model(input_tensor)
    
    logits = outputs[0]
    probabilities = F.softmax(logits, dim=0)
    top_probs, top_idxs = torch.topk(probabilities, k=top_k)
        
    categories = weights.meta["categories"]

    predictions = []

    for prob, idx in zip(top_probs, top_idxs):
        idx_int = idx.item()
        prob_float = prob.item()   
        pred = {
                "class_index": idx_int,
                "class_name": categories[idx_int],
                "confidence": prob_float,
            }

        predictions.append(pred)

    return logits, predictions

def generate_cam(model, input_tensor, class_idx):
    model.zero_grad()

    with LayerCAM(model) as cam_extractor:
        outputs = model(input_tensor)   
        activation_map = cam_extractor(class_idx, outputs)
    return activation_map



def analyze_image(model, weights, image_path, top_k=5, class_rank=0):
    pil_image, input_tensor = load_preprocess(image_path, weights)
    logits, predictions = predict(model, input_tensor, weights, top_k=top_k)
    class_idx = predictions[class_rank]["class_index"]
    cam = generate_cam(model, input_tensor, class_idx)

    cam = cam[0]

    cam = cam.squeeze(0)

    width, height = pil_image.size

    cam_resized = cam.unsqueeze(0).unsqueeze(0)
    cam_resized = F.interpolate(
        cam_resized,
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )
    cam_resized = cam_resized.squeeze().detach().cpu().numpy()
    
    return {
    "pil_image": pil_image,
    "predictions": predictions,
    "class_idx": class_idx,
    "cam_resized": cam_resized,
    "logits": logits
    }

def analyze_class(model, weights, class_name, classes):
    paths = classes[class_name]

    pos_result = analyze_image(model, weights, paths["positive"])
    neg_result = analyze_image(model, weights, paths["negative"])

    return pos_result, neg_result

def plot_class_results(pos_result, neg_result, class_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (res, title) in enumerate([
        (pos_result, "Positive"),
        (neg_result, "Negative")
    ]):
        pil_image = res["pil_image"]
        cam = res["cam_resized"]
        pred = res["predictions"][0]

        axes[i].imshow(pil_image)
        axes[i].imshow(cam, cmap="jet", alpha=0.4)

        axes[i].set_title(
            f'{title}\nPred: {pred["class_name"]} ({pred["confidence"]*100:.1f}%)'
        )
        axes[i].axis("off")

    plt.suptitle(f"Class: {class_name}")
    plt.tight_layout()
    plt.show()