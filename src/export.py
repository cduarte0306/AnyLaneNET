import torch
from anylane.models import LaneNet
import argparse


def export_local():
    model = LaneNet(num_classes=1, pretrained=False)
    ckpt = torch.load("checkpoints/best.pth", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, 192, 320)
    torch.onnx.export(
        model, dummy, "lanenet.onnx",
        input_names=["image"],
        output_names=["logits"],
        opset_version=11,
        dynamo=False,
    )

def main():
    parser = argparse.ArgumentParser(description="Export LaneNet to ONNX format")
    parser.add_argument("--remote", default=False, help="Path to model checkpoint")
    export_local()

if __name__ == "__main__":
    main()