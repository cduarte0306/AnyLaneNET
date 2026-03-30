import argparse

import cv2
import numpy as np
import torch

from anylane.data import get_val_transforms
from anylane.models import LaneNet


def main():
    parser = argparse.ArgumentParser(description="Run AnyLaneNET inference on a video")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LaneNet(num_classes=1, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = get_val_transforms(360, 640)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            mask = (torch.sigmoid(logits) > args.threshold).squeeze().cpu().numpy().astype(np.uint8) * 255

        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        overlay = frame.copy()
        overlay[mask_resized > 0] = (0, 255, 0)
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        cv2.imshow("AnyLaneNET", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()