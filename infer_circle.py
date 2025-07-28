import argparse
import numpy as np
import yaml
import os
import glob
from PIL import Image
import torch
from ultralytics.utils.plotting import Annotator


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with circle detection model")
    parser.add_argument('--weights', type=str, default='circle_trained.pt', help='Trained weights')
    parser.add_argument('--data', type=str, default='datasets/PipeCircle/data_PipeCircle.yaml', help='Dataset YAML for class names')
    parser.add_argument('--source', type=str, default='datasets/PipeCircle/images/test', help='Image path or directory')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--save', default = True, action='store_true')
    parser.add_argument('--device', type=str, default='cuda', help='Computation device')
    return parser.parse_args()


def main():
    args = parse_args()
    from circle_model import load_yolov8_circle

    cfg = yaml.safe_load(open(args.data))
    nc = cfg['nc']
    names = cfg.get('names', list(range(nc)))
    model = load_yolov8_circle(args.weights, nc)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model.model.to(device)
    
    # Check if source is a directory or file
    if os.path.isdir(args.source):
        # Process all images in directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(args.source, ext)))
            image_files.extend(glob.glob(os.path.join(args.source, ext.upper())))
        
        if not image_files:
            print(f"No image files found in directory: {args.source}")
            return
            
        print(f"Found {len(image_files)} images to process")
        
        for img_path in image_files:
            process_single_image(img_path, model, device, names, args, nc)
    else:
        # Process single image
        process_single_image(args.source, model, device, names, args, nc)


def process_single_image(img_path, model, device, names, args, nc):
    """Process a single image for circle detection"""
    try:
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((args.img_size, args.img_size))
        # Process image through the model
        img_tensor = torch.tensor(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Get raw predictions
        with torch.no_grad():
            preds = model.model(img_tensor)
            if isinstance(preds, tuple):
                preds = preds[0]
            elif isinstance(preds, list):
                preds = preds[0]
        
        # Process predictions (similar to how the loss function processes them)
        batch_size, channels, height, width = preds.shape
        preds_reshaped = preds.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        
        # Split into circle predictions and classification scores
        reg_channels = channels - nc
        circles, scores = preds_reshaped.split([reg_channels, nc], dim=-1)
        
        # For now, just take the best prediction from each spatial location
        # This is a simplified approach - in practice you'd want proper NMS
        best_scores, best_classes = torch.max(scores[0], dim=1)
        best_circles = circles[0, :, :3]  # Take first 3 channels as x, y, r
        
        # Filter by confidence threshold
        confidence_threshold = 0.5
        mask = best_scores > confidence_threshold
        
        if mask.any():
            # Get the best predictions
            filtered_circles = best_circles[mask]
            filtered_scores = best_scores[mask]
            filtered_classes = best_classes[mask]
            
            annotator = Annotator(img)
            
            # Draw circles
            for i in range(min(len(filtered_circles), 10)):  # Limit to top 10 detections
                x, y, r = filtered_circles[i].cpu().numpy()
                cls = filtered_classes[i].cpu().numpy()
                score = filtered_scores[i].cpu().numpy()
                
                # Scale coordinates back to original image size
                x = x * img.width / args.img_size
                y = y * img.height / args.img_size
                r = r * min(img.width, img.height) / args.img_size
                
                # For now, just print the detection info
                print(f"Detection: class={names[int(cls)]}, score={float(score):.2f}, x={float(x):.1f}, y={float(y):.1f}, r={float(r):.1f}")
        else:
            print(f"No detections above confidence threshold in {img_path}")
            annotator = Annotator(img)
        
        if args.save:
            # Create output filename
            base_name = os.path.splitext(img_path)[0]
            ext = os.path.splitext(img_path)[1]
            out_path = f"{base_name}_pred{ext}"
            # Save the annotated image
            Image.fromarray(annotator.result()).save(out_path)
            print(f"Saved to {out_path}")
        else:
            annotator.result().show()
            
    except Exception as e:
        print(f"Error processing {img_path}: {e}")


if __name__ == '__main__':
    main()
