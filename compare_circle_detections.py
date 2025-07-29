import argparse
import numpy as np
import yaml
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import torch
from circle_model import load_yolov8_circle


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ground truth and predicted circle detections")
    parser.add_argument('--weights', type=str, default='circle_trained.pt', help='Trained weights')
    parser.add_argument('--data', type=str, default='datasets/PipeCircle/data_PipeCircle.yaml', help='Dataset YAML for class names')
    parser.add_argument('--source', type=str, default='datasets/PipeCircle/images/test', help='Image directory')
    parser.add_argument('--labels', type=str, default='datasets/PipeCircle/labels/test', help='Ground truth labels directory')
    parser.add_argument('--output', type=str, default='comparison_results', help='Output directory for comparison images')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default='cuda', help='Computation device')
    return parser.parse_args()


def load_ground_truth(label_path, img_width, img_height):
    """Load ground truth annotations from YOLO format"""
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    radius = float(parts[3]) * ((img_width**2 + img_height**2) ** 0.5)
                    annotations.append({
                        'class_id': class_id,
                        'x': x_center,
                        'y': y_center,
                        'radius': radius
                    })
    return annotations


def draw_circles_on_image(image, annotations, color, label_prefix=""):
    """Draw circles on image using PIL ImageDraw"""
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for i, ann in enumerate(annotations):
        x, y, r = ann['x'], ann['y'], ann['radius']
        class_id = ann['class_id']
        
        # Draw circle (ensure valid coordinates and perfect circle)
        if r > 0:  # Only draw if radius is positive
            # Ensure the bounding box is square to draw a perfect circle
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = min(image.width, x + r), min(image.height, y + r)
            
            # Make the bounding box square by using the smaller dimension
            width = x2 - x1
            height = y2 - y1
            size = min(width, height)
            
            # Recalculate the bounding box to be square
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            x1 = center_x - size / 2
            y1 = center_y - size / 2
            x2 = center_x + size / 2
            y2 = center_y + size / 2
            
            # Ensure the square fits within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)
            
            if x2 > x1 and y2 > y1:  # Ensure valid bounding box
                bbox = [x1, y1, x2, y2]
                draw.ellipse(bbox, outline=color, width=2)
        
        # Draw label
        label = f"{label_prefix}Class {class_id}"
        if 'score' in ann:
            label += f" ({ann['score']:.2f})"
        
        # Get text size and position it above the circle
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        text_x = x - text_width // 2
        text_y = y - r - text_height - 5
        
        # Draw text background
        draw.rectangle([text_x-2, text_y-2, text_x + text_width+2, text_y + text_height+2], 
                      fill='black')
        draw.text((text_x, text_y), label, fill=color, font=font)
    
    return image


def process_single_image(img_path, label_path, model, device, names, args, nc):
    """Process a single image and create comparison"""
    try:
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        
        # Load ground truth
        gt_annotations = load_ground_truth(label_path, img_width, img_height)
        
        # Get predictions
        img_resized = img.resize((args.img_size, args.img_size))
        img_tensor = torch.tensor(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Get raw predictions
        with torch.no_grad():
            preds = model.model(img_tensor)
            if isinstance(preds, tuple):
                preds = preds[0]
            elif isinstance(preds, list):
                preds = preds[0]
        
        # Process predictions
        batch_size, channels, height, width = preds.shape
        preds_reshaped = preds.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        
        # Split into circle predictions and classification scores
        reg_channels = channels - nc
        circles, scores = preds_reshaped.split([reg_channels, nc], dim=-1)
        
        # Get best predictions
        best_scores, best_classes = torch.max(scores[0], dim=1)
        best_circles = circles[0, :, :3]  # Take first 3 channels as x, y, r
        
        # Filter by confidence threshold
        confidence_threshold = 0.5
        mask = best_scores > confidence_threshold
        
        pred_annotations = []
        if mask.any():
            filtered_circles = best_circles[mask]
            filtered_scores = best_scores[mask]
            filtered_classes = best_classes[mask]
            
            # Convert predictions to image coordinates
            for i in range(min(len(filtered_circles), 10)):  # Limit to top 10 detections
                x, y, r = filtered_circles[i].cpu().numpy()
                cls = filtered_classes[i].cpu().numpy()
                score = filtered_scores[i].cpu().numpy()
                
                # Scale coordinates back to original image size
                # The model outputs are in normalized coordinates (0-1) relative to the input size
                x = (x + 0.5) * img_width  # Center the coordinates
                y = (y + 0.5) * img_height
                r = abs(r) * min(img_width, img_height) / 2  # Use absolute radius and scale appropriately
                
                # Debug: print some values to understand the scale
                if i == 0:  # Only for first detection
                    print(f"Debug - Raw values: x={filtered_circles[i][0].cpu().numpy():.3f}, y={filtered_circles[i][1].cpu().numpy():.3f}, r={filtered_circles[i][2].cpu().numpy():.3f}")
                    print(f"Debug - Scaled values: x={x:.1f}, y={y:.1f}, r={r:.1f}")
                
                pred_annotations.append({
                    'class_id': int(cls),
                    'x': float(x),
                    'y': float(y),
                    'radius': float(r),
                    'score': float(score)
                })
        
        # Create side-by-side comparison
        # Create a new image with double width
        comparison_img = Image.new('RGB', (img_width * 2, img_height))
        
        # Left side: Ground truth
        gt_img = img.copy()
        gt_img = draw_circles_on_image(gt_img, gt_annotations, (0, 255, 0), "GT ")
        comparison_img.paste(gt_img, (0, 0))
        
        # Right side: Predictions
        pred_img = img.copy()
        pred_img = draw_circles_on_image(pred_img, pred_annotations, (255, 0, 0), "Pred ")
        comparison_img.paste(pred_img, (img_width, 0))
        
        # Add title
        draw = ImageDraw.Draw(comparison_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw title
        title = f"Ground Truth (Green) vs Predictions (Red) - {os.path.basename(img_path)}"
        bbox_text = draw.textbbox((0, 0), title, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        title_x = (img_width * 2 - text_width) // 2
        draw.text((title_x, 10), title, fill=(255, 255, 255), font=font)
        
        # Save comparison image
        os.makedirs(args.output, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output, f"{base_name}_comparison.png")
        comparison_img.save(out_path)
        print(f"Saved comparison to {out_path}")
        
        # Print summary
        print(f"Image: {os.path.basename(img_path)}")
        print(f"  Ground truth circles: {len(gt_annotations)}")
        print(f"  Predicted circles: {len(pred_annotations)}")
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")


def main():
    args = parse_args()
    
    # Load model
    cfg = yaml.safe_load(open(args.data))
    nc = cfg['nc']
    names = cfg.get('names', list(range(nc)))
    model = load_yolov8_circle(args.weights, nc)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model.model.to(device)
    
    # Check if source is a directory or file
    if os.path.isdir(args.source):
        # Get all image files in directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(args.source, ext)))
            image_files.extend(glob.glob(os.path.join(args.source, ext.upper())))
        
        if not image_files:
            print(f"No image files found in directory: {args.source}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(args.labels, f"{base_name}.txt")
            process_single_image(img_path, label_path, model, device, names, args, nc)
    else:
        # Process single image
        base_name = os.path.splitext(os.path.basename(args.source))[0]
        label_path = os.path.join(args.labels, f"{base_name}.txt")
        process_single_image(args.source, label_path, model, device, names, args, nc)
    
    print(f"All comparisons saved to {args.output}/")


if __name__ == '__main__':
    main() 