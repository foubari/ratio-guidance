"""
Image Separation Script for DDPM Training

This script separates the coupled night2day images into separate folders
for training individual DDPM models on night and day images respectively.

Usage:
    python separate_images.py --input_path ../data/night2day/train --output_path ../data/separated --resolution 64
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import shutil


def separate_and_resize_images(input_path, output_path, resolution=64):
    """
    Separate coupled images into night and day folders with specified resolution.
    Preserves train/test/val folder structure.
    
    Args:
        input_path: Path to the input dataset (night2day format)
        output_path: Path where separated images will be saved
        resolution: Target resolution for both width and height
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Remove existing directories if they exist
    if output_path.exists():
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    
    print(f"Input path: {input_path.absolute()}")
    print(f"Output path: {output_path.absolute()}")
    print(f"Target resolution: {resolution}x{resolution}")
    
    # Find all subdirectories (train, test, val, etc.)
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if len(subdirs) == 0:
        # No subdirectories, treat input_path as single dataset
        subdirs = [input_path]
        subdir_names = [""]
    else:
        subdir_names = [d.name for d in subdirs]
    
    print(f"Found subdirectories: {subdir_names}")
    
    total_successful = 0
    total_failed = 0
    
    # Process each subdirectory
    for subdir, subdir_name in zip(subdirs, subdir_names):
        print(f"\n=== Processing {subdir_name or 'root'} ===")
        
        # Create output directories for this subdirectory
        if subdir_name:
            night_dir = output_path / "night" / subdir_name
            day_dir = output_path / "day" / subdir_name
        else:
            night_dir = output_path / "night"
            day_dir = output_path / "day"
        
        night_dir.mkdir(parents=True, exist_ok=True)
        day_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Night images will be saved to: {night_dir}")
        print(f"Day images will be saved to: {day_dir}")
        
        # Find all image files in this subdirectory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(subdir.rglob(f"*{ext}")))
            image_paths.extend(list(subdir.rglob(f"*{ext.upper()}")))
        
        print(f"Found {len(image_paths)} images in {subdir_name or 'root'}")
        
        successful_separations = 0
        failed_separations = 0
        
        # Process each image
        for i, img_path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"  Processing image {i+1}/{len(image_paths)}: {img_path.name}")
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                width, height = img.size
                
                # Determine how to split based on dimensions  
                if width == 512 and height == 256:
                    # Split horizontally: 512x256 -> two 256x256 images
                    left_img = img.crop((0, 0, 256, 256))           # LEFT half (x: 0-256)
                    right_img = img.crop((256, 0, 512, 256))        # RIGHT half (x: 256-512)
                    
                    # Assume LEFT=night, RIGHT=day (verify with your dataset)
                    night_img = left_img
                    day_img = right_img
                    
                elif width == 256 and height == 512:
                    # Split vertically: 256x512 -> two 256x256 images  
                    top_img = img.crop((0, 0, 256, 256))            # TOP half (y: 0-256)
                    bottom_img = img.crop((0, 256, 256, 512))       # BOTTOM half (y: 256-512)
                    
                    # Assume TOP=night, BOTTOM=day (verify with your dataset)
                    night_img = top_img
                    day_img = bottom_img
                else:
                    print(f"  Skipping {img_path.name}: unexpected size {width}x{height}")
                    failed_separations += 1
                    continue
                
                # Resize to target resolution
                night_img_resized = night_img.resize((resolution, resolution), Image.LANCZOS)
                day_img_resized = day_img.resize((resolution, resolution), Image.LANCZOS)
                
                # Generate output filenames
                base_name = img_path.stem
                night_filename = f"{base_name}_night.jpg"
                day_filename = f"{base_name}_day.jpg"
                
                # Save images
                night_img_resized.save(night_dir / night_filename, 'JPEG', quality=95)
                day_img_resized.save(day_dir / day_filename, 'JPEG', quality=95)
                
                successful_separations += 1
                
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                failed_separations += 1
        
        print(f"  {subdir_name or 'root'}: {successful_separations} successful, {failed_separations} failed")
        total_successful += successful_separations
        total_failed += failed_separations
    
    print(f"\n=== Overall Separation Complete ===")
    print(f"Total successfully processed: {total_successful} images")
    print(f"Total failed to process: {total_failed} images")
    
    # Show final structure
    print(f"\nFinal directory structure:")
    for domain in ["night", "day"]:
        domain_path = output_path / domain
        if domain_path.exists():
            print(f"  {domain}/")
            for subdir in domain_path.iterdir():
                if subdir.is_dir():
                    count = len(list(subdir.glob('*.jpg')))
                    print(f"    {subdir.name}/ ({count} images)")
    
    return total_successful, total_failed


def main():
    parser = argparse.ArgumentParser(description='Separate night2day images for DDPM training')
    parser.add_argument('--input_path', type=str, default='../../data/night2day',
                       help='Path to input night2day dataset (should contain train/test/val folders)')
    parser.add_argument('--output_path', type=str, default='../../data/separated_night_day',
                       help='Path to output separated images')
    parser.add_argument('--resolution', type=int, default=64,
                       help='Target resolution for output images')
    
    args = parser.parse_args()
    
    try:
        separate_and_resize_images(args.input_path, args.output_path, args.resolution)
        print("\n✅ Image separation completed successfully!")
        print(f"Final structure:")
        print(f"  {args.output_path}/night/train/  <- Train night images")
        print(f"  {args.output_path}/night/test/   <- Test night images") 
        print(f"  {args.output_path}/night/val/    <- Val night images")
        print(f"  {args.output_path}/day/train/    <- Train day images")
        print(f"  {args.output_path}/day/test/     <- Test day images")
        print(f"  {args.output_path}/day/val/      <- Val day images")
        print(f"\nYou can now train DDPM models using:")
        print(f"  Night DDPM: '{args.output_path}/night/train'")
        print(f"  Day DDPM: '{args.output_path}/day/train'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
