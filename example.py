"""
Example usage of the Face to 3D Pipeline

This file demonstrates different ways to use the simplified pipeline.
"""

from __init__ import FaceTo3DPipeline, face_to_3d
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


def example_1_quick_start():
    """Example 1: Simplest usage - one-liner"""
    print("\n" + "="*60)
    print("Example 1: Quick Start (One-Liner)")
    print("="*60)
    
    # Convert face to 3D in one line
    model_path = face_to_3d(
        image_path="path/to/face.jpg",
        output_dir="./output",
        num_views=8,
        device="auto"  # Automatically uses CUDA if available
    )
    
    print(f"✓ 3D model saved to: {model_path}")


def example_2_standard_8view():
    """Example 2: Standard 8-view configuration"""
    print("\n" + "="*60)
    print("Example 2: Standard 8-View (Fast)")
    print("="*60)
    
    # Initialize pipeline
    pipeline = FaceTo3DPipeline(
        device="cuda",
        lora_strength=0.9,
        output_dir="./output/standard_8view"
    )
    
    # Run with standard 8-view configuration
    # 8 views at eye-level, varying azimuth (0°, 45°, 90°, ...)
    result = pipeline.run(
        face_image_path="path/to/face.jpg",
        camera_config="standard_8view",
        output_format="ply"
    )
    
    print(f"✓ Result: {result}")


def example_3_full_16view():
    """Example 3: Full 16-view configuration"""
    print("\n" + "="*60)
    print("Example 3: Full 16-View (Balanced)")
    print("="*60)
    
    # 16 views with 2 elevation levels
    # Better 3D coverage for reconstruction
    pipeline = FaceTo3DPipeline(
        device="cuda",
        lora_strength=0.9,
        output_dir="./output/full_16view"
    )
    
    result = pipeline.run(
        face_image_path="path/to/face.jpg",
        camera_config="full_16view",
        output_format="ply"
    )
    
    print(f"✓ Result: {result}")


def example_4_complete_32view():
    """Example 4: Complete 32-view configuration"""
    print("\n" + "="*60)
    print("Example 4: Complete 32-View (Comprehensive)")
    print("="*60)
    
    # 32 views with 4 elevation levels
    # Maximum coverage for high-quality reconstruction
    pipeline = FaceTo3DPipeline(
        device="cuda",
        lora_strength=0.9,
        output_dir="./output/complete_32view"
    )
    
    result = pipeline.run(
        face_image_path="path/to/face.jpg",
        camera_config="complete_32view",
        output_format="ply"
    )
    
    print(f"✓ Result: {result}")


def example_5_step_by_step():
    """Example 5: Manual step-by-step pipeline"""
    print("\n" + "="*60)
    print("Example 5: Step-by-Step Manual Control")
    print("="*60)
    
    # Initialize pipeline
    pipeline = FaceTo3DPipeline(
        device="cuda",
        output_dir="./output/manual"
    )
    
    # Load image
    face_image = Image.open("path/to/face.jpg").convert("RGB")
    
    # Step 1: Generate multiview images
    print("\n[Step 1] Generating multiview images...")
    multiview_images = pipeline.generate_multiview(
        face_image,
        camera_config="standard_8view"
    )
    print(f"✓ Generated {len(multiview_images)} views")
    
    # Step 2: 3D reconstruction (TODO: implement)
    print("\n[Step 2] Reconstructing 3D model...")
    point_cloud = pipeline.reconstruct_3d(multiview_images)
    print(f"✓ Generated point cloud with {len(point_cloud)} points")
    
    # Step 3: Mesh generation (TODO: implement)
    print("\n[Step 3] Generating mesh...")
    mesh = pipeline.generate_mesh(point_cloud, method="poisson")
    vertices, faces = mesh
    print(f"✓ Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Step 4: Body extension (TODO: implement)
    print("\n[Step 4] Extending to full body...")
    full_body = pipeline.extend_to_full_body(mesh, body_model="smplx")
    print("✓ Extended to full body")


def example_6_custom_camera_config():
    """Example 6: Using custom camera positions"""
    print("\n" + "="*60)
    print("Example 6: Custom Camera Configuration")
    print("="*60)
    
    pipeline = FaceTo3DPipeline(device="cuda")
    
    # You can access camera configs
    configs = pipeline._get_camera_configs()
    
    print("\nAvailable camera configurations:")
    for name, positions in configs.items():
        print(f"  - {name}: {len(positions)} views")
    
    # Create custom prompts
    print("\nExample camera prompts:")
    examples = [
        (0, 0, 1.0),      # Front view, eye-level, medium
        (45, 30, 0.6),    # Front-right, elevated, close-up
        (180, -30, 1.8),  # Back view, low-angle, wide shot
    ]
    
    for azimuth, elevation, distance in examples:
        prompt = pipeline._create_camera_prompt(azimuth, elevation, distance)
        print(f"  {azimuth}° az, {elevation}° el, {distance}x → {prompt}")


def example_7_batch_processing():
    """Example 7: Batch process multiple images"""
    print("\n" + "="*60)
    print("Example 7: Batch Processing")
    print("="*60)
    
    image_paths = [
        "path/to/face1.jpg",
        "path/to/face2.jpg",
        "path/to/face3.jpg",
    ]
    
    pipeline = FaceTo3DPipeline(
        device="cuda",
        output_dir="./output/batch"
    )
    
    results = []
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
        
        # Create separate output directory for each image
        pipeline.output_dir = pipeline.output_dir.parent / f"batch/image_{i+1}"
        pipeline.output_dir.mkdir(exist_ok=True, parents=True)
        
        result = pipeline.run(
            face_image_path=image_path,
            camera_config="standard_8view"
        )
        results.append(result)
        print(f"✓ Saved to: {result}")
    
    print(f"\n✓ Processed {len(results)} images")


def example_8_adjusting_lora_strength():
    """Example 8: Experimenting with LoRA strength"""
    print("\n" + "="*60)
    print("Example 8: Adjusting LoRA Strength")
    print("="*60)
    
    face_image = Image.open("path/to/face.jpg")
    
    # Try different LoRA strengths
    strengths = [0.7, 0.8, 0.9, 1.0]
    
    for strength in strengths:
        print(f"\n--- Testing LoRA strength: {strength} ---")
        
        pipeline = FaceTo3DPipeline(
            device="cuda",
            lora_strength=strength,
            output_dir=f"./output/lora_strength_{strength}"
        )
        
        multiview_images = pipeline.generate_multiview(
            face_image,
            camera_config="standard_8view"
        )
        
        print(f"✓ Generated {len(multiview_images)} views with strength {strength}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Face to 3D Pipeline - Example Usage")
    print("="*60)
    
    # Uncomment the examples you want to run
    
    # example_1_quick_start()
    # example_2_standard_8view()
    # example_3_full_16view()
    # example_4_complete_32view()
    # example_5_step_by_step()
    example_6_custom_camera_config()
    # example_7_batch_processing()
    # example_8_adjusting_lora_strength()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
