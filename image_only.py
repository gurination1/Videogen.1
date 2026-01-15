#!/usr/bin/env python3
"""
videogen.image_only - Standalone Image Generation Module
Optional module for generating images without full video pipeline.

Version: 1.0 - PRODUCTION READY

This module provides image-only generation capability that:
- Works alongside existing video generation (no conflicts)
- Uses same quality settings and caching
- Generates standalone images with prompts
- Can be run independently or imported

Usage:
    # As standalone script
    python image_only.py --prompt "sunset over mountains" --quality ultra
    
    # From Python code
    from videogen.image_only import ImageOnlyGenerator
    gen = ImageOnlyGenerator(config)
    image_path = gen.generate_from_prompt("sunset over mountains")
    
    # From main application
    from videogen.main import main as video_main
    from videogen.image_only import main as image_main
    
    if args.mode == 'image':
        image_main()
    else:
        video_main()

Features:
- Full quality presets (draft/standard/high/ultra)
- Prompt enhancement with quality boosters
- Caching support (reuses existing cache)
- Batch generation support
- Same security validation as video pipeline
- Progress tracking and metrics
- No audio/video dependencies required
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime, timezone

# Import from existing modules (no modifications needed)
from .config import (
    Config,
    VideoQuality,
    AspectRatio,
    VERSION,
    BUILD_DATE,
    ExitCode,
    QUALITY_PRESETS,
    ASPECT_RATIO_SIZES
)
from .utils import (
    safe_print,
    StructuredLogger,
    MetricsCollector,
    ProgressTracker,
    normalize_unicode
)
from .security import SecurityValidator, DNSCache
from .services import ComfyUIClient
from .orchestrator import (
    QUALITY_ENHANCERS,
    LIGHTING_STYLES,
    CAMERA_ANGLES,
    COMPOSITION_TERMS,
    COLOR_PALETTES,
    TIME_OF_DAY,
    ATMOSPHERE_EFFECTS,
    TEXTURE_DETAILS,
    MATERIAL_SPECS,
    FILM_STOCK,
    LENS_SPECS,
    ARTIST_REFERENCES
)


class ImageOnlyGenerator:
    """
    Standalone image generator using existing infrastructure.
    
    This class provides image-only generation without requiring
    the full video pipeline (no audio, no video rendering).
    
    Features:
    - Reuses existing ComfyUI client
    - Uses same caching system
    - Applies quality enhancements
    - No conflicts with video generation
    """
    
    def __init__(
        self,
        config: Config,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
        cache_manager=None,
        dns_cache: Optional[DNSCache] = None
    ):
        """
        Initialize image-only generator.
        
        Args:
            config: Configuration object
            logger: Optional logger instance
            metrics: Optional metrics collector
            cache_manager: Optional cache manager (reuses video cache)
            dns_cache: Optional DNS cache
        """
        self.config = config
        self.logger = logger or StructuredLogger()
        self.metrics = metrics or MetricsCollector()
        self.cache = cache_manager
        self.dns_cache = dns_cache or DNSCache()
        
        # Initialize ComfyUI client (reuses existing client logic)
        self.comfyui = ComfyUIClient(
            self.config.comfyui_url,
            self.config.comfyui_model,
            self.logger,
            self.dns_cache
        )
        
        # Create output directory
        self.output_dir = self.config.video_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def enhance_prompt(
        self,
        base_prompt: str,
        quality: VideoQuality,
        style: Optional[str] = None,
        aspect_ratio: Optional[AspectRatio] = None
    ) -> str:
        """
        Enhance prompt with quality boosters and style elements.
        
        Args:
            base_prompt: Base prompt text
            quality: Quality preset
            style: Optional style (motivational/emotional/tech/nature)
            aspect_ratio: Optional aspect ratio
        
        Returns:
            Enhanced prompt with quality boosters
        """
        # Get quality enhancers
        quality_boost = QUALITY_ENHANCERS.get(
            quality,
            QUALITY_ENHANCERS[VideoQuality.STANDARD]
        )
        
        # Get style-specific enhancements if provided
        components = [base_prompt, quality_boost]
        
        if style:
            style = style.lower()
            components.extend([
                LIGHTING_STYLES.get(style, "professional lighting"),
                CAMERA_ANGLES.get(style, "balanced perspective"),
                COMPOSITION_TERMS.get(style, "rule of thirds"),
                COLOR_PALETTES.get(style, "accurate colors"),
                TIME_OF_DAY.get(style, "natural daylight"),
                ATMOSPHERE_EFFECTS.get(style, "clear atmosphere"),
                TEXTURE_DETAILS.get(style, "detailed textures"),
                MATERIAL_SPECS.get(style, "realistic materials"),
                FILM_STOCK.get(style, "digital cinema"),
                LENS_SPECS.get(style, "professional lens"),
                ARTIST_REFERENCES.get(style, "professional photography")
            ])
        
        # Add aspect ratio hints
        if aspect_ratio:
            if aspect_ratio == AspectRatio.PORTRAIT:
                components.append("vertical composition, portrait orientation")
            elif aspect_ratio == AspectRatio.LANDSCAPE:
                components.append("horizontal composition, landscape orientation")
            elif aspect_ratio == AspectRatio.SQUARE:
                components.append("square format, centered composition")
        
        # Combine and truncate if needed
        prompt = ', '.join(filter(None, components))
        
        # Smart truncation at comma boundary
        if len(prompt) > 2000:
            prompt = prompt[:2000]
            last_comma = prompt.rfind(',')
            if last_comma > 1500:
                prompt = prompt[:last_comma]
        
        return prompt
    
    def generate_from_prompt(
        self,
        prompt: str,
        quality: Optional[VideoQuality] = None,
        aspect_ratio: Optional[AspectRatio] = None,
        style: Optional[str] = None,
        seed: Optional[int] = None,
        output_name: Optional[str] = None
    ) -> Path:
        """
        Generate single image from prompt.
        
        Args:
            prompt: Text prompt for image generation
            quality: Quality preset (uses config default if None)
            aspect_ratio: Aspect ratio (uses config default if None)
            style: Optional style preset
            seed: Optional seed for reproducibility
            output_name: Optional output filename
        
        Returns:
            Path to generated image
        
        Raises:
            Exception: If generation fails
        """
        start_time = time.time()
        
        # Use config defaults if not specified
        quality = quality or self.config.quality
        aspect_ratio = aspect_ratio or self.config.aspect_ratio
        
        # Get dimensions
        width, height = QUALITY_PRESETS[quality]
        if aspect_ratio != AspectRatio.PORTRAIT:
            width, height = ASPECT_RATIO_SIZES[aspect_ratio]
        
        # Enhance prompt
        enhanced_prompt = self.enhance_prompt(prompt, quality, style, aspect_ratio)
        
        self.logger.info(f"Generating image: {prompt[:50]}...")
        safe_print(f"\n🎨 Generating image...")
        safe_print(f"   Prompt: {prompt}")
        safe_print(f"   Quality: {quality.value}")
        safe_print(f"   Size: {width}x{height}")
        
        # Check cache if available
        cache_key = None
        output_path = None
        
        if self.cache:
            # Normalize prompt for cache key
            normalized_prompt = normalize_unicode(prompt.lower())
            normalized_prompt = ''.join(c for c in normalized_prompt if c.isalnum() or c == ' ')
            
            # Generate cache key
            cache_key = self.cache.get_cache_key(
                asset_type='image',
                prompt=normalized_prompt,
                seed=str(seed) if seed else "random",
                quality=quality.value,
                width=str(width),
                height=str(height)
            )
            
            # Generate output filename
            if not output_name:
                timestamp = int(time.time())
                output_name = f"image_{timestamp}.png"
            
            output_path = self.output_dir / output_name
            
            # Try cache
            if self.cache.get_file(cache_key, output_path):
                self.logger.info("Using cached image")
                safe_print("✅ Image retrieved from cache")
                self.metrics.record_success()
                return output_path
        
        # Not in cache, generate new image
        if not output_path:
            timestamp = int(time.time())
            output_name = output_name or f"image_{timestamp}.png"
            output_path = self.output_dir / output_name
        
        # Generate seed if not provided
        if seed is None:
            import random
            seed = random.randint(0, 2**31 - 1)
        
        # Get model
        model = self.comfyui.select_model()
        
        # Build workflow (using VideoConfig as template)
        from .config import VideoConfig
        video_config = VideoConfig(
            width=width,
            height=height,
            quality=quality,
            aspect_ratio=aspect_ratio,
            camera_movement=self.config.camera_movement
        )
        
        # Get quality-specific parameters
        from .config import AudioVideoConstants
        steps = AudioVideoConstants.COMFYUI_STEPS_BY_QUALITY.get(
            quality,
            AudioVideoConstants.COMFYUI_STEPS
        )
        cfg = AudioVideoConstants.COMFYUI_CFG_BY_QUALITY.get(
            quality,
            AudioVideoConstants.COMFYUI_CFG
        )
        
        workflow = self.comfyui.build_workflow(
            enhanced_prompt,
            model,
            seed,
            video_config,
            steps=steps,
            cfg=cfg
        )
        
        # Queue and wait
        safe_print("   Queuing generation...")
        prompt_id = self.comfyui.queue_prompt(workflow)
        
        safe_print("   Waiting for result...")
        progress = ProgressTracker(steps)
        
        def progress_callback(stage: str, prog: float, msg: str):
            """Progress callback for tracking."""
            try:
                progress.update(int(prog * steps), msg)
            except Exception:
                pass
        
        result = self.comfyui.poll_result(prompt_id, progress_callback)
        
        if not result:
            raise Exception("Image generation timeout")
        
        filename, img_type, subfolder = result
        
        safe_print("   Downloading image...")
        success = self.comfyui.download_image(filename, img_type, subfolder, output_path)
        
        if not success:
            raise Exception("Image download failed")
        
        # Validate
        if not output_path.exists() or output_path.stat().st_size < 20480:
            raise Exception("Image validation failed")
        
        # Cache if available
        if self.cache and cache_key:
            self.cache.put_file(cache_key, output_path)
        
        # Record metrics
        duration = time.time() - start_time
        self.metrics.record_generation_time('image', duration)
        self.metrics.record_success()
        
        safe_print(f"\n✅ Image generated successfully!")
        safe_print(f"   Location: {output_path}")
        safe_print(f"   Size: {output_path.stat().st_size / 1024:.1f}KB")
        safe_print(f"   Time: {duration:.1f}s\n")
        
        self.logger.info(f"Image generated: {output_path}")
        
        return output_path
    
    def generate_batch(
        self,
        prompts: List[str],
        quality: Optional[VideoQuality] = None,
        aspect_ratio: Optional[AspectRatio] = None,
        style: Optional[str] = None
    ) -> List[Path]:
        """
        Generate multiple images from list of prompts.
        
        Args:
            prompts: List of text prompts
            quality: Quality preset
            aspect_ratio: Aspect ratio
            style: Optional style preset
        
        Returns:
            List of paths to generated images
        """
        results = []
        
        safe_print(f"\n{'='*70}")
        safe_print(f"BATCH IMAGE GENERATION - {len(prompts)} images")
        safe_print(f"{'='*70}\n")
        
        for i, prompt in enumerate(prompts, 1):
            try:
                safe_print(f"[{i}/{len(prompts)}] Generating: {prompt[:50]}...")
                
                output_name = f"batch_{i:03d}_{int(time.time())}.png"
                image_path = self.generate_from_prompt(
                    prompt,
                    quality=quality,
                    aspect_ratio=aspect_ratio,
                    style=style,
                    output_name=output_name
                )
                
                results.append(image_path)
                
            except Exception as e:
                self.logger.error(f"Image {i} failed: {e}")
                safe_print(f"❌ Failed: {e}")
                self.metrics.record_failure(type(e).__name__)
        
        # Summary
        safe_print(f"\n{'='*70}")
        safe_print(f"BATCH COMPLETE")
        safe_print(f"{'='*70}")
        safe_print(f"  Success: {len(results)}/{len(prompts)}")
        safe_print(f"  Output: {self.output_dir}")
        safe_print(f"{'='*70}\n")
        
        return results


def main():
    """
    Main entry point for standalone image generation.
    
    This can be called independently without the video pipeline.
    """
    parser = argparse.ArgumentParser(
        description=f"Image-Only Generator v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single image
  python image_only.py --prompt "sunset over mountains"
  
  # High quality with style
  python image_only.py --prompt "cyberpunk city" --quality ultra --style tech
  
  # Batch generation from file
  python image_only.py --batch prompts.txt --quality high
  
  # Landscape with seed
  python image_only.py --prompt "ocean waves" --aspect 16:9 --seed 12345

Styles:
  motivational, emotional, tech, nature
  
Quality Presets:
  draft     - 512x768   (fast, testing)
  standard  - 720x1280  (balanced, default)
  high      - 1080x1920 (slow, high quality)
  ultra     - 1440x2560 (very slow, maximum quality)
        """
    )
    
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        help='Text prompt for image generation'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='File containing prompts (one per line)'
    )
    
    parser.add_argument(
        '--quality', '-q',
        type=str,
        choices=['draft', 'standard', 'high', 'ultra'],
        default='standard',
        help='Quality preset (default: standard)'
    )
    
    parser.add_argument(
        '--aspect', '-a',
        type=str,
        choices=['9:16', '1:1', '16:9'],
        default='9:16',
        help='Aspect ratio (default: 9:16)'
    )
    
    parser.add_argument(
        '--style', '-s',
        type=str,
        choices=['motivational', 'emotional', 'tech', 'nature'],
        help='Style preset for enhancements'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output filename (for single image)'
    )
    
    parser.add_argument(
        '--comfyui-url',
        type=str,
        default=os.getenv('COMFYUI_URL', 'http://localhost:8188'),
        help='ComfyUI server URL'
    )
    
    parser.add_argument(
        '--video-dir',
        type=str,
        default=os.getenv('VIDEO_DIR', str(Path.home() / "ai_videos")),
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.prompt and not args.batch:
        parser.error("Either --prompt or --batch must be specified")
    
    if args.prompt and args.batch:
        parser.error("Cannot specify both --prompt and --batch")
    
    # Print header
    safe_print("\n" + "🎨" * 35)
    safe_print(f"IMAGE-ONLY GENERATOR v{VERSION}")
    safe_print(f"Build: {BUILD_DATE}")
    safe_print("🎨" * 35 + "\n")
    
    # Create config
    config = Config(
        qwen_url="http://localhost:11434/v1/chat/completions",  # Not used
        comfyui_url=args.comfyui_url,
        video_dir=Path(args.video_dir),
        quality=VideoQuality(args.quality),
        aspect_ratio=AspectRatio(args.aspect)
    )
    
    # Initialize components
    logger = StructuredLogger()
    metrics = MetricsCollector()
    dns_cache = DNSCache()
    dns_cache.start_cleanup()
    
    # Check ComfyUI connection
    safe_print("🔄 Checking ComfyUI connection...")
    generator = ImageOnlyGenerator(config, logger, metrics, dns_cache=dns_cache)
    
    if not generator.comfyui.check_connection():
        safe_print(f"❌ Cannot connect to ComfyUI at {args.comfyui_url}")
        safe_print("   Make sure ComfyUI is running")
        sys.exit(ExitCode.GENERAL_ERROR)
    
    safe_print("✅ ComfyUI connected\n")
    
    try:
        # Single image mode
        if args.prompt:
            result = generator.generate_from_prompt(
                args.prompt,
                quality=config.quality,
                aspect_ratio=config.aspect_ratio,
                style=args.style,
                seed=args.seed,
                output_name=args.output
            )
            
            sys.exit(ExitCode.SUCCESS)
        
        # Batch mode
        elif args.batch:
            batch_file = Path(args.batch)
            if not batch_file.exists():
                safe_print(f"❌ Batch file not found: {args.batch}")
                sys.exit(ExitCode.GENERAL_ERROR)
            
            # Read prompts
            prompts = []
            with open(batch_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        prompts.append(line)
            
            if not prompts:
                safe_print(f"❌ No prompts found in {args.batch}")
                sys.exit(ExitCode.GENERAL_ERROR)
            
            safe_print(f"📋 Loaded {len(prompts)} prompts from {args.batch}")
            
            results = generator.generate_batch(
                prompts,
                quality=config.quality,
                aspect_ratio=config.aspect_ratio,
                style=args.style
            )
            
            if len(results) == len(prompts):
                sys.exit(ExitCode.SUCCESS)
            elif len(results) > 0:
                sys.exit(ExitCode.PARTIAL_SUCCESS)
            else:
                sys.exit(ExitCode.GENERAL_ERROR)
    
    except KeyboardInterrupt:
        safe_print("\n⊘ Interrupted\n")
        sys.exit(ExitCode.SHUTDOWN_REQUESTED)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        safe_print(f"\n❌ Error: {e}\n")
        sys.exit(ExitCode.GENERAL_ERROR)
    
    finally:
        dns_cache.shutdown()


if __name__ == "__main__":
    main()
