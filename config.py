#!/usr/bin/env python3
"""
videogen.config - Configuration and Constants Module
Contains all enums, constants, and configuration dataclasses.
This module has NO internal dependencies - it's the base layer.

Version: 42.1 (ULTRA QUALITY ENHANCED - PRODUCTION CERTIFIED)
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from urllib.parse import urlparse

VERSION = "42.1"
BUILD_DATE = "2026-01-03"

def parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse semantic version string into major, minor, patch tuple"""
    try:
        parts = version_str.split('.')
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return (major, minor, patch)
    except (ValueError, IndexError):
        return (0, 0, 0)

VERSION_TUPLE = parse_version(VERSION)


class VideoQuality(str, Enum):
    """Video quality presets with proper encoding parameters"""
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    ULTRA = "ultra"


class AspectRatio(str, Enum):
    """Aspect ratio presets for various platforms"""
    PORTRAIT = "9:16"
    SQUARE = "1:1"
    LANDSCAPE = "16:9"


class CameraMovement(str, Enum):
    """Camera movement styles for cinematic effect"""
    STATIC = "static"
    SLOW_ZOOM = "slow_zoom"
    ACCELERATE_ZOOM = "accelerate_zoom"
    DRIFT = "drift"
    PAN = "pan"
    CINEMATIC = "cinematic"


class ExitCode(IntEnum):
    """Exit codes for shell integration and error handling"""
    SUCCESS = 0
    GENERAL_ERROR = 1
    NO_VIDEOS_GENERATED = 2
    PARTIAL_SUCCESS = 3
    SHUTDOWN_REQUESTED = 130


class ServiceConstants:
    """Service-specific constants for external API interactions"""
    COMFYUI_POLL_ATTEMPTS: int = 180
    COMFYUI_POLL_INTERVAL: float = 2.0
    API_TIMEOUT: int = 45
    DOWNLOAD_TIMEOUT: int = 120
    API_RATE_LIMIT_SECONDS: float = 0.8
    RETRY_JITTER_MAX: float = 1.0


class CacheConstants:
    """Cache and storage constants for optimized performance"""
    MAX_CACHE_SIZE_MB: int = 3000
    CACHE_VERSION: str = f"v{VERSION}"
    FILE_STABILITY_CHECKS: int = 3
    FILE_STABILITY_INTERVAL: float = 0.5
    DNS_CACHE_TTL: int = 300
    MAX_CACHE_ENTRIES: int = 10000
    CACHE_MAX_AGE_DAYS: int = 30
    MAX_CACHE_KEY_SIZE: int = 10240
    MIN_FILE_SIZES: Dict[str, int] = {
        'json': 10,
        'txt': 1,
        'sig': 32,
        'audio': 10240,
        'image': 20480,
        'video': 102400,
        'model': 1048576
    }
    PREWARM_VOICES: List[str] = ['bryce', 'ryan', 'amy']
    PREWARM_MUSIC_GENRES: List[str] = ['motivational', 'emotional']


class ResourceConstants:
    """Resource management constants for system stability"""
    MIN_DISK_SPACE_MB: int = 3072
    MAX_MEMORY_PERCENT: float = 85.0
    CRITICAL_MEMORY_PERCENT: float = 95.0
    MAX_ASSET_RETRIES: int = 3
    MAX_ACTIVE_PROMPTS: int = 15
    MAX_URL_LENGTH: int = 2048
    MAX_SCRIPT_LENGTH_CHARS: int = 800
    CHUNK_SIZE: int = 16384
    MIN_VIDEO_DURATION: float = 3.0
    MAX_VIDEO_DURATION: float = 300.0
    MAX_DOWNLOAD_SIZE_MB: int = 500
    MAX_TTS_TEXT_LENGTH: int = 800
    MAX_CONCURRENT_TTS: int = 2
    MAX_CONCURRENT_LLM: int = 2
    MAX_REDIRECT_DEPTH: int = 10
    MIN_VIDEO_WIDTH: int = 64
    MIN_VIDEO_HEIGHT: int = 64
    MAX_FILTER_LENGTH: int = 32768
    MAX_DRAWTEXT_FILTERS: int = 10
    METRICS_HISTORY_LIMIT: int = 1000
    MAX_TRACKED_ASSETS: int = 50


class AudioVideoConstants:
    """Audio and video processing constants - ULTRA QUALITY ENHANCED"""
    AUDIO_SAMPLE_RATE: int = 48000
    TARGET_VIDEO_DURATION: float = 12.0
    SCRIPT_MAX_WORDS: int = 45
    SCRIPT_TARGET_WORDS: int = 35
    TEXT_WRAP_CHARS_PER_LINE: int = 40
    
    FFMPEG_CRF_DRAFT: int = 28
    FFMPEG_CRF_STANDARD: int = 23
    FFMPEG_CRF_HIGH: int = 18
    FFMPEG_CRF_ULTRA: int = 15
    
    FFMPEG_PRESET_DRAFT: str = "veryfast"
    FFMPEG_PRESET_STANDARD: str = "medium"
    FFMPEG_PRESET_HIGH: str = "slow"
    FFMPEG_PRESET_ULTRA: str = "slower"
    
    AUDIO_BITRATE_DRAFT: str = "96k"
    AUDIO_BITRATE_STANDARD: str = "128k"
    AUDIO_BITRATE_HIGH: str = "192k"
    AUDIO_BITRATE_ULTRA: str = "320k"
    
    BGM_VOLUME: float = 0.12
    NARRATION_VOLUME: float = 1.0
    VIDEO_FPS: int = 60
    KEYFRAME_INTERVAL: int = 120
    
    TEXT_BOX_OPACITY: float = 0.85
    TEXT_BOX_BORDER: int = 20
    TEXT_BORDER_WIDTH: int = 4
    TEXT_BOX_COLOR: str = "black@0.85"
    
    MIN_FONT_SIZE: int = 24
    MAX_FONT_SIZE: int = 180
    MAX_TEXT_LINES: int = 15
    MIN_FADE_DURATION: float = 0.5
    MAX_FADE_PERCENT: float = 0.2
    MIN_CONTENT_PERCENT: float = 0.6
    
    DRIFT_AMPLITUDE_4K: int = 40
    DRIFT_AMPLITUDE_HD: int = 30
    DRIFT_AMPLITUDE_SD: int = 20
    
    COMFYUI_STEPS: int = 50
    COMFYUI_CFG: float = 7.5
    COMFYUI_SAMPLER: str = "dpmpp_2m_karras"
    COMFYUI_SCHEDULER: str = "karras"
    
    COMFYUI_STEPS_BY_QUALITY: Dict[VideoQuality, int] = {
        VideoQuality.DRAFT: 20,
        VideoQuality.STANDARD: 35,
        VideoQuality.HIGH: 50,
        VideoQuality.ULTRA: 80
    }
    
    COMFYUI_CFG_BY_QUALITY: Dict[VideoQuality, float] = {
        VideoQuality.DRAFT: 6.0,
        VideoQuality.STANDARD: 7.5,
        VideoQuality.HIGH: 8.5,
        VideoQuality.ULTRA: 9.5
    }
    
    NEGATIVE_PROMPT: str = (
        "blurry, low quality, pixelated, distorted, watermark, text overlay, logo, signature, username, "
        "compression artifacts, jpeg artifacts, noise, grainy, amateur, low resolution, lowres, "
        "worst quality, low quality, normal quality, bad quality, poor quality, terrible quality, "
        "aliasing, jagged edges, stair-stepping, pixelation, moiré pattern, screen door effect, scan lines, "
        "noise artifacts, banding, posterization, color banding, gradient banding, "
        "bad encoding, encoding artifacts, transcoding artifacts, macro blocking, video compression artifacts, "
        "deformed, disfigured, ugly, mutated, mutation, deformed body, deformed face, deformed hands, deformed feet, "
        "extra limbs, missing limbs, extra arms, missing arms, extra legs, missing legs, "
        "extra fingers, missing fingers, fused fingers, too many fingers, long fingers, fewer digits, extra digits, "
        "bad anatomy, bad hands, bad fingers, bad feet, bad proportions, gross proportions, "
        "poorly drawn hands, poorly drawn face, poorly drawn feet, malformed limbs, "
        "mutated hands, mutated fingers, mutated body parts, "
        "long neck, elongated neck, extra heads, two heads, multiple heads, cloned face, "
        "cropped, cut off, out of frame, body out of frame, head out of frame, "
        "dehydrated, extra faces, gross face, badly drawn face, disconnected limbs, floating limbs, detached limbs, "
        "missing body parts, bad teeth, bad eyes, bad nose, bad ears, bad mouth, bad lips, "
        "cross-eyed, lazy eye, asymmetric eyes, uneven eyes, different sized eyes, "
        "bad hair, bad skin, bad skin texture, pores too visible, blemishes, acne, "
        "wrinkles, age spots, veins, scars unless intentional, "
        "overexposed, underexposed, oversaturated, desaturated, washed out colors, "
        "dull colors, muted colors, flat colors, bad colors, wrong colors, ugly colors, "
        "bad lighting, flat lighting, harsh lighting, unnatural lighting, poor lighting, "
        "backlit, silhouette unless intentional, overlit, underlit, "
        "bad shadows, harsh shadows, no shadows, wrong shadows, multiple light sources, "
        "bad color temperature, wrong white balance, color cast, "
        "bad color grading, amateur color grading, "
        "bad contrast, low contrast, high contrast, crushed blacks, blown highlights, "
        "clipping, overexposed areas, underexposed areas, lost detail in shadows, "
        "bad dynamic range, low dynamic range, hdr artifacts when not wanted, "
        "cartoon, anime, 3d render, cgi, painting, drawing, sketch, illustration, "
        "unrealistic, fake, plastic, synthetic, doll-like, toy-like, "
        "bad materials, wrong materials, plastic-looking, glossy when should be matte, "
        "matte when should be glossy, wrong texture, bad texture, flat texture, "
        "repetitive texture, tiled texture, texture artifacts, "
        "bad composition, poor composition, unbalanced composition, centered composition, "
        "bad framing, bad cropping, awkward framing, "
        "bad perspective, wrong perspective, distorted perspective, impossible perspective, "
        "bad scale, wrong scale, objects wrong size, size inconsistency, "
        "floating objects, objects in wrong place, spatial inconsistency, "
        "bad focus, soft focus unless intentional, front focus, back focus, "
        "focus on wrong subject, everything in focus, nothing in focus, "
        "bad sharpness, oversharpened, sharpening artifacts, halo, "
        "bad depth, flat image, no depth, wrong depth cues, "
        "bad aerial perspective, atmospheric perspective missing, "
        "bad depth of field, everything in focus, nothing in focus, "
        "bad bokeh, bad background blur, bad foreground blur, "
        "lens distortion, chromatic aberration, vignette unless intentional, motion blur, zoom blur, "
        "fish eye unless intentional, barrel distortion, pincushion distortion, "
        "bad reflection, bad refraction, bad transparency, bad translucency, "
        "bad alpha channel, alpha artifacts, edge artifacts, halo artifacts, "
        "bad compositing, visible seams, compositing errors, "
        "duplicate, cloned, repeated, tiling, pattern, mirrored, symmetrical artifacts, "
        "watermark, text overlay, caption, subtitle, copyright, trademark, artist name, "
        "border, frame, multiple views, collage, grid, split screen, "
        "morbid, mutilated, gore, blood, violence, nsfw, adult content, "
        "wrong era, anachronism, historically inaccurate, "
        "bad weather, wrong weather, weather inconsistency, "
        "bad time of day, wrong time of day, inconsistent lighting time, "
        "bad season, wrong season, seasonal inconsistency, "
        "bad camera angle, unflattering angle, dutch angle unless intentional, "
        "bad noise reduction, noise reduction artifacts, detail loss from noise reduction, "
        "bad upscaling, upscaling artifacts, blurry from upscaling, "
        "bad downscaling, moiré from downscaling, aliasing from downscaling, "
        "bad aspect ratio, stretched, squashed, wrong aspect ratio, "
        "bad resolution, wrong resolution, resolution mismatch, "
        "amateur photography, beginner mistake, novice error, "
        "stock photo, generic, cliché, overused, uninspired, boring, "
        "bad art direction, poor artistic choices, kitsch, tacky, gaudy, "
        "bad taste, offensive, inappropriate, insensitive, "
        "bad concept, poorly executed, failed attempt, "
        "unfinished, incomplete, work in progress, placeholder, "
        "test image, proof of concept, mockup, wireframe, "
        "screenshot, screen capture, phone photo, webcam, "
        "bad quality scan, scanner artifacts, dust, scratches on scan, "
        "bad print, printing artifacts, print quality issues, "
        "bad display, display artifacts, screen artifacts, dead pixels, "
        "bad file handling, file corruption, data loss, missing data, "
        "error, glitch, artifact, bug, defect, flaw, mistake"
    )
    
    QUALITY_ENHANCERS: Dict[VideoQuality, str] = {
        VideoQuality.ULTRA: (
            "masterpiece, best quality, ultra detailed, extremely detailed, hyper-detailed, intricate details, "
            "meticulous details, fine details, micro details, perfect details, absurdres, highres, "
            "8k uhd, 16k resolution, 32k quality, ultra high resolution, maximum resolution, "
            "professional photography, professional grade, award-winning photography, "
            "gallery quality, museum quality, exhibition quality, fine art photography, "
            "perfect composition, rule of thirds, golden ratio composition, cinematic composition, "
            "professional framing, expert framing, balanced composition, dynamic composition, "
            "photorealistic, hyper-realistic, ultra-realistic, photo-realistic rendering, "
            "physically-based rendering, PBR materials, accurate materials, realistic materials, "
            "subsurface scattering, translucency, realistic skin, skin pores visible, "
            "realistic textures, high-resolution textures, 4k textures, 8k textures, "
            "unreal engine 5, octane render, cycles render, ray tracing, path tracing, "
            "global illumination, accurate lighting simulation, physically accurate lighting, "
            "studio lighting, professional lighting, three-point lighting, Rembrandt lighting, "
            "volumetric lighting, atmospheric lighting, cinematic lighting, dramatic lighting, "
            "god rays, light shafts, crepuscular rays, tyndall effect, caustics, "
            "rim lighting, edge lighting, backlighting, hair light, fill light, key light, "
            "natural lighting, golden hour lighting, blue hour lighting, magic hour, "
            "soft lighting, diffused lighting, hard lighting, directional lighting, "
            "ARRI Alexa, RED camera, professional cinema camera, cinema quality, theatrical quality, "
            "shot on 70mm, IMAX quality, large format, medium format, full frame sensor, "
            "anamorphic lens, cinema lens, prime lens, professional glass, "
            "perfect focus, tack sharp, razor sharp, crystal clear, ultra sharp, "
            "shallow depth of field, bokeh, creamy bokeh, perfect bokeh, cinematic bokeh, "
            "f/1.4 aperture, f/1.8, wide aperture, telephoto compression, "
            "HDR, high dynamic range, wide dynamic range, extended dynamic range, "
            "wide color gamut, rec.2020, DCI-P3, 10-bit color, 12-bit color, "
            "professional color grading, cinematic color grading, color correction, "
            "teal and orange, complementary colors, color harmony, perfect color balance, "
            "vivid colors, rich colors, deep colors, saturated colors, vibrant colors, "
            "natural colors, accurate colors, true-to-life colors, color accuracy, "
            "film grain, organic texture, natural texture, realistic texture, "
            "micro details visible, every detail visible, extreme detail, infinite detail, "
            "perfect anatomy, accurate anatomy, correct proportions, perfect symmetry, "
            "perfect eyes, detailed eyes, perfect hands, detailed hands, perfect skin, "
            "atmospheric perspective, aerial perspective, depth, layering, "
            "volumetric fog, atmospheric haze, mist, particles in air, dust particles, "
            "light rays, sun rays, light beams, volumetric effects, "
            "post-processing, color grading applied, professional editing, "
            "no artifacts, flawless, pristine, immaculate, perfect, "
            "trending on artstation, featured on behance, portfolio piece, "
            "professional work, commercial quality, broadcast quality, cinema quality"
        ),
        VideoQuality.HIGH: (
            "high quality, very detailed, professional photography, "
            "sharp focus, clear details, good composition, "
            "proper lighting, good color balance, "
            "depth of field, photographic quality, "
            "high resolution, 4k uhd, professional grade, "
            "HDR, ray tracing, physically-based rendering, "
            "cinematic lighting, studio lighting, professional camera, "
            "natural textures, realistic materials, accurate colors, "
            "detailed textures, intricate details, fine details"
        ),
        VideoQuality.STANDARD: (
            "good quality, detailed, clear, focused, "
            "professional, well composed, balanced lighting, "
            "high resolution, sharp details, proper exposure, "
            "accurate colors, natural look, photographic"
        ),
        VideoQuality.DRAFT: ""
    }
    
    SSML_BREATH_PAUSE: float = 0.3
    SSML_SENTENCE_PAUSE: float = 0.5
    ABBREVIATION_EXPANSION: Dict[str, str] = {
        'AI': 'Artificial Intelligence',
        'ML': 'Machine Learning',
        'USA': 'United States',
        'UK': 'United Kingdom',
        'CEO': 'Chief Executive Officer',
        'CFO': 'Chief Financial Officer',
        'CTO': 'Chief Technology Officer',
        'PhD': 'Doctor of Philosophy',
        'Mr.': 'Mister',
        'Mrs.': 'Missus',
        'Dr.': 'Doctor',
    }


class TimeoutConstants:
    """Timeout constants with overflow protection"""
    SHUTDOWN_TIMEOUT: float = 15.0
    EXECUTOR_SHUTDOWN_TIMEOUT: float = 30.0
    CRITICAL_SHUTDOWN_TIMEOUT: float = 5.0
    FFMPEG_BASE_TIMEOUT: int = 180
    FFMPEG_PER_SECOND: int = 8
    MAX_FFMPEG_TIMEOUT: int = 3600
    TTS_TIMEOUT: int = 60
    PIPER_VERSION_TIMEOUT: int = 2
    SERVICE_VALIDATION_TIMEOUT: int = 10


QUALITY_PRESETS = {
    VideoQuality.DRAFT: (512, 768),
    VideoQuality.STANDARD: (720, 1280),
    VideoQuality.HIGH: (1080, 1920),
    VideoQuality.ULTRA: (1440, 2560)
}

ASPECT_RATIO_SIZES = {
    AspectRatio.PORTRAIT: (1080, 1920),
    AspectRatio.SQUARE: (1080, 1080),
    AspectRatio.LANDSCAPE: (1920, 1080),
}

REQUEST_HEADERS = {
    'User-Agent': f'VideoGenerator/{VERSION} (Python; Professional)',
    'Accept': 'application/json,*/*',
}


class ErrorMessages:
    """Centralized error messages with templates"""
    
    @staticmethod
    def missing_dependency(dep: str, install_cmd: Optional[str] = None) -> str:
        """Generate missing dependency error message"""
        msg = f"Missing dependency: {dep}"
        if install_cmd:
            msg += f"\nInstall with: {install_cmd}"
        return msg
    
    @staticmethod
    def service_unavailable(service: str, url: str, details: Optional[str] = None) -> str:
        """Generate service unavailable error message"""
        msg = f"Cannot connect to {service} at {url}"
        if details:
            msg += f"\nDetails: {details}"
        msg += f"\n💡 Check if {service} is running"
        return msg
    
    @staticmethod
    def invalid_input(field: str, value: Union[str, int, float], reason: str) -> str:
        """Generate invalid input error message"""
        return f"Invalid {field}: {value}\nReason: {reason}"
    
    @staticmethod
    def cache_corruption(path: Path, reason: str) -> str:
        """Generate cache corruption error message"""
        return f"Cache file corrupted: {path.name}\nReason: {reason}\nFile will be regenerated"
    
    @staticmethod
    def resource_exhausted(resource: str, current: Union[int, float], limit: Union[int, float]) -> str:
        """Generate resource exhausted error message"""
        return f"{resource} exhausted: {current} >= {limit}\nFree up resources and try again"
    
    @staticmethod
    def validation_failed(item: str, details: str) -> str:
        """Generate validation failed error message"""
        return f"Validation failed for {item}\nDetails: {details}"


@dataclass
class GenreConfig:
    """Genre configuration dataclass with validation"""
    keywords: List[str]
    img_prompt: str
    music_urls: List[str]
    voice: str
    camera_movement: CameraMovement
    tonality: str = "neutral"
    
    def __post_init__(self) -> None:
        """Validate and convert camera_movement to enum"""
        if isinstance(self.camera_movement, str):
            self.camera_movement = CameraMovement(self.camera_movement)


GENRE_CONFIG: Dict[str, GenreConfig] = {
    'motivational': GenreConfig(
        keywords=['discipline', 'power', 'success', 'determination', 'achievement'],
        img_prompt="Epic cinematic golden hour sunrise over majestic mountain peak, powerful silhouette of determined athlete standing triumphantly at summit, arms raised in victory pose, dramatic volumetric god rays piercing through billowing clouds, atmospheric fog rolling through valleys below, stunning color palette with deep oranges, brilliant golds, and rich purples, professional photography shot on ARRI Alexa with anamorphic cinema lens, 8k uhd ultra high resolution, razor sharp focus on subject, perfect rim lighting creating glowing edge around silhouette, three-point lighting setup with golden key light from sunrise, subtle fill light from cloud reflections, dramatic hair light separating subject from background, tyndall effect visible in atmosphere, crepuscular rays creating sense of divine inspiration, shallow depth of field f/2.8 with creamy bokeh on distant mountain ranges, teal and orange cinematic color grading, perfect composition following rule of thirds and golden ratio, inspiring and powerful mood, award-winning photography, gallery quality, masterpiece, hyper-detailed terrain textures, realistic rock formations, natural cloud patterns, atmospheric perspective showing depth and scale, film grain for organic texture, high dynamic range capturing full spectrum from deep shadow details to brilliant highlight information, physically-based rendering, subsurface scattering on clouds, realistic atmospheric haze, professional color correction, broadcast quality, IMAX theatrical presentation, shot on 70mm film stock, perfect exposure, meticulous attention to detail, every element in harmony, inspirational and uplifting, hero's journey aesthetic, triumph over adversity theme, unstoppable force energy",
        music_urls=[
            'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Tours/Enthusiast/Tours_-_01_-_Enthusiast.mp3',
            'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/WFMU/Broke_For_Free/Directionless_EP/Broke_For_Free_-_04_-_Something_Elated.mp3'
        ],
        voice='bryce',
        camera_movement=CameraMovement.ACCELERATE_ZOOM,
        tonality='intense'
    ),
    'emotional': GenreConfig(
        keywords=['healing', 'hope', 'love', 'compassion', 'growth'],
        img_prompt="Deeply cinematic moody atmospheric scene, intimate close-up shot of contemplative figure silhouette beside rain-streaked window during melancholic blue hour, thousands of realistic water droplets cascading down glass surface catching and refracting warm amber light from interior space, soft bokeh city lights in distant background creating dreamy out-of-focus orbs of gold, cyan, and crimson, professional cinema camera RED shot with 85mm f/1.4 portrait prime lens, ultra shallow depth of field creating ethereal separation between subject and environment, 8k uhd resolution with meticulous micro details visible in water droplets, skin texture, and fabric weave, natural window light as key creating Rembrandt lighting pattern on subject's face with characteristic triangle of light on shadow side cheek, melancholic and introspective mood, desaturated color palette emphasizing blues, teals, and muted amber tones, professional color grading with teal shadows and warm highlights, film grain texture for organic cinematic feel, perfect composition with subject positioned using golden ratio and rule of thirds, negative space emphasizing isolation and contemplation, atmospheric haze visible in light rays, volumetric lighting creating soft god rays through window, subsurface scattering on skin showing realistic translucency, physically-based rendering of all materials including realistic glass refraction, accurate water behavior and surface tension, high dynamic range preserving detail in both deep shadows and bright window highlights, award-winning cinematography, gallery quality fine art photography, masterpiece composition, hyper-realistic textures, intimate and personal feeling, emotional depth, universal human experience, vulnerability and strength, quiet moment of reflection, hope emerging from darkness, healing journey aesthetic, photorealistic rendering, broadcast quality, theatrical presentation, shot on anamorphic lens, cinema aspect ratio, professional production value, meticulous attention to lighting and mood",
        music_urls=[
            'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Kai_Engel/Satin/Kai_Engel_-_04_-_Sentinel.mp3',
        ],
        voice='amy',
        camera_movement=CameraMovement.SLOW_ZOOM,
        tonality='calm'
    ),
    'tech': GenreConfig(
        keywords=['future', 'innovation', 'AI', 'technology', 'digital'],
        img_prompt="Breathtaking futuristic cyberpunk cityscape at night, towering skyscrapers with intricate high-tech architecture featuring holographic advertisements and neon signage in electric blues, vibrant magentas, and brilliant cyans, rain-slicked streets reflecting kaleidoscope of neon colors creating mirror-like surface, flying vehicles with glowing trails streaking through misty atmosphere, massive holographic interfaces floating in air displaying complex data visualizations and code matrices, dramatic low-angle perspective emphasizing scale and power of technological advancement, professional cinema camera ARRI Alexa shot with wide-angle 24mm anamorphic lens, 8k uhd ultra high resolution capturing every intricate detail of circuit patterns, LED arrays, and architectural elements, cinematic blade runner aesthetic with heavy atmospheric fog and volumetric lighting creating dramatic shafts of colored light cutting through darkness, cyberpunk color grading with pushed cyan and magenta complementary palette, teal shadows with warm neon highlights, perfect composition using dramatic diagonal lines and vanishing point perspective, ray traced reflections showing accurate light behavior on wet surfaces and metallic materials, physically-based rendering of all materials including chrome, brushed aluminum, holographic displays, and glass with realistic refraction, global illumination simulating complex light bounces between neon sources and reflective surfaces, subsurface scattering on translucent materials, caustics from refracting light through rain and glass, sharp focus on foreground architectural details with atmospheric perspective fading distant buildings into haze, film grain for organic cinematic texture, high dynamic range capturing full spectrum from deep shadow detail to brilliant neon highlights, award-winning cinematography, gallery quality sci-fi art, masterpiece composition, hyper-detailed textures on every surface, meticulous attention to technological details like circuit boards, holographic particle effects, LED matrices, fiber optic cables, and mechanical components, sense of endless vertical scale and density, bustling energy of future megacity, innovation and progress theme, digital transformation aesthetic, unreal engine 5 rendering quality, octane render, theatrical IMAX presentation, professional color correction, broadcast quality, shot on 70mm, perfect exposure balance, vivid saturated colors, crisp sharp details, photorealistic yet stylized, visually stunning, cutting-edge aesthetic",
        music_urls=[
            'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/BoxCat_Games/Nameless_the_Hackers_RPG_Soundtrack/BoxCat_Games_-_10_-_Epic_Song.mp3',
        ],
        voice='ryan',
        camera_movement=CameraMovement.DRIFT,
        tonality='energetic'
    ),
    'nature': GenreConfig(
        keywords=['wildlife', 'ocean', 'forest', 'earth', 'beauty'],
        img_prompt="Stunning National Geographic style nature documentary cinematography, majestic wildlife subject in pristine natural habitat, breathtaking landscape vista during magical golden hour with warm honey-toned sunlight filtering through morning mist, professional wildlife photography shot on Canon cinema camera with 300mm super-telephoto prime lens creating beautiful compression and isolation of subject, 8k uhd ultra high resolution capturing every intricate detail of fur texture, feather patterns, and environmental elements, razor sharp focus on animal's eyes showing catch lights and incredible detail while background melts into creamy bokeh with soft out-of-focus foliage creating natural frame, perfect shallow depth of field at f/2.8 aperture separating subject from environment, natural lighting as key source with sun rays piercing through trees creating dramatic god rays and volumetric lighting effects in atmospheric moisture, rim lighting from backlit sun creating glowing outline around subject's silhouette emphasizing form and separating from background, warm golden color palette with rich earth tones, natural greens, and warm amber highlights, professional color grading preserving natural accurate colors while enhancing vibrancy and depth, film grain texture for organic documentary feel, perfect composition following rule of thirds with subject positioned at power point and environmental context providing story, atmospheric perspective showing depth through layers of misty forest receding into distance, subsurface scattering on translucent leaves and organic materials, physically-based rendering of natural materials including realistic fur dynamics, accurate feather structure, organic plant textures, and natural surface properties, high dynamic range capturing full detail from deep forest shadows to bright highlight areas where sunlight breaks through canopy, award-winning nature photography, gallery quality fine art, masterpiece composition, hyper-realistic textures showing individual hairs, scales, or feathers, meticulous attention to natural details like dew drops on spider webs, moss patterns on tree bark, and subtle variations in foliage, sense of untouched wilderness and pristine ecology, majesty of natural world, conservation message, respect for wildlife, intimate moment in nature, patient observation aesthetic, documentary realism, broadcast quality BBC Earth standard, theatrical IMAX nature presentation, shot on large format cinema camera, perfect exposure maintaining natural look, accurate color reproduction, crystal clear details, photorealistic, awe-inspiring beauty, environmental storytelling",
        music_urls=[
            'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/WFMU/Broke_For_Free/Directionless_EP/Broke_For_Free_-_01_-_Night_Owl.mp3',
        ],
        voice='ryan',
        camera_movement=CameraMovement.SLOW_ZOOM,
        tonality='serene'
    )
}

FALLBACK_SCRIPTS = {
    'motivational': "Every moment is a chance to transform yourself into something extraordinary. Embrace the challenge and become unstoppable.",
    'emotional': "In the quiet moments of reflection, we find the strength to heal and the courage to hope again.",
    'tech': "The future is being written in code today. Innovation knows no bounds, and tomorrow's possibilities are limitless.",
    'nature': "Nature's beauty reminds us of the delicate balance of life on Earth. Every creature, every forest, matters.",
    'custom': "Every story has power. Every voice matters. This is your moment to inspire and create change."
}


@dataclass
class VideoConfig:
    """Professional video configuration with quality optimization"""
    width: int
    height: int
    quality: VideoQuality
    aspect_ratio: AspectRatio
    camera_movement: CameraMovement
    font_size: int = 48
    enable_upscaling: bool = False
    enable_interpolation: bool = False
    enable_subtitles: bool = False
    enable_multi_scene: bool = False
    gpu_acceleration: bool = False
    
    def __post_init__(self) -> None:
        """Validate and adjust configuration parameters"""
        if self.width < ResourceConstants.MIN_VIDEO_WIDTH or self.height < ResourceConstants.MIN_VIDEO_HEIGHT:
            raise ValueError(
                f"Invalid dimensions: {self.width}x{self.height}, "
                f"minimum: {ResourceConstants.MIN_VIDEO_WIDTH}x{ResourceConstants.MIN_VIDEO_HEIGHT}"
            )
        
        if self.width % 2 != 0:
            self.width = (self.width // 2) * 2
        if self.height % 2 != 0:
            self.height = (self.height // 2) * 2
        
        min_dim = min(self.width, self.height)
        calculated = int(48 * (min_dim / 720.0))
        calculated = max(AudioVideoConstants.MIN_FONT_SIZE, (calculated // 2) * 2)
        self.font_size = min(AudioVideoConstants.MAX_FONT_SIZE, calculated)
    
    def get_crf(self) -> int:
        """Get CRF value for current quality level"""
        return {
            VideoQuality.DRAFT: AudioVideoConstants.FFMPEG_CRF_DRAFT,
            VideoQuality.STANDARD: AudioVideoConstants.FFMPEG_CRF_STANDARD,
            VideoQuality.HIGH: AudioVideoConstants.FFMPEG_CRF_HIGH,
            VideoQuality.ULTRA: AudioVideoConstants.FFMPEG_CRF_ULTRA
        }[self.quality]
    
    def get_preset(self) -> str:
        """Get FFmpeg encoding preset for current quality level"""
        return {
            VideoQuality.DRAFT: AudioVideoConstants.FFMPEG_PRESET_DRAFT,
            VideoQuality.STANDARD: AudioVideoConstants.FFMPEG_PRESET_STANDARD,
            VideoQuality.HIGH: AudioVideoConstants.FFMPEG_PRESET_HIGH,
            VideoQuality.ULTRA: AudioVideoConstants.FFMPEG_PRESET_ULTRA
        }[self.quality]
    
    def get_audio_bitrate(self) -> str:
        """Get audio bitrate for current quality level"""
        return {
            VideoQuality.DRAFT: AudioVideoConstants.AUDIO_BITRATE_DRAFT,
            VideoQuality.STANDARD: AudioVideoConstants.AUDIO_BITRATE_STANDARD,
            VideoQuality.HIGH: AudioVideoConstants.AUDIO_BITRATE_HIGH,
            VideoQuality.ULTRA: AudioVideoConstants.AUDIO_BITRATE_ULTRA
        }[self.quality]
    
    def get_pixel_format(self) -> str:
        """Get pixel format for current quality level"""
        if self.quality in [VideoQuality.HIGH, VideoQuality.ULTRA]:
            return 'yuv444p'
        return 'yuv420p'
    
    def get_b_frames(self) -> int:
        """Get number of B-frames for current quality level"""
        return {
            VideoQuality.DRAFT: 0,
            VideoQuality.STANDARD: 3,
            VideoQuality.HIGH: 5,
            VideoQuality.ULTRA: 8
        }[self.quality]
    
    def get_drift_amplitude(self) -> int:
        """Get camera drift amplitude based on resolution"""
        min_dim = min(self.width, self.height)
        if min_dim >= 1440:
            return AudioVideoConstants.DRIFT_AMPLITUDE_4K
        elif min_dim >= 720:
            return AudioVideoConstants.DRIFT_AMPLITUDE_HD
        else:
            return AudioVideoConstants.DRIFT_AMPLITUDE_SD
    
    def __repr__(self) -> str:
        return (f"VideoConfig(size={self.width}x{self.height}, quality={self.quality.value}, "
                f"aspect={self.aspect_ratio.value}, crf={self.get_crf()})")


@dataclass
class Config:
    """Application configuration with comprehensive validation"""
    qwen_url: str
    comfyui_url: str
    video_dir: Path
    quality: VideoQuality = VideoQuality.STANDARD
    aspect_ratio: AspectRatio = AspectRatio.PORTRAIT
    camera_movement: CameraMovement = CameraMovement.CINEMATIC
    
    enable_caching: bool = True
    cleanup_on_success: bool = True
    cleanup_on_failure: bool = True
    comfyui_model: Optional[str] = None
    max_workers: int = 2
    allow_private_urls: bool = False
    
    enable_ssml: bool = True
    enable_profiling: bool = False
    enable_cloud_upload: bool = False
    cloud_provider: Optional[str] = None
    cloud_bucket: Optional[str] = None
    notification_webhook: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate and normalize configuration parameters"""
        if isinstance(self.video_dir, str):
            self.video_dir = Path(self.video_dir)
        
        original_workers = self.max_workers
        if self.max_workers < 1 or self.max_workers > 4:
            self.max_workers = max(1, min(4, self.max_workers))
            if original_workers != self.max_workers:
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"max_workers clamped: {original_workers} → {self.max_workers}"
                )
    
    def get_video_config(self) -> VideoConfig:
        """Get video configuration with quality optimization and validation"""
        width, height = QUALITY_PRESETS[self.quality]
        
        if self.aspect_ratio != AspectRatio.PORTRAIT:
            target_w, target_h = ASPECT_RATIO_SIZES[self.aspect_ratio]
            expected_ratio = target_w / target_h
            actual_ratio = width / height
            if abs(expected_ratio - actual_ratio) > 0.05:
                width, height = target_w, target_h
        
        gpu_available = self._detect_gpu_acceleration()
        
        return VideoConfig(
            width=width,
            height=height,
            quality=self.quality,
            aspect_ratio=self.aspect_ratio,
            camera_movement=self.camera_movement,
            gpu_acceleration=gpu_available
        )
    
    def _detect_gpu_acceleration(self) -> bool:
        """Detect if GPU hardware acceleration is available for encoding"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                if 'h264_nvenc' in result.stdout or 'hevc_nvenc' in result.stdout:
                    return True
                if 'h264_amf' in result.stdout:
                    return True
                if 'h264_qsv' in result.stdout:
                    return True
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        return False
    
    @staticmethod
    def check_dependencies() -> List[str]:
        """Check critical dependencies and their versions"""
        errors: List[str] = []
        
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                errors.append("FFmpeg not functional")
            else:
                version_line = result.stdout.split('\n')[0]
                if 'version' in version_line.lower():
                    version_str = version_line.split()[2]
                    try:
                        major = int(version_str.split('.')[0])
                        if major < 4:
                            errors.append(f"FFmpeg version {version_str} too old (need 4.0+)")
                    except (ValueError, IndexError):
                        pass
        except FileNotFoundError:
            errors.append("FFmpeg not found - install: sudo apt install ffmpeg")
        except subprocess.TimeoutExpired:
            errors.append("FFmpeg check timed out")
        except Exception as e:
            errors.append(f"FFmpeg check failed: {str(e)}")
        
        python_version = sys.version_info
        if python_version < (3, 9):
            errors.append(f"Python {python_version.major}.{python_version.minor} too old (need 3.9+)")
        elif python_version >= (3, 13):
            errors.append(f"Python {python_version.major}.{python_version.minor} untested (recommend 3.9-3.12)")
        
        return errors
    
    @classmethod
    def from_env_with_fallback(cls) -> 'Config':
        """Create configuration from environment variables with interactive fallback"""
        qwen_url = os.getenv('QWEN_URL')
        comfyui_url = os.getenv('COMFYUI_URL')
        video_dir_str = os.getenv('VIDEO_DIR')
        
        if not qwen_url and sys.stdin.isatty():
            print("\n🤖 Qwen/Ollama URL (press Enter for http://localhost:11434/v1/chat/completions):")
            user_input = input().strip()
            qwen_url = user_input if user_input else "http://localhost:11434/v1/chat/completions"
        elif not qwen_url:
            qwen_url = "http://localhost:11434/v1/chat/completions"
        
        if not comfyui_url and sys.stdin.isatty():
            print("\n🎨 ComfyUI URL (press Enter for http://localhost:8188):")
            user_input = input().strip()
            comfyui_url = user_input if user_input else "http://localhost:8188"
        elif not comfyui_url:
            comfyui_url = "http://localhost:8188"
        
        if not video_dir_str:
            video_dir_str = str(Path.home() / "ai_videos")
        
        video_dir = Path(video_dir_str)
        
        quality_str = os.getenv('VIDEO_QUALITY', 'standard')
        try:
            quality = VideoQuality(quality_str.lower())
        except ValueError:
            quality = VideoQuality.STANDARD
        
        aspect_str = os.getenv('ASPECT_RATIO', '9:16')
        try:
            aspect_ratio = AspectRatio(aspect_str)
        except ValueError:
            aspect_ratio = AspectRatio.PORTRAIT
        
        camera_str = os.getenv('CAMERA_MOVEMENT', 'cinematic')
        try:
            camera_movement = CameraMovement(camera_str.lower())
        except ValueError:
            camera_movement = CameraMovement.CINEMATIC
        
        return cls(
            qwen_url=qwen_url,
            comfyui_url=comfyui_url,
            video_dir=video_dir,
            quality=quality,
            aspect_ratio=aspect_ratio,
            camera_movement=camera_movement,
            comfyui_model=os.getenv('COMFYUI_MODEL'),
            max_workers=int(os.getenv('MAX_WORKERS', '2')),
            allow_private_urls=os.getenv('ALLOW_PRIVATE_URLS', 'false').lower() == 'true',
            enable_ssml=os.getenv('ENABLE_SSML', 'true').lower() == 'true',
            enable_profiling=os.getenv('ENABLE_PROFILING', 'false').lower() == 'true',
            enable_cloud_upload=os.getenv('ENABLE_CLOUD_UPLOAD', 'false').lower() == 'true',
            cloud_provider=os.getenv('CLOUD_PROVIDER'),
            cloud_bucket=os.getenv('CLOUD_BUCKET'),
            notification_webhook=os.getenv('NOTIFICATION_WEBHOOK')
        )
    
    def to_json(self) -> str:
        """Serialize configuration to JSON string"""
        return json.dumps({
            'qwen_url': self.qwen_url,
            'comfyui_url': self.comfyui_url,
            'video_dir': str(self.video_dir),
            'quality': self.quality.value,
            'aspect_ratio': self.aspect_ratio.value,
            'camera_movement': self.camera_movement.value,
            'comfyui_model': self.comfyui_model,
            'max_workers': self.max_workers,
        }, indent=2)
    
    def __repr__(self) -> str:
        return f"Config(quality={self.quality.value}, aspect={self.aspect_ratio.value}, dir={self.video_dir})"