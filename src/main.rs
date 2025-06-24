mod bitmap;

use clap::Parser;
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use std::path::PathBuf;

#[derive(Clone, Copy)]
struct EdgeInfo {
    magnitude: f32,
    direction: f32, // in radians
}

#[derive(Clone)]
struct CharInfo {
    character: char,
    density: usize, // Number of set pixels in bitmap
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short, long, default_value = "input.jpg")]
    pub input: PathBuf,

    #[arg(short, long, default_value = "output.png")]
    pub output: PathBuf,

    #[arg(long, default_value = "8")]
    pub cell_size: u32,

    #[arg(long)]
    pub no_dither: bool,

    #[arg(long, default_value = "1.0")]
    pub upscale_factor: f32,

    #[arg(long)]
    pub terminal: bool,

    #[arg(long, default_value = "ascii")]
    pub charset_type: String,

    #[arg(long, default_value = "0.1")]
    pub edge_threshold: f32,

    #[arg(long)]
    pub color: bool,

    #[arg(long, default_value = "1")]
    pub color_intensity: f32,

    #[arg(long, default_value = "#2d2d2d")]
    pub fg_color: String,

    #[arg(long, default_value = "#15091b")]
    pub bg_color: String,

    #[arg(long, default_value = "1.0")]
    pub sigma1: f32,

    #[arg(long, default_value = "1.6")]
    pub sigma2: f32,

    #[arg(long, default_value = "1.0")]
    pub gamma: f32,

    #[arg(long, default_value = "1.0")]
    pub exposure: f32,

    #[arg(long, default_value = "1.0")]
    pub attenuation: f32,

    #[arg(long)]
    pub invert_luminance: bool,

    #[arg(long, default_value = "false")]
    pub debug: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let img = image::open(&args.input)?;
    let img = img.to_rgb8();

    let upscaled_img = if args.upscale_factor != 1.0 {
        upscale_image(img, args.upscale_factor)?
    } else {
        img
    };

    let (processed_img, edge_info, color_img, final_exposure, final_attenuation, final_invert) =
        process_image(upscaled_img, &args)?;

    // Create modified args with final exposure values
    let mut final_args = args.clone();
    final_args.exposure = final_exposure;
    final_args.attenuation = final_attenuation;
    final_args.invert_luminance = final_invert;

    if args.terminal {
        render_to_terminal(&processed_img, &edge_info, &color_img, &final_args)?;
    } else {
        render_to_ascii(&processed_img, &edge_info, &color_img, &final_args)?;
    }

    Ok(())
}

fn upscale_image(img: RgbImage, scale_factor: f32) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let (width, height) = img.dimensions();
    let new_width = (width as f32 * scale_factor) as u32;
    let new_height = (height as f32 * scale_factor) as u32;

    // Convert to DynamicImage for resizing with high-quality filter
    let dynamic_img = DynamicImage::ImageRgb8(img);
    let resized = dynamic_img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);

    Ok(resized.to_rgb8())
}

fn process_image(
    img: RgbImage,
    args: &Args,
) -> Result<
    (
        RgbImage,
        Vec<Vec<EdgeInfo>>,
        Option<RgbImage>,
        f32,
        f32,
        bool,
    ),
    Box<dyn std::error::Error>,
> {
    let color_img = if args.color { Some(img.clone()) } else { None };

    // Calculate content-aware exposure if using defaults
    let (calculated_exposure, calculated_attenuation, calculated_invert) =
        calculate_content_aware_exposure(&img);

    // Use calculated values only if user hasn't specified custom values
    let final_exposure = if args.exposure == 1.0 {
        calculated_exposure
    } else {
        args.exposure
    };
    let final_attenuation = if args.attenuation == 1.0 {
        calculated_attenuation
    } else {
        args.attenuation
    };
    let final_invert = if !args.invert_luminance {
        calculated_invert
    } else {
        args.invert_luminance
    };

    // Print exposure analysis for debugging
    if args.debug {
        println!("Content-aware exposure analysis:");
        println!("  Calculated exposure: {:.2}", calculated_exposure);
        println!("  Calculated attenuation: {:.2}", calculated_attenuation);
        println!("  Calculated invert: {}", calculated_invert);
        println!("  Using exposure: {:.2}", final_exposure);
        println!("  Using attenuation: {:.2}", final_attenuation);
        println!("  Using invert: {}", final_invert);

        analyze_image_characteristics(&img);
    }

    let processed_img = if args.no_dither {
        let grayscale = apply_grayscale_and_tone_mapping(img);
        let enhanced = apply_enhanced_contrast(grayscale);
        let gamma_corrected = apply_gamma_correction(enhanced, args.gamma);
        apply_unsharp_mask(gamma_corrected)
    } else {
        let grayscale = apply_grayscale_and_tone_mapping(img);
        let dithered = apply_floyd_steinberg_dithering(grayscale);
        apply_difference_of_gaussians(dithered, args)?
    };

    let edge_info = compute_edge_information(&processed_img);

    Ok((
        processed_img,
        edge_info,
        color_img,
        final_exposure,
        final_attenuation,
        final_invert,
    ))
}

const DEFAULT_BLOCK: &str = " .:coPO?@█";
const DEFAULT_ASCII: &str = " .:-=+*%@#";
const FULL_CHARACTERS: &str =
    " .-:=+iltIcsv1x%7aejorzfnuCJT3*69LYpqy5SbdgFGOVXkPhmw48AQDEHKUZR@B#NW0M";

// Edge direction characters
const EDGE_HORIZONTAL: char = '-';
const EDGE_VERTICAL: char = '|';
const EDGE_DIAGONAL_1: char = '\\';
const EDGE_DIAGONAL_2: char = '/';

fn get_charset_by_type(charset_type: &str) -> &str {
    match charset_type {
        "block" => DEFAULT_BLOCK,
        "ascii" => DEFAULT_ASCII,
        "full" => FULL_CHARACTERS,
        _ => DEFAULT_BLOCK,
    }
}

fn calculate_bitmap_density(character: char) -> usize {
    let pattern = bitmap::get_char_pattern(character);
    pattern.iter().map(|&row| row.count_ones() as usize).sum()
}

fn create_sorted_charset(charset: &str) -> Vec<CharInfo> {
    let mut char_infos: Vec<CharInfo> = charset
        .chars()
        .map(|c| CharInfo {
            character: c,
            density: calculate_bitmap_density(c),
        })
        .collect();

    // Sort by density (ascending - lighter to darker)
    char_infos.sort_by(|a, b| a.density.cmp(&b.density));
    char_infos
}

fn parse_hex_color(hex: &str) -> Result<(u8, u8, u8), Box<dyn std::error::Error>> {
    let hex = hex.trim_start_matches('#');
    if hex.len() != 6 {
        return Err("Invalid hex color format".into());
    }

    let r = u8::from_str_radix(&hex[0..2], 16)?;
    let g = u8::from_str_radix(&hex[2..4], 16)?;
    let b = u8::from_str_radix(&hex[4..6], 16)?;

    Ok((r, g, b))
}

fn calculate_content_aware_exposure(img: &RgbImage) -> (f32, f32, bool) {
    let mut dark_pixels = 0;
    let mut bright_pixels = 0;
    let mut mid_tone_sum = 0.0;
    let mut mid_tone_count = 0;
    let mut total_luminance = 0.0;
    let total_pixels = (img.width() * img.height()) as f32;

    for pixel in img.pixels() {
        let luminance =
            (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) / 255.0;

        total_luminance += luminance;

        if luminance < 0.25 {
            dark_pixels += 1;
        } else if luminance > 0.75 {
            bright_pixels += 1;
        } else {
            mid_tone_sum += luminance;
            mid_tone_count += 1;
        }
    }

    let average_luminance = total_luminance / total_pixels;
    let dark_ratio = dark_pixels as f32 / total_pixels;
    let bright_ratio = bright_pixels as f32 / total_pixels;

    // Calculate exposure compensation
    let exposure = if dark_ratio > 0.6 {
        // Very dark image - boost exposure significantly
        let boost = 1.5 + (dark_ratio - 0.6) * 2.0; // 1.5x to 2.3x boost
        boost.min(3.0)
    } else if bright_ratio > 0.6 {
        // Very bright image - reduce exposure
        let reduction = 0.7 - (bright_ratio - 0.6) * 0.5; // 0.7x to 0.5x reduction
        reduction.max(0.3)
    } else if average_luminance < 0.3 {
        // Generally dark image
        1.0 + (0.3 - average_luminance) * 2.0 // Up to 1.6x boost
    } else if average_luminance > 0.7 {
        // Generally bright image
        1.0 - (average_luminance - 0.7) * 1.0 // Down to 0.7x
    } else {
        // Well-balanced image
        1.0
    };

    // Calculate attenuation (contrast curve)
    let contrast_ratio = if mid_tone_count > 0 {
        // Calculate contrast in mid-tones
        let mid_tone_avg = mid_tone_sum / mid_tone_count as f32;
        let mut variance = 0.0;
        let mut variance_count = 0;

        for pixel in img.pixels() {
            let luminance =
                (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32)
                    / 255.0;

            if luminance >= 0.25 && luminance <= 0.75 {
                let diff = luminance - mid_tone_avg;
                variance += diff * diff;
                variance_count += 1;
            }
        }

        if variance_count > 0 {
            (variance / variance_count as f32).sqrt()
        } else {
            0.1
        }
    } else {
        0.1
    };

    let attenuation = if contrast_ratio < 0.15 {
        // Low contrast - increase contrast curve
        0.8
    } else if contrast_ratio > 0.3 {
        // High contrast - flatten curve slightly
        1.2
    } else {
        // Good contrast - linear response
        1.0
    };

    // Determine inversion based on overall brightness
    let invert = average_luminance > 0.65 && bright_ratio > 0.4;

    (exposure, attenuation, invert)
}

fn apply_exposure_processing(luminance: f32, exposure: f32, attenuation: f32, invert: bool) -> f32 {
    // Apply exposure (like AcerolaFX)
    let mut result = (luminance * exposure).clamp(0.0, 1.0);

    // Apply attenuation curve (contrast adjustment)
    result = result.powf(attenuation);

    // Apply inversion if needed
    if invert {
        result = 1.0 - result;
    }

    // Quantize to discrete levels for cleaner ASCII output (like AcerolaFX)
    let quantized = ((result * 10.0).floor().max(0.0) - 1.0).max(0.0) / 10.0;
    quantized
}

fn select_character_with_edge(
    brightness: f32,
    edge_info: &EdgeInfo,
    char_infos: &[CharInfo],
    edge_threshold: f32,
    exposure: f32,
    attenuation: f32,
    invert_luminance: bool,
) -> char {
    if edge_info.magnitude > edge_threshold {
        // For strong edges, use directional characters
        get_edge_character(edge_info.direction)
    } else {
        // Apply exposure processing to brightness
        let normalized_brightness = brightness / 255.0;
        let processed_luminance = apply_exposure_processing(
            normalized_brightness,
            exposure,
            attenuation,
            invert_luminance,
        );

        // Map processed luminance to character index
        let char_index = (processed_luminance * (char_infos.len() - 1) as f32).round() as usize;
        char_infos[char_index.min(char_infos.len() - 1)].character
    }
}

fn get_terminal_size() -> (u32, u32) {
    if let Some((terminal_size::Width(w), terminal_size::Height(h))) =
        terminal_size::terminal_size()
    {
        (w as u32, h as u32)
    } else {
        (80, 24) // Default fallback
    }
}

fn render_to_terminal(
    img: &RgbImage,
    edge_info: &Vec<Vec<EdgeInfo>>,
    color_img: &Option<RgbImage>,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let (term_width, term_height) = get_terminal_size();
    let (img_width, img_height) = img.dimensions();

    // Stretch ASCII to fill terminal dimensions (with small margin)
    let ascii_width = term_width.saturating_sub(2); // Leave 1 char margin on each side
    let ascii_height = term_height.saturating_sub(3); // Leave margin for spacing

    let charset = get_charset_by_type(&args.charset_type);
    let char_infos = create_sorted_charset(charset);

    // Calculate sampling step sizes
    let step_x = img_width as f32 / ascii_width as f32;
    let step_y = img_height as f32 / ascii_height as f32;

    // Set terminal to black background with white text
    print!("\x1b[40m\x1b[37m\x1b[2J\x1b[H"); // Black bg, white text, clear screen, move to home

    for y in 0..ascii_height {
        for x in 0..ascii_width {
            // Sample multiple pixels for better quality
            let sample_x = (x as f32 * step_x) as u32;
            let sample_y = (y as f32 * step_y) as u32;

            let mut total_brightness = 0.0;
            let mut pixel_count = 0;

            // Sample a small area around the point for better quality
            let sample_size = 1.max((step_x.min(step_y) / 2.0) as u32).min(3); // Limit sample size

            for dy in 0..sample_size {
                for dx in 0..sample_size {
                    let px = (sample_x + dx).min(img_width - 1);
                    let py = (sample_y + dy).min(img_height - 1);

                    let pixel = img.get_pixel(px, py);
                    total_brightness += pixel[0] as f32;
                    pixel_count += 1;
                }
            }

            let avg_brightness = total_brightness / pixel_count as f32;

            // Calculate brightness for character selection from original color image
            let selection_brightness = if args.color && color_img.is_some() {
                let color_img = color_img.as_ref().unwrap();
                let (r, g, b) = get_average_color(color_img, sample_x, sample_y, sample_size);
                0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
            } else {
                avg_brightness
            };

            // Use flow-based character selection for terminal
            let ascii_char = if sample_y < edge_info.len() as u32
                && sample_x < edge_info[0].len() as u32
            {
                let edge = edge_info[sample_y as usize][sample_x as usize];

                // Use higher edge threshold in no-dither mode to reduce false edges
                let effective_threshold = if args.no_dither {
                    args.edge_threshold * 3.0 // Much higher threshold for no-dither
                } else {
                    args.edge_threshold
                };

                // For terminal, we use normal brightness mapping (not inverted)
                select_character_with_edge(
                    255.0 - selection_brightness,
                    &edge,
                    &char_infos,
                    effective_threshold,
                    args.exposure,
                    args.attenuation,
                    args.invert_luminance,
                )
            } else {
                // Fallback to simple brightness mapping using original color brightness
                let char_index = ((selection_brightness / 255.0) * (char_infos.len() - 1) as f32)
                    .round() as usize;
                char_infos[char_index.min(char_infos.len() - 1)].character
            };

            if args.color && color_img.is_some() {
                let color_img = color_img.as_ref().unwrap();
                let (r, g, b) = get_average_color(color_img, sample_x, sample_y, sample_size);

                // Calculate brightness for terminal color adjustment
                let color_brightness = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
                let brightness_factor = color_brightness / 255.0;

                // Apply color intensity with brightness-based modulation
                let intensity = args.color_intensity;
                let brightness_adjusted_intensity = intensity * (0.3 + 0.7 * brightness_factor);

                let adjusted_r = ((r as f32) * brightness_adjusted_intensity
                    + (255.0 * (1.0 - brightness_adjusted_intensity)))
                    as u8;
                let adjusted_g = ((g as f32) * brightness_adjusted_intensity
                    + (255.0 * (1.0 - brightness_adjusted_intensity)))
                    as u8;
                let adjusted_b = ((b as f32) * brightness_adjusted_intensity
                    + (255.0 * (1.0 - brightness_adjusted_intensity)))
                    as u8;

                print!(
                    "{}{}",
                    rgb_to_ansi_color(adjusted_r, adjusted_g, adjusted_b),
                    ascii_char
                );
            } else {
                print!("{}", ascii_char);
            }
        }
        println!(); // New line after each row
    }

    // Reset terminal colors
    print!("\x1b[0m");
    println!(); // Add some spacing at the end
    Ok(())
}

fn apply_grayscale_and_tone_mapping(img: RgbImage) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut grayscale = RgbImage::new(width, height);

    // Convert to grayscale first
    for (x, y, pixel) in img.enumerate_pixels() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;

        let luminance = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
        grayscale.put_pixel(x, y, Rgb([luminance, luminance, luminance]));
    }

    // Apply histogram-based auto-contrast adjustment
    apply_auto_contrast(grayscale)
}

fn apply_auto_contrast(img: RgbImage) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut histogram = [0u32; 256];

    // Build histogram
    for pixel in img.pixels() {
        histogram[pixel[0] as usize] += 1;
    }

    let total_pixels = (width * height) as f32;
    let clip_percent = 0.01; // 1% clipping on each side
    let clip_threshold = (total_pixels * clip_percent) as u32;

    // Find clipping points
    let mut low_clip = 0u8;
    let mut high_clip = 255u8;

    // Find lower clipping point
    let mut cumulative = 0u32;
    for (i, &count) in histogram.iter().enumerate() {
        cumulative += count;
        if cumulative >= clip_threshold {
            low_clip = i as u8;
            break;
        }
    }

    // Find upper clipping point
    cumulative = 0;
    for (i, &count) in histogram.iter().enumerate().rev() {
        cumulative += count;
        if cumulative >= clip_threshold {
            high_clip = i as u8;
            break;
        }
    }

    // Avoid division by zero
    if high_clip <= low_clip {
        return img;
    }

    let range = high_clip as f32 - low_clip as f32;
    let alpha = 255.0 / range;
    let beta = -(low_clip as f32) * alpha;

    let mut output = RgbImage::new(width, height);

    // Apply linear transformation: new_pixel = pixel * alpha + beta
    for (x, y, pixel) in img.enumerate_pixels() {
        let old_value = pixel[0] as f32;
        let new_value = (old_value * alpha + beta).clamp(0.0, 255.0) as u8;
        output.put_pixel(x, y, Rgb([new_value, new_value, new_value]));
    }

    output
}

fn compute_edge_information(img: &RgbImage) -> Vec<Vec<EdgeInfo>> {
    let (width, height) = img.dimensions();
    let mut edge_info = vec![
        vec![
            EdgeInfo {
                magnitude: 0.0,
                direction: 0.0
            };
            width as usize
        ];
        height as usize
    ];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let gx = sobel_x(img, x, y);
            let gy = sobel_y(img, x, y);

            let magnitude = (gx * gx + gy * gy).sqrt();
            let direction = gy.atan2(gx); // Returns angle in radians

            edge_info[y as usize][x as usize] = EdgeInfo {
                magnitude,
                direction,
            };
        }
    }

    edge_info
}

fn get_edge_character(direction: f32) -> char {
    // Convert direction to degrees and normalize to 0-180 range
    let degrees = direction.to_degrees().rem_euclid(180.0);

    match degrees {
        d if d < 22.5 || d >= 157.5 => EDGE_HORIZONTAL, // Horizontal: -
        d if d >= 22.5 && d < 67.5 => EDGE_DIAGONAL_2,  // Diagonal /
        d if d >= 67.5 && d < 112.5 => EDGE_VERTICAL,   // Vertical: |
        _ => EDGE_DIAGONAL_1,                           // Diagonal \
    }
}

fn rgb_to_ansi_color(r: u8, g: u8, b: u8) -> String {
    format!("\x1b[38;2;{};{};{}m", r, g, b)
}

fn get_average_color(img: &RgbImage, x: u32, y: u32, sample_size: u32) -> (u8, u8, u8) {
    let (width, height) = img.dimensions();
    let mut total_r = 0.0;
    let mut total_g = 0.0;
    let mut total_b = 0.0;
    let mut pixel_count = 0;

    for dy in 0..sample_size {
        for dx in 0..sample_size {
            let px = (x + dx).min(width - 1);
            let py = (y + dy).min(height - 1);

            let pixel = img.get_pixel(px, py);
            total_r += pixel[0] as f32;
            total_g += pixel[1] as f32;
            total_b += pixel[2] as f32;
            pixel_count += 1;
        }
    }

    (
        (total_r / pixel_count as f32) as u8,
        (total_g / pixel_count as f32) as u8,
        (total_b / pixel_count as f32) as u8,
    )
}

fn apply_floyd_steinberg_dithering(img: RgbImage) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut output = img.clone();

    // Convert to working with raw pixel data for efficient manipulation
    let mut pixels: Vec<Vec<f32>> = Vec::with_capacity(height as usize);

    // Convert RGB to grayscale values for dithering
    for y in 0..height {
        let mut row: Vec<f32> = Vec::with_capacity(width as usize);
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            // Use luminance calculation
            let gray = 0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
            row.push(gray);
        }
        pixels.push(row);
    }

    // Apply Floyd-Steinberg dithering
    for y in 0..height {
        for x in 0..width {
            let old_pixel = pixels[y as usize][x as usize];
            let new_pixel = if old_pixel > 127.5 { 255.0 } else { 0.0 };
            let error = old_pixel - new_pixel;

            // Set the quantized pixel
            pixels[y as usize][x as usize] = new_pixel;

            // Distribute error to neighboring pixels using Floyd-Steinberg pattern
            // Pattern:    X   7/16
            //           3/16  5/16  1/16

            // Right pixel (7/16)
            if x + 1 < width {
                pixels[y as usize][(x + 1) as usize] += error * 7.0 / 16.0;
            }

            // Bottom row pixels
            if y + 1 < height {
                // Bottom-left pixel (3/16)
                if x > 0 {
                    pixels[(y + 1) as usize][(x - 1) as usize] += error * 3.0 / 16.0;
                }

                // Bottom pixel (5/16)
                pixels[(y + 1) as usize][x as usize] += error * 5.0 / 16.0;

                // Bottom-right pixel (1/16)
                if x + 1 < width {
                    pixels[(y + 1) as usize][(x + 1) as usize] += error * 1.0 / 16.0;
                }
            }
        }
    }

    // Convert back to RGB image
    for y in 0..height {
        for x in 0..width {
            let value = pixels[y as usize][x as usize].clamp(0.0, 255.0) as u8;
            output.put_pixel(x, y, Rgb([value, value, value]));
        }
    }

    output
}

fn apply_difference_of_gaussians(
    img: RgbImage,
    args: &Args,
) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let (width, height) = img.dimensions();
    let sigma1 = args.sigma1;
    let sigma2 = args.sigma2;

    let blur1 = gaussian_blur(&img, sigma1);
    let blur2 = gaussian_blur(&img, sigma2);

    let mut output = RgbImage::new(width, height);

    for (x, y, _) in img.enumerate_pixels() {
        let pixel1 = blur1.get_pixel(x, y);
        let pixel2 = blur2.get_pixel(x, y);

        let diff = (pixel1[0] as f32 - pixel2[0] as f32)
            .abs()
            .clamp(0.0, 255.0) as u8;
        output.put_pixel(x, y, Rgb([diff, diff, diff]));
    }

    Ok(output)
}

fn gaussian_blur(img: &RgbImage, sigma: f32) -> RgbImage {
    let (width, height) = img.dimensions();
    let kernel_size = (sigma * 2.45).ceil() as i32;
    let mut output = RgbImage::new(width, height);

    for (x, y, _) in img.enumerate_pixels() {
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;

        for ky in -kernel_size..=kernel_size {
            for kx in -kernel_size..=kernel_size {
                let px = (x as i32 + kx).clamp(0, width as i32 - 1) as u32;
                let py = (y as i32 + ky).clamp(0, height as i32 - 1) as u32;

                let distance = ((kx * kx + ky * ky) as f32).sqrt();
                let weight = gaussian_weight(sigma, distance);

                let pixel = img.get_pixel(px, py);
                sum += pixel[0] as f32 * weight;
                weight_sum += weight;
            }
        }

        let blurred = (sum / weight_sum).clamp(0.0, 255.0) as u8;
        output.put_pixel(x, y, Rgb([blurred, blurred, blurred]));
    }

    output
}

fn gaussian_weight(sigma: f32, distance: f32) -> f32 {
    let two_sigma_sq = 2.0 * sigma * sigma;
    (1.0 / (two_sigma_sq * std::f32::consts::PI).sqrt())
        * (-distance * distance / two_sigma_sq).exp()
}

fn sobel_x(img: &RgbImage, x: u32, y: u32) -> f32 {
    let tl = img.get_pixel(x - 1, y - 1)[0] as f32;
    let _tm = img.get_pixel(x, y - 1)[0] as f32;
    let tr = img.get_pixel(x + 1, y - 1)[0] as f32;
    let ml = img.get_pixel(x - 1, y)[0] as f32;
    let mr = img.get_pixel(x + 1, y)[0] as f32;
    let bl = img.get_pixel(x - 1, y + 1)[0] as f32;
    let _bm = img.get_pixel(x, y + 1)[0] as f32;
    let br = img.get_pixel(x + 1, y + 1)[0] as f32;

    (-tl - 2.0 * ml - bl + tr + 2.0 * mr + br) / 4.0 / 255.0
}

fn sobel_y(img: &RgbImage, x: u32, y: u32) -> f32 {
    let tl = img.get_pixel(x - 1, y - 1)[0] as f32;
    let tm = img.get_pixel(x, y - 1)[0] as f32;
    let tr = img.get_pixel(x + 1, y - 1)[0] as f32;
    let bl = img.get_pixel(x - 1, y + 1)[0] as f32;
    let bm = img.get_pixel(x, y + 1)[0] as f32;
    let br = img.get_pixel(x + 1, y + 1)[0] as f32;

    (-tl - 2.0 * tm - tr + bl + 2.0 * bm + br) / 4.0 / 255.0
}

fn render_to_ascii(
    img: &RgbImage,
    edge_info: &Vec<Vec<EdgeInfo>>,
    color_img: &Option<RgbImage>,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let (width, height) = img.dimensions();
    let charset = get_charset_by_type(&args.charset_type);
    let char_infos = create_sorted_charset(charset);

    // Parse colors
    let fg_color = parse_hex_color(&args.fg_color)?;
    let bg_color = parse_hex_color(&args.bg_color)?;

    // Calculate output dimensions based on block size
    let output_width = (width / args.cell_size) * args.cell_size;
    let output_height = (height / args.cell_size) * args.cell_size;

    // Create output image buffer and fill with background color
    let mut img_buffer: RgbImage = ImageBuffer::new(output_width, output_height);
    for pixel in img_buffer.pixels_mut() {
        *pixel = Rgb([bg_color.0, bg_color.1, bg_color.2]);
    }

    // Process each block
    for y in (0..output_height).step_by(args.cell_size as usize) {
        for x in (0..output_width).step_by(args.cell_size as usize) {
            // Calculate average brightness for this block
            let mut total_brightness = 0.0;
            let mut pixel_count = 0;
            let mut total_r = 0.0;
            let mut total_g = 0.0;
            let mut total_b = 0.0;

            let block_end_x = std::cmp::min(x + args.cell_size, output_width);
            let block_end_y = std::cmp::min(y + args.cell_size, output_height);

            for by in y..block_end_y {
                for bx in x..block_end_x {
                    if bx < width && by < height {
                        let pixel = img.get_pixel(bx, by);
                        total_brightness += pixel[0] as f32;
                        pixel_count += 1;

                        // Also get color information if available
                        if let Some(color_img_ref) = color_img {
                            let color_pixel = color_img_ref.get_pixel(bx, by);
                            total_r += color_pixel[0] as f32;
                            total_g += color_pixel[1] as f32;
                            total_b += color_pixel[2] as f32;
                        }
                    }
                }
            }

            let avg_brightness = if pixel_count > 0 {
                total_brightness / pixel_count as f32
            } else {
                0.0
            };

            // Calculate brightness from original color image for character selection
            let selection_brightness = if args.color && color_img.is_some() && pixel_count > 0 {
                // Use the original color image brightness for character selection
                let color_brightness = 0.299 * (total_r / pixel_count as f32)
                    + 0.587 * (total_g / pixel_count as f32)
                    + 0.114 * (total_b / pixel_count as f32);
                color_brightness
            } else {
                // Use processed image brightness
                avg_brightness
            };

            // Select character based on brightness and edge info
            let ascii_char = if (y / args.cell_size) < edge_info.len() as u32
                && (x / args.cell_size) < edge_info[0].len() as u32
            {
                let edge_y = (y / args.cell_size) as usize;
                let edge_x = (x / args.cell_size) as usize;
                let edge = edge_info[edge_y][edge_x];

                // Use higher edge threshold in no-dither mode to reduce false edges
                let effective_threshold = if args.no_dither {
                    args.edge_threshold * 3.0 // Much higher threshold for no-dither
                } else {
                    args.edge_threshold
                };

                select_character_with_edge(
                    selection_brightness,
                    &edge,
                    &char_infos,
                    effective_threshold,
                    args.exposure,
                    args.attenuation,
                    args.invert_luminance,
                )
            } else {
                // Fallback to simple brightness mapping using original image brightness
                let char_index = ((1.0 - selection_brightness / 255.0)
                    * (char_infos.len() - 1) as f32)
                    .round() as usize;
                char_infos[char_index.min(char_infos.len() - 1)].character
            };

            // Determine colors to use
            let char_color = if args.color && color_img.is_some() && pixel_count > 0 {
                let avg_r = (total_r / pixel_count as f32) as u8;
                let avg_g = (total_g / pixel_count as f32) as u8;
                let avg_b = (total_b / pixel_count as f32) as u8;

                // Calculate brightness factor for color intensity adjustment
                let brightness_factor = selection_brightness / 255.0;

                // Apply color intensity with brightness-based modulation
                let intensity = args.color_intensity;

                // Darker areas get more color intensity, brighter areas get less
                // This creates better contrast between light and dark regions
                let brightness_adjusted_intensity = intensity * (0.3 + 0.7 * brightness_factor);

                let final_r = ((avg_r as f32) * brightness_adjusted_intensity
                    + (fg_color.0 as f32) * (1.0 - brightness_adjusted_intensity))
                    as u8;
                let final_g = ((avg_g as f32) * brightness_adjusted_intensity
                    + (fg_color.1 as f32) * (1.0 - brightness_adjusted_intensity))
                    as u8;
                let final_b = ((avg_b as f32) * brightness_adjusted_intensity
                    + (fg_color.2 as f32) * (1.0 - brightness_adjusted_intensity))
                    as u8;

                (final_r, final_g, final_b)
            } else {
                fg_color
            };

            // Render the character bitmap
            render_ascii_char_to_image(
                &mut img_buffer,
                ascii_char,
                x,
                y,
                args.cell_size,
                char_color,
                bg_color,
            );
        }
    }

    // Save the image
    img_buffer.save(&args.output)?;
    println!("ASCII art image saved to: {}", args.output.display());

    Ok(())
}

fn render_ascii_char_to_image(
    img_buffer: &mut RgbImage,
    character: char,
    start_x: u32,
    start_y: u32,
    block_size: u32,
    char_color: (u8, u8, u8),
    bg_color: (u8, u8, u8),
) {
    let pattern = bitmap::get_char_pattern(character);
    let (img_width, img_height) = img_buffer.dimensions();

    let block_w = std::cmp::min(block_size, img_width - start_x);
    let block_h = std::cmp::min(block_size, img_height - start_y);

    for dy in 0..block_h {
        for dx in 0..block_w {
            let img_x = start_x + dx;
            let img_y = start_y + dy;

            if img_x < img_width && img_y < img_height {
                // Scale bitmap coordinates to 8x8 pattern
                let pattern_x = (dx * 8) / block_size;
                let pattern_y = (dy * 8) / block_size;

                if pattern_y < 8 && pattern_x < 8 {
                    let shift = 7 - pattern_x;
                    let bit = 1u8 << shift;

                    let pixel = img_buffer.get_pixel_mut(img_x, img_y);

                    if (pattern[pattern_y as usize] & bit) != 0 {
                        // Character pixel: use character color
                        *pixel = Rgb([char_color.0, char_color.1, char_color.2]);
                    } else {
                        // Background pixel: use background color
                        *pixel = Rgb([bg_color.0, bg_color.1, bg_color.2]);
                    }
                }
            }
        }
    }
}

fn apply_enhanced_contrast(img: RgbImage) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut histogram = [0u32; 256];

    // Build histogram
    for pixel in img.pixels() {
        histogram[pixel[0] as usize] += 1;
    }

    let total_pixels = (width * height) as f32;
    // More aggressive clipping for no-dither mode
    let clip_percent = 0.02; // 2% clipping on each side
    let clip_threshold = (total_pixels * clip_percent) as u32;

    // Find clipping points
    let mut low_clip = 0u8;
    let mut high_clip = 255u8;

    // Find lower clipping point
    let mut cumulative = 0u32;
    for (i, &count) in histogram.iter().enumerate() {
        cumulative += count;
        if cumulative >= clip_threshold {
            low_clip = i as u8;
            break;
        }
    }

    // Find upper clipping point
    cumulative = 0;
    for (i, &count) in histogram.iter().enumerate().rev() {
        cumulative += count;
        if cumulative >= clip_threshold {
            high_clip = i as u8;
            break;
        }
    }

    // Avoid division by zero
    if high_clip <= low_clip {
        return img;
    }

    let range = high_clip as f32 - low_clip as f32;
    let alpha = 255.0 / range;
    let beta = -(low_clip as f32) * alpha;

    let mut output = RgbImage::new(width, height);

    // Apply linear transformation with S-curve for better contrast
    for (x, y, pixel) in img.enumerate_pixels() {
        let old_value = pixel[0] as f32;
        let linear_value = (old_value * alpha + beta).clamp(0.0, 255.0);

        // Apply S-curve for better contrast
        let normalized = linear_value / 255.0;
        let s_curve = normalized * normalized * (3.0 - 2.0 * normalized); // Smoothstep
        let final_value = (s_curve * 255.0) as u8;

        output.put_pixel(x, y, Rgb([final_value, final_value, final_value]));
    }

    output
}

fn apply_gamma_correction(img: RgbImage, gamma: f32) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut output = RgbImage::new(width, height);

    // Pre-compute gamma lookup table for efficiency
    let mut gamma_lut = [0u8; 256];
    for i in 0..256 {
        let normalized = i as f32 / 255.0;
        let corrected = normalized.powf(1.0 / gamma);
        gamma_lut[i] = (corrected * 255.0).clamp(0.0, 255.0) as u8;
    }

    for (x, y, pixel) in img.enumerate_pixels() {
        let corrected_value = gamma_lut[pixel[0] as usize];
        output.put_pixel(
            x,
            y,
            Rgb([corrected_value, corrected_value, corrected_value]),
        );
    }

    output
}

fn apply_unsharp_mask(img: RgbImage) -> RgbImage {
    let (width, height) = img.dimensions();
    let blurred = gaussian_blur(&img, 1.0);
    let mut output = RgbImage::new(width, height);

    let amount = 1.5; // Sharpening strength
    let threshold = 0.05; // Minimum difference to apply sharpening

    for (x, y, pixel) in img.enumerate_pixels() {
        let original = pixel[0] as f32;
        let blurred_val = blurred.get_pixel(x, y)[0] as f32;
        let difference = original - blurred_val;

        if difference.abs() > threshold as f32 {
            let sharpened = original + (difference * amount);
            let final_value = sharpened.clamp(0.0, 255.0) as u8;
            output.put_pixel(x, y, Rgb([final_value, final_value, final_value]));
        } else {
            output.put_pixel(x, y, *pixel);
        }
    }

    output
}

fn analyze_image_characteristics(_img: &RgbImage) {
    let (width, height) = _img.dimensions();
    let total_pixels = (width * height) as f32;

    // Analyze brightness distribution
    let mut brightness_sum = 0.0;
    let mut brightness_histogram = [0u32; 256];
    let mut unique_colors = std::collections::HashSet::new();

    for pixel in _img.pixels() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;

        let brightness = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
        brightness_sum += brightness as f32;
        brightness_histogram[brightness as usize] += 1;

        unique_colors.insert((pixel[0], pixel[1], pixel[2]));
    }

    let avg_brightness = brightness_sum / total_pixels;

    // Calculate contrast (standard deviation of brightness)
    let mut variance_sum = 0.0;
    for pixel in _img.pixels() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        let brightness = 0.299 * r + 0.587 * g + 0.114 * b;
        let diff = brightness - avg_brightness;
        variance_sum += diff * diff;
    }
    let contrast = (variance_sum / total_pixels).sqrt();

    // Find brightness range
    let mut min_brightness = 255u8;
    let mut max_brightness = 0u8;
    for (i, &count) in brightness_histogram.iter().enumerate() {
        if count > 0 {
            min_brightness = min_brightness.min(i as u8);
            max_brightness = max_brightness.max(i as u8);
        }
    }

    println!("=== Image Analysis ===");
    println!("Average brightness: {:.1}", avg_brightness);
    println!("Contrast (std dev): {:.1}", contrast);
    println!("Brightness range: {} - {}", min_brightness, max_brightness);
    println!("Unique colors: {}", unique_colors.len());

    // Determine if image is likely to have issues with no-dither
    if contrast < 30.0 {
        println!("⚠️  LOW CONTRAST detected - may have issues with --no-dither");
    }
    if max_brightness - min_brightness < 100 {
        println!("⚠️  NARROW BRIGHTNESS RANGE detected - may have issues with --no-dither");
    }
    if unique_colors.len() < 50 {
        println!("⚠️  LIMITED COLOR PALETTE detected - may benefit from dithering");
    }

    println!("========================");
}
