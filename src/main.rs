use clap::Parser;
use image::{ImageBuffer, Rgb, RgbImage, DynamicImage};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "pixas")]
#[command(about = "ASCII art generator with pencil-like drawing effects")]
pub struct Args {
    #[arg(short, long, default_value = "input.jpg")]
    pub input: PathBuf,

    #[arg(short, long, default_value = "output.png")]
    pub output: PathBuf,

    #[arg(long, default_value = " ░▒▓█")]
    pub charset: String,

    #[arg(long, default_value = "8")]
    pub cell_size: u32,

    #[arg(long, default_value = "16")]
    pub font_size: u32,

    #[arg(long)]
    pub no_dither: bool,

    #[arg(long, default_value = "0.7")]
    pub stroke_density: f32,

    #[arg(long, default_value = "0,45,90")]
    pub hatch_angles: String,

    #[arg(long, default_value = "2.0")]
    pub upscale_factor: f32,
    
    #[arg(long)]
    pub terminal: bool,
    
    #[arg(long, default_value = "block")]
    pub charset_type: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let img = image::open(&args.input)?.to_rgb8();

    let upscaled_img = if args.upscale_factor > 1.0 {
        upscale_image(img, args.upscale_factor)?
    } else {
        img
    };

    let processed_img = process_image(upscaled_img, &args)?;

    if args.terminal {
        render_to_terminal(&processed_img, &args)?;
    } else {
        render_to_ascii(&processed_img, &args)?;
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

fn process_image(img: RgbImage, args: &Args) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let mut processed = apply_grayscale_and_tone_mapping(img);

    if !args.no_dither {
        processed = apply_floyd_steinberg_dithering(processed);
        processed = apply_difference_of_gaussians(processed, args)?;
    }

    processed = apply_edge_tangent_flow(processed, args)?;

    Ok(processed)
}

const DEFAULT_BLOCK: &str = " .:coPO?@█";
const DEFAULT_ASCII: &str = " .:-=+*%@#";
const FULL_CHARACTERS: &str = " .-:=+iltIcsv1x%7aejorzfnuCJT3*69LYpqy25SbdgFGOVXkPhmw48AQDEHKUZR@B#NW0M";

fn get_charset_by_type(charset_type: &str) -> &str {
    match charset_type {
        "block" => DEFAULT_BLOCK,
        "ascii" => DEFAULT_ASCII,
        "full" => FULL_CHARACTERS,
        _ => DEFAULT_BLOCK,
    }
}

fn get_terminal_size() -> (u32, u32) {
    if let Some((terminal_size::Width(w), terminal_size::Height(h))) = terminal_size::terminal_size() {
        (w as u32, h as u32)
    } else {
        (80, 24) // Default fallback
    }
}

fn render_to_terminal(img: &RgbImage, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let (term_width, term_height) = get_terminal_size();
    let (img_width, img_height) = img.dimensions();
    
    // Stretch ASCII to fill terminal dimensions (with small margin)
    let ascii_width = term_width.saturating_sub(2); // Leave 1 char margin on each side
    let ascii_height = term_height.saturating_sub(3); // Leave margin for spacing
    
    let charset = if args.charset != " ░▒▓█" {
        args.charset.as_str()
    } else {
        get_charset_by_type(&args.charset_type)
    };
    
    let charset_chars: Vec<char> = charset.chars().collect();
    let charset_len = charset_chars.len();
    
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
            
            // Map brightness to character (normal mapping for white text on black background)
            let char_index = ((avg_brightness / 255.0) * (charset_len - 1) as f32).round() as usize;
            let ascii_char = charset_chars[char_index.min(charset_len - 1)];
            
            print!("{}", ascii_char);
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
    let mut output = RgbImage::new(width, height);

    for (x, y, pixel) in img.enumerate_pixels() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;

        let luminance = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
        output.put_pixel(x, y, Rgb([luminance, luminance, luminance]));
    }

    output
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
    _args: &Args,
) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let (width, height) = img.dimensions();
    let sigma1 = 1.0f32;
    let sigma2 = sigma1 * 1.6;

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

fn apply_edge_tangent_flow(
    img: RgbImage,
    _args: &Args,
) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let (width, height) = img.dimensions();
    let structure_tensor = compute_structure_tensor(&img);
    let flow_field = compute_edge_tangent_flow(&structure_tensor);

    let mut output = RgbImage::new(width, height);

    for (x, y, pixel) in img.enumerate_pixels() {
        let flow = flow_field.get_pixel(x, y);
        let dx = (flow[0] as f32 - 127.5) / 127.5;
        let dy = (flow[1] as f32 - 127.5) / 127.5;

        let nx = x as f32 + dx * 2.0;
        let ny = y as f32 + dy * 2.0;

        if nx >= 0.0 && nx < width as f32 && ny >= 0.0 && ny < height as f32 {
            let sample_pixel = img.get_pixel(nx as u32, ny as u32);
            let enhanced = ((pixel[0] as f32 + sample_pixel[0] as f32) / 2.0) as u8;
            output.put_pixel(x, y, Rgb([enhanced, enhanced, enhanced]));
        } else {
            output.put_pixel(x, y, *pixel);
        }
    }

    Ok(output)
}

fn compute_structure_tensor(img: &RgbImage) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut tensor = RgbImage::new(width, height);

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let gx = sobel_x(img, x, y);
            let gy = sobel_y(img, x, y);

            let gxx = (gx * gx * 255.0) as u8;
            let gyy = (gy * gy * 255.0) as u8;
            let gxy = (gx * gy * 255.0) as u8;

            tensor.put_pixel(x, y, Rgb([gxx, gyy, gxy]));
        }
    }

    tensor
}

fn compute_edge_tangent_flow(tensor: &RgbImage) -> RgbImage {
    let (width, height) = tensor.dimensions();
    let mut flow = RgbImage::new(width, height);

    for (x, y, pixel) in tensor.enumerate_pixels() {
        let gxx = pixel[0] as f32 / 255.0;
        let gyy = pixel[1] as f32 / 255.0;
        let gxy = pixel[2] as f32 / 255.0;

        let lambda1 =
            0.5 * (gyy + gxx + (gyy * gyy - 2.0 * gxx * gyy + gxx * gxx + 4.0 * gxy * gxy).sqrt());
        let dx = gxx - lambda1;
        let dy = gxy;

        let length = (dx * dx + dy * dy).sqrt();
        let normalized_dx = if length > 0.0 { dx / length } else { 0.0 };
        let normalized_dy = if length > 0.0 { dy / length } else { 1.0 };

        let flow_x = ((normalized_dx + 1.0) * 127.5) as u8;
        let flow_y = ((normalized_dy + 1.0) * 127.5) as u8;

        flow.put_pixel(x, y, Rgb([flow_x, flow_y, 0]));
    }

    flow
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

fn render_to_ascii(img: &RgbImage, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let (width, height) = img.dimensions();
    let charset: Vec<char> = args.charset.chars().collect();
    let charset_len = charset.len();

    // Calculate ASCII grid dimensions
    let ascii_cols = (width / (args.cell_size / 2)) as u32;
    let ascii_rows = (height / args.cell_size) as u32;

    // Calculate output image dimensions
    let font_size = args.font_size;
    let char_width = (font_size as f32 * 0.6) as u32;
    let char_height = font_size;

    let output_width = ascii_cols * char_width;
    let output_height = ascii_rows * char_height;

    // Create ASCII character grid
    let mut ascii_grid = Vec::new();

    for y in (0..height).step_by(args.cell_size as usize) {
        let mut row = Vec::new();

        for x in (0..width).step_by(args.cell_size as usize / 2) {
            let mut total_brightness = 0.0;
            let mut pixel_count = 0;

            // Sample brightness from the cell
            for cy in y..std::cmp::min(y + args.cell_size, height) {
                for cx in x..std::cmp::min(x + args.cell_size / 2, width) {
                    let pixel = img.get_pixel(cx, cy);
                    total_brightness += pixel[0] as f32;
                    pixel_count += 1;
                }
            }

            let avg_brightness = if pixel_count > 0 {
                total_brightness / pixel_count as f32
            } else {
                0.0
            };

            // Map brightness to character
            let char_index =
                ((1.0 - avg_brightness / 255.0) * (charset_len - 1) as f32).round() as usize;
            let ascii_char = charset[char_index.min(charset_len - 1)];

            row.push((ascii_char, avg_brightness));
        }
        ascii_grid.push(row);
    }

    // Create output image buffer
    let mut img_buffer: RgbImage = ImageBuffer::new(output_width, output_height);

    // Fill with white background
    for pixel in img_buffer.pixels_mut() {
        *pixel = Rgb([255, 255, 255]);
    }

    // Render ASCII characters as bitmap patterns
    for (row_idx, row) in ascii_grid.iter().enumerate() {
        for (col_idx, &(ascii_char, brightness)) in row.iter().enumerate() {
            let start_x = col_idx as u32 * char_width;
            let start_y = row_idx as u32 * char_height;

            render_char_bitmap(
                &mut img_buffer,
                ascii_char,
                start_x,
                start_y,
                char_width,
                char_height,
                brightness,
            );
        }
    }

    // Save the image
    img_buffer.save(&args.output)?;
    println!("ASCII art image saved to: {}", args.output.display());

    Ok(())
}

fn render_char_bitmap(
    img_buffer: &mut RgbImage,
    character: char,
    start_x: u32,
    start_y: u32,
    char_width: u32,
    char_height: u32,
    brightness: f32,
) {
    // Simple bitmap patterns for common ASCII characters
    let pattern = get_char_pattern(character);
    let pattern_size = 8; // 8x8 bitmap

    let scale_x = char_width as f32 / pattern_size as f32;
    let scale_y = char_height as f32 / pattern_size as f32;

    for py in 0..pattern_size {
        for px in 0..pattern_size {
            let bit = (pattern[py] >> (7 - px)) & 1;
            if bit == 1 {
                // Calculate scaled position
                let img_x = start_x + (px as f32 * scale_x) as u32;
                let img_y = start_y + (py as f32 * scale_y) as u32;

                // Draw scaled pixel block
                for dy in 0..(scale_y.ceil() as u32) {
                    for dx in 0..(scale_x.ceil() as u32) {
                        let final_x = img_x + dx;
                        let final_y = img_y + dy;

                        if final_x < img_buffer.width() && final_y < img_buffer.height() {
                            // Use brightness to determine darkness
                            let color_value = (255.0 - brightness).clamp(0.0, 255.0) as u8;
                            let pixel = img_buffer.get_pixel_mut(final_x, final_y);
                            *pixel = Rgb([color_value, color_value, color_value]);
                        }
                    }
                }
            }
        }
    }
}

/// Complete 8x8 font set from font8x8 by Daniel Hepper
/// https://github.com/dhepper/font8x8
const FONT8X8_BASIC: [[u8; 8]; 128] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0000 (null)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0001
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0002
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0003
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0004
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0005
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0006
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0007
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0008
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0009
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+000A
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+000B
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+000C
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+000D
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+000E
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+000F
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0010
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0011
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0012
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0013
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0014
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0015
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0016
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0017
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0018
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0019
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+001A
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+001B
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+001C
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+001D
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+001E
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+001F
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0020 (space)
    [0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00], // U+0021 (!)
    [0x6C, 0x6C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0022 (")
    [0x6C, 0x6C, 0xFE, 0x6C, 0xFE, 0x6C, 0x6C, 0x00], // U+0023 (#)
    [0x30, 0x7C, 0xC0, 0x78, 0x0C, 0xF8, 0x30, 0x00], // U+0024 ($)
    [0x00, 0xC6, 0xCC, 0x18, 0x30, 0x66, 0xC6, 0x00], // U+0025 (%)
    [0x38, 0x6C, 0x38, 0x76, 0xDC, 0xCC, 0x76, 0x00], // U+0026 (&)
    [0x60, 0x60, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0027 (')
    [0x18, 0x30, 0x60, 0x60, 0x60, 0x30, 0x18, 0x00], // U+0028 (()
    [0x60, 0x30, 0x18, 0x18, 0x18, 0x30, 0x60, 0x00], // U+0029 ())
    [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00], // U+002A (*)
    [0x00, 0x30, 0x30, 0xFC, 0x30, 0x30, 0x00, 0x00], // U+002B (+)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x30, 0x60], // U+002C (,)
    [0x00, 0x00, 0x00, 0xFC, 0x00, 0x00, 0x00, 0x00], // U+002D (-)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x30, 0x00], // U+002E (.)
    [0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x80, 0x00], // U+002F (/)
    [0x7C, 0xC6, 0xCE, 0xDE, 0xF6, 0xE6, 0x7C, 0x00], // U+0030 (0)
    [0x30, 0x70, 0x30, 0x30, 0x30, 0x30, 0xFC, 0x00], // U+0031 (1)
    [0x78, 0xCC, 0x0C, 0x38, 0x60, 0xCC, 0xFC, 0x00], // U+0032 (2)
    [0x78, 0xCC, 0x0C, 0x38, 0x0C, 0xCC, 0x78, 0x00], // U+0033 (3)
    [0x1C, 0x3C, 0x6C, 0xCC, 0xFE, 0x0C, 0x1E, 0x00], // U+0034 (4)
    [0xFC, 0xC0, 0xF8, 0x0C, 0x0C, 0xCC, 0x78, 0x00], // U+0035 (5)
    [0x38, 0x60, 0xC0, 0xF8, 0xCC, 0xCC, 0x78, 0x00], // U+0036 (6)
    [0xFC, 0xCC, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00], // U+0037 (7)
    [0x78, 0xCC, 0xCC, 0x78, 0xCC, 0xCC, 0x78, 0x00], // U+0038 (8)
    [0x78, 0xCC, 0xCC, 0x7C, 0x0C, 0x18, 0x70, 0x00], // U+0039 (9)
    [0x00, 0x30, 0x30, 0x00, 0x00, 0x30, 0x30, 0x00], // U+003A (:)
    [0x00, 0x30, 0x30, 0x00, 0x00, 0x30, 0x30, 0x60], // U+003B (;)
    [0x18, 0x30, 0x60, 0xC0, 0x60, 0x30, 0x18, 0x00], // U+003C (<)
    [0x00, 0x00, 0xFC, 0x00, 0x00, 0xFC, 0x00, 0x00], // U+003D (=)
    [0x60, 0x30, 0x18, 0x0C, 0x18, 0x30, 0x60, 0x00], // U+003E (>)
    [0x78, 0xCC, 0x0C, 0x18, 0x30, 0x00, 0x30, 0x00], // U+003F (?)
    [0x7C, 0xC6, 0xDE, 0xDE, 0xDE, 0xC0, 0x78, 0x00], // U+0040 (@)
    [0x30, 0x78, 0xCC, 0xCC, 0xFC, 0xCC, 0xCC, 0x00], // U+0041 (A)
    [0xFC, 0x66, 0x66, 0x7C, 0x66, 0x66, 0xFC, 0x00], // U+0042 (B)
    [0x3C, 0x66, 0xC0, 0xC0, 0xC0, 0x66, 0x3C, 0x00], // U+0043 (C)
    [0xF8, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0xF8, 0x00], // U+0044 (D)
    [0xFE, 0x62, 0x68, 0x78, 0x68, 0x62, 0xFE, 0x00], // U+0045 (E)
    [0xFE, 0x62, 0x68, 0x78, 0x68, 0x60, 0xF0, 0x00], // U+0046 (F)
    [0x3C, 0x66, 0xC0, 0xC0, 0xCE, 0x66, 0x3E, 0x00], // U+0047 (G)
    [0xCC, 0xCC, 0xCC, 0xFC, 0xCC, 0xCC, 0xCC, 0x00], // U+0048 (H)
    [0x78, 0x30, 0x30, 0x30, 0x30, 0x30, 0x78, 0x00], // U+0049 (I)
    [0x1E, 0x0C, 0x0C, 0x0C, 0xCC, 0xCC, 0x78, 0x00], // U+004A (J)
    [0xE6, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0xE6, 0x00], // U+004B (K)
    [0xF0, 0x60, 0x60, 0x60, 0x62, 0x66, 0xFE, 0x00], // U+004C (L)
    [0xC6, 0xEE, 0xFE, 0xFE, 0xD6, 0xC6, 0xC6, 0x00], // U+004D (M)
    [0xC6, 0xE6, 0xF6, 0xDE, 0xCE, 0xC6, 0xC6, 0x00], // U+004E (N)
    [0x38, 0x6C, 0xC6, 0xC6, 0xC6, 0x6C, 0x38, 0x00], // U+004F (O)
    [0xFC, 0x66, 0x66, 0x7C, 0x60, 0x60, 0xF0, 0x00], // U+0050 (P)
    [0x78, 0xCC, 0xCC, 0xCC, 0xDC, 0x78, 0x1C, 0x00], // U+0051 (Q)
    [0xFC, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0xE6, 0x00], // U+0052 (R)
    [0x78, 0xCC, 0xE0, 0x70, 0x1C, 0xCC, 0x78, 0x00], // U+0053 (S)
    [0xFC, 0xB4, 0x30, 0x30, 0x30, 0x30, 0x78, 0x00], // U+0054 (T)
    [0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xFC, 0x00], // U+0055 (U)
    [0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0x78, 0x30, 0x00], // U+0056 (V)
    [0xC6, 0xC6, 0xC6, 0xD6, 0xFE, 0xEE, 0xC6, 0x00], // U+0057 (W)
    [0xC6, 0xC6, 0x6C, 0x38, 0x38, 0x6C, 0xC6, 0x00], // U+0058 (X)
    [0xCC, 0xCC, 0xCC, 0x78, 0x30, 0x30, 0x78, 0x00], // U+0059 (Y)
    [0xFE, 0xC6, 0x8C, 0x18, 0x32, 0x66, 0xFE, 0x00], // U+005A (Z)
    [0x78, 0x60, 0x60, 0x60, 0x60, 0x60, 0x78, 0x00], // U+005B ([)
    [0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00], // U+005C (\)
    [0x78, 0x18, 0x18, 0x18, 0x18, 0x18, 0x78, 0x00], // U+005D (])
    [0x10, 0x38, 0x6C, 0xC6, 0x00, 0x00, 0x00, 0x00], // U+005E (^)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF], // U+005F (_)
    [0x30, 0x30, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0060 (`)
    [0x00, 0x00, 0x78, 0x0C, 0x7C, 0xCC, 0x76, 0x00], // U+0061 (a)
    [0xE0, 0x60, 0x60, 0x7C, 0x66, 0x66, 0xDC, 0x00], // U+0062 (b)
    [0x00, 0x00, 0x78, 0xCC, 0xC0, 0xCC, 0x78, 0x00], // U+0063 (c)
    [0x1C, 0x0C, 0x0C, 0x7C, 0xCC, 0xCC, 0x76, 0x00], // U+0064 (d)
    [0x00, 0x00, 0x78, 0xCC, 0xFC, 0xC0, 0x78, 0x00], // U+0065 (e)
    [0x38, 0x6C, 0x60, 0xF0, 0x60, 0x60, 0xF0, 0x00], // U+0066 (f)
    [0x00, 0x00, 0x76, 0xCC, 0xCC, 0x7C, 0x0C, 0xF8], // U+0067 (g)
    [0xE0, 0x60, 0x6C, 0x76, 0x66, 0x66, 0xE6, 0x00], // U+0068 (h)
    [0x30, 0x00, 0x70, 0x30, 0x30, 0x30, 0x78, 0x00], // U+0069 (i)
    [0x0C, 0x00, 0x0C, 0x0C, 0x0C, 0xCC, 0xCC, 0x78], // U+006A (j)
    [0xE0, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0xE6, 0x00], // U+006B (k)
    [0x70, 0x30, 0x30, 0x30, 0x30, 0x30, 0x78, 0x00], // U+006C (l)
    [0x00, 0x00, 0xCC, 0xFE, 0xFE, 0xD6, 0xC6, 0x00], // U+006D (m)
    [0x00, 0x00, 0xF8, 0xCC, 0xCC, 0xCC, 0xCC, 0x00], // U+006E (n)
    [0x00, 0x00, 0x78, 0xCC, 0xCC, 0xCC, 0x78, 0x00], // U+006F (o)
    [0x00, 0x00, 0xDC, 0x66, 0x66, 0x7C, 0x60, 0xF0], // U+0070 (p)
    [0x00, 0x00, 0x76, 0xCC, 0xCC, 0x7C, 0x0C, 0x1E], // U+0071 (q)
    [0x00, 0x00, 0xDC, 0x76, 0x66, 0x60, 0xF0, 0x00], // U+0072 (r)
    [0x00, 0x00, 0x7C, 0xC0, 0x78, 0x0C, 0xF8, 0x00], // U+0073 (s)
    [0x10, 0x30, 0x7C, 0x30, 0x30, 0x34, 0x18, 0x00], // U+0074 (t)
    [0x00, 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0x76, 0x00], // U+0075 (u)
    [0x00, 0x00, 0xCC, 0xCC, 0xCC, 0x78, 0x30, 0x00], // U+0076 (v)
    [0x00, 0x00, 0xC6, 0xD6, 0xFE, 0xFE, 0x6C, 0x00], // U+0077 (w)
    [0x00, 0x00, 0xC6, 0x6C, 0x38, 0x6C, 0xC6, 0x00], // U+0078 (x)
    [0x00, 0x00, 0xCC, 0xCC, 0xCC, 0x7C, 0x0C, 0xF8], // U+0079 (y)
    [0x00, 0x00, 0xFC, 0x98, 0x30, 0x64, 0xFC, 0x00], // U+007A (z)
    [0x1C, 0x30, 0x30, 0xE0, 0x30, 0x30, 0x1C, 0x00], // U+007B ({)
    [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00], // U+007C (|)
    [0xE0, 0x30, 0x30, 0x1C, 0x30, 0x30, 0xE0, 0x00], // U+007D (})
    [0x76, 0xDC, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+007E (~)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+007F
];

/// Block characters (U+2580 - U+259F)
const FONT8X8_BLOCK: [[u8; 8]; 32] = [
    [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00], // U+2580 (top half)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF], // U+2581 (box 1/8)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF], // U+2582 (box 2/8)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF], // U+2583 (box 3/8)
    [0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF], // U+2584 (bottom half)
    [0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF], // U+2585 (box 5/8)
    [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF], // U+2586 (box 6/8)
    [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF], // U+2587 (box 7/8)
    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF], // U+2588 (solid)
    [0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F], // U+2589 (box 7/8)
    [0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F], // U+258A (box 6/8)
    [0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F], // U+258B (box 5/8)
    [0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F], // U+258C (left half)
    [0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07], // U+258D (box 3/8)
    [0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03], // U+258E (box 2/8)
    [0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01], // U+258F (box 1/8)
    [0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0], // U+2590 (right half)
    [0x55, 0x00, 0xAA, 0x00, 0x55, 0x00, 0xAA, 0x00], // U+2591 (25% solid)
    [0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA], // U+2592 (50% solid)
    [0xFF, 0xAA, 0xFF, 0x55, 0xFF, 0xAA, 0xFF, 0x55], // U+2593 (75% solid)
    [0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+2594 (box 1/8)
    [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80], // U+2595 (box 1/8)
    [0x00, 0x00, 0x00, 0x00, 0x0F, 0x0F, 0x0F, 0x0F], // U+2596 (box bottom left)
    [0x00, 0x00, 0x00, 0x00, 0xF0, 0xF0, 0xF0, 0xF0], // U+2597 (box bottom right)
    [0x0F, 0x0F, 0x0F, 0x0F, 0x00, 0x00, 0x00, 0x00], // U+2598 (box top left)
    [0x0F, 0x0F, 0x0F, 0x0F, 0xFF, 0xFF, 0xFF, 0xFF], // U+2599 (boxes left and bottom)
    [0x0F, 0x0F, 0x0F, 0x0F, 0xF0, 0xF0, 0xF0, 0xF0], // U+259A (boxes top-left and bottom right)
    [0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x0F, 0x0F, 0x0F], // U+259B (boxes top and left)
    [0xFF, 0xFF, 0xFF, 0xFF, 0xF0, 0xF0, 0xF0, 0xF0], // U+259C (boxes top and right)
    [0xF0, 0xF0, 0xF0, 0xF0, 0x00, 0x00, 0x00, 0x00], // U+259D (box top right)
    [0xF0, 0xF0, 0xF0, 0xF0, 0x0F, 0x0F, 0x0F, 0x0F], // U+259E (boxes top right and bottom left)
    [0xF0, 0xF0, 0xF0, 0xF0, 0xFF, 0xFF, 0xFF, 0xFF], // U+259F (boxes right and bottom)
];

fn get_char_pattern(c: char) -> [u8; 8] {
    let code = c as u32;

    // Block characters (U+2580 - U+259F)
    if code >= 0x2580 && code <= 0x259F {
        return FONT8X8_BLOCK[(code - 0x2580) as usize];
    }

    // Basic ASCII characters (U+0000 - U+007F)
    if code <= 0x007F {
        return FONT8X8_BASIC[code as usize];
    }

    // Default pattern for unsupported characters
    [0x7E, 0x81, 0xA5, 0x81, 0xBD, 0x99, 0x81, 0x7E]
}
