use clap::Parser;
use photon_rs::native::open_image;
use photon_rs::PhotonImage;
use std::path::PathBuf;
use image::{ImageBuffer, Rgb, RgbImage};

#[derive(Parser)]
#[command(name = "pixas")]
#[command(about = "ASCII art generator with pencil-like drawing effects")]
pub struct Args {
    #[arg(short, long, default_value = "input.jpg")]
    pub input: PathBuf,
    
    #[arg(short, long, default_value = "output.png")]
    pub output: PathBuf,
    
    #[arg(long, default_value = " .:-=+*#%@")]
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    let img = open_image(&args.input)?;
    
    let processed_img = process_image(img, &args)?;
    
    render_to_ascii(&processed_img, &args)?;
    
    Ok(())
}

fn process_image(mut img: PhotonImage, args: &Args) -> Result<PhotonImage, Box<dyn std::error::Error>> {
    apply_grayscale_and_tone_mapping(&mut img);
    
    if !args.no_dither {
        apply_detail_extraction(&mut img);
    }
    
    apply_pencil_stroke_simulation(&mut img, args);
    
    Ok(img)
}

fn apply_grayscale_and_tone_mapping(img: &mut PhotonImage) {
    let width = img.get_width();
    let height = img.get_height();
    let mut raw_pixels = img.get_raw_pixels();
    
    for y in 0..height {
        for x in 0..width {
            let index = ((y * width + x) * 4) as usize;
            
            if index + 2 < raw_pixels.len() {
                let r = raw_pixels[index] as f32;
                let g = raw_pixels[index + 1] as f32;
                let b = raw_pixels[index + 2] as f32;
                
                let luminance = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                
                raw_pixels[index] = luminance;
                raw_pixels[index + 1] = luminance;
                raw_pixels[index + 2] = luminance;
            }
        }
    }
    
    *img = PhotonImage::new(raw_pixels, width, height);
}

fn apply_detail_extraction(img: &mut PhotonImage) {
    apply_floyd_steinberg_dithering(img);
    apply_histogram_equalization(img);
    apply_sobel_edge_detection(img);
}

fn apply_floyd_steinberg_dithering(img: &mut PhotonImage) {
    let width = img.get_width() as i32;
    let height = img.get_height() as i32;
    let mut raw_pixels = img.get_raw_pixels();
    
    for y in 0..height {
        for x in 0..width {
            let index = ((y * width + x) * 4) as usize;
            
            if index < raw_pixels.len() {
                let old_pixel = raw_pixels[index] as f32;
                let new_pixel = if old_pixel > 127.0 { 255 } else { 0 };
                let error = old_pixel - new_pixel as f32;
                
                raw_pixels[index] = new_pixel;
                
                if x + 1 < width {
                    let right_index = ((y * width + (x + 1)) * 4) as usize;
                    if right_index < raw_pixels.len() {
                        raw_pixels[right_index] = ((raw_pixels[right_index] as f32) + error * 7.0 / 16.0).clamp(0.0, 255.0) as u8;
                    }
                }
                
                if y + 1 < height {
                    if x > 0 {
                        let below_left_index = (((y + 1) * width + (x - 1)) * 4) as usize;
                        if below_left_index < raw_pixels.len() {
                            raw_pixels[below_left_index] = ((raw_pixels[below_left_index] as f32) + error * 3.0 / 16.0).clamp(0.0, 255.0) as u8;
                        }
                    }
                    
                    let below_index = (((y + 1) * width + x) * 4) as usize;
                    if below_index < raw_pixels.len() {
                        raw_pixels[below_index] = ((raw_pixels[below_index] as f32) + error * 5.0 / 16.0).clamp(0.0, 255.0) as u8;
                    }
                    
                    if x + 1 < width {
                        let below_right_index = (((y + 1) * width + (x + 1)) * 4) as usize;
                        if below_right_index < raw_pixels.len() {
                            raw_pixels[below_right_index] = ((raw_pixels[below_right_index] as f32) + error * 1.0 / 16.0).clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }
    }
    
    *img = PhotonImage::new(raw_pixels, width as u32, height as u32);
}

fn apply_histogram_equalization(img: &mut PhotonImage) {
    let width = img.get_width();
    let height = img.get_height();
    let mut raw_pixels = img.get_raw_pixels();
    let mut histogram = [0u32; 256];
    let total_pixels = (width * height) as u32;
    
    for i in (0..raw_pixels.len()).step_by(4) {
        histogram[raw_pixels[i] as usize] += 1;
    }
    
    let mut cdf = [0u32; 256];
    cdf[0] = histogram[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + histogram[i];
    }
    
    for i in (0..raw_pixels.len()).step_by(4) {
        let old_value = raw_pixels[i];
        let new_value = ((cdf[old_value as usize] as f32 / total_pixels as f32) * 255.0) as u8;
        raw_pixels[i] = new_value;
        raw_pixels[i + 1] = new_value;
        raw_pixels[i + 2] = new_value;
    }
    
    *img = PhotonImage::new(raw_pixels, width, height);
}

fn apply_sobel_edge_detection(img: &mut PhotonImage) {
    let width = img.get_width() as i32;
    let height = img.get_height() as i32;
    let raw_pixels = img.get_raw_pixels();
    let mut edge_pixels = raw_pixels.clone();
    
    let sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    let sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
    
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx = 0.0;
            let mut gy = 0.0;
            
            for ky in -1..=1 {
                for kx in -1..=1 {
                    let pixel_index = (((y + ky) * width + (x + kx)) * 4) as usize;
                    if pixel_index < raw_pixels.len() {
                        let pixel_value = raw_pixels[pixel_index] as f32;
                        gx += pixel_value * sobel_x[(ky + 1) as usize][(kx + 1) as usize] as f32;
                        gy += pixel_value * sobel_y[(ky + 1) as usize][(kx + 1) as usize] as f32;
                    }
                }
            }
            
            let magnitude = (gx * gx + gy * gy).sqrt().min(255.0) as u8;
            let index = ((y * width + x) * 4) as usize;
            
            if index + 2 < edge_pixels.len() {
                edge_pixels[index] = magnitude;
                edge_pixels[index + 1] = magnitude;
                edge_pixels[index + 2] = magnitude;
            }
        }
    }
    
    *img = PhotonImage::new(edge_pixels, width as u32, height as u32);
}

fn apply_pencil_stroke_simulation(img: &mut PhotonImage, args: &Args) {
    let angles: Vec<f32> = args.hatch_angles
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    
    apply_gradient_based_hatching(img, &angles, args.stroke_density);
    apply_cross_hatching(img, &angles, args.stroke_density);
}

fn apply_gradient_based_hatching(img: &mut PhotonImage, _angles: &[f32], density: f32) {
    let width = img.get_width() as i32;
    let height = img.get_height() as i32;
    let mut raw_pixels = img.get_raw_pixels();
    
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let index = ((y * width + x) * 4) as usize;
            
            if index + 2 < raw_pixels.len() {
                let center = raw_pixels[index] as f32;
                let right = raw_pixels[((y * width + (x + 1)) * 4) as usize] as f32;
                let bottom = raw_pixels[(((y + 1) * width + x) * 4) as usize] as f32;
                
                let dx = right - center;
                let dy = bottom - center;
                let gradient_magnitude = (dx * dx + dy * dy).sqrt();
                
                let stroke_intensity = (gradient_magnitude * density).min(255.0) as u8;
                let modulated_value = ((center as f32) * (1.0 - density) + (stroke_intensity as f32) * density) as u8;
                
                raw_pixels[index] = modulated_value;
                raw_pixels[index + 1] = modulated_value;
                raw_pixels[index + 2] = modulated_value;
            }
        }
    }
    
    *img = PhotonImage::new(raw_pixels, width as u32, height as u32);
}

fn apply_cross_hatching(img: &mut PhotonImage, angles: &[f32], density: f32) {
    let width = img.get_width();
    let height = img.get_height();
    let mut raw_pixels = img.get_raw_pixels();
    
    for &angle in angles {
        let rad = angle.to_radians();
        let dx = rad.cos();
        let dy = rad.sin();
        
        for y in 0..height {
            for x in 0..width {
                let index = ((y * width + x) * 4) as usize;
                
                if index + 2 < raw_pixels.len() {
                    let brightness = raw_pixels[index] as f32 / 255.0;
                    let hatch_pattern = ((x as f32 * dx + y as f32 * dy) * density).sin().abs();
                    let cross_hatch_value = (brightness * (1.0 - hatch_pattern * density)) * 255.0;
                    
                    let final_value = cross_hatch_value.clamp(0.0, 255.0) as u8;
                    raw_pixels[index] = final_value;
                    raw_pixels[index + 1] = final_value;
                    raw_pixels[index + 2] = final_value;
                }
            }
        }
    }
    
    *img = PhotonImage::new(raw_pixels, width, height);
}

fn render_to_ascii(img: &PhotonImage, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let width = img.get_width();
    let height = img.get_height();
    let raw_pixels = img.get_raw_pixels();
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
                    let index = ((cy * width + cx) * 4) as usize;
                    if index < raw_pixels.len() {
                        total_brightness += raw_pixels[index] as f32;
                        pixel_count += 1;
                    }
                }
            }
            
            let avg_brightness = if pixel_count > 0 {
                total_brightness / pixel_count as f32
            } else {
                0.0
            };
            
            // Map brightness to character
            let char_index = ((1.0 - avg_brightness / 255.0) * (charset_len - 1) as f32).round() as usize;
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
            
            render_char_bitmap(&mut img_buffer, ascii_char, start_x, start_y, 
                             char_width, char_height, brightness);
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

fn get_char_pattern(c: char) -> [u8; 8] {
    match c {
        ' ' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00],
        ':' => [0x00, 0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00],
        '=' => [0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00],
        '+' => [0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00],
        '*' => [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00],
        '#' => [0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00],
        '%' => [0x62, 0x66, 0x0C, 0x18, 0x30, 0x66, 0x46, 0x00],
        '@' => [0x3C, 0x66, 0x6E, 0x6E, 0x60, 0x62, 0x3C, 0x00],
        '/' => [0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x00],
        '\\' => [0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x00],
        '|' => [0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00],
        '~' => [0x00, 0x00, 0x76, 0xDC, 0x00, 0x00, 0x00, 0x00],
        _ => [0x7E, 0x81, 0xA5, 0x81, 0xBD, 0x99, 0x81, 0x7E], // Default pattern
    }
}