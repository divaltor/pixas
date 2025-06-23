# Pixas

ASCII art generator with advanced image processing features.

## 🚀 Quick Start

### Build & Install
```bash
# Clone and build
git clone https://github.com/divaltor/pixas

cd pixas

cargo build --release

# Binary will be in target/release/pixas
```

### Basic Usage
```bash
# Convert image to ASCII art
./target/release/pixas -i input.jpg -o output.png

# Display in terminal
./target/release/pixas -i input.jpg --terminal

# With color output
./target/release/pixas -i input.jpg --color --terminal
```

## 📖 Features

### Core Features
- **Multiple Character Sets**: ASCII, block characters, or full character range
- **Edge Detection**: Sobel edge detection with directional character mapping  
- **Color Support**: RGB color preservation with intensity control
- **Terminal Output**: Direct terminal rendering with ANSI colors
- **Content-Aware Processing**: Automatic exposure and contrast adjustment

### Advanced Processing
- **Floyd-Steinberg Dithering**: High-quality dithering for better detail
- **Gaussian Blur & DoG**: Difference of Gaussians for enhanced edges
- **Gamma Correction**: Customizable gamma curves
- **Unsharp Masking**: Edge enhancement and sharpening

## 🛠️ Command Line Options

### Basic Options
- `-i, --input <FILE>` - Input image (default: input.jpg)
- `-o, --output <FILE>` - Output image (default: output.png)  
- `--terminal` - Display in terminal instead of saving image
- `--cell-size <SIZE>` - Character cell size in pixels (default: 8)

### Character Sets
- `--charset-type <TYPE>` - Character set: ascii, block, full (default: ascii)

### Image Processing
- `--no-dither` - Disable Floyd-Steinberg dithering (recommended for monochrome output)
- `--upscale-factor <FACTOR>` - Scale input image (default: 1.0)
- `--gamma <VALUE>` - Gamma correction (default: 1.0)
- `--exposure <VALUE>` - Exposure adjustment (default: 1.0, auto-calculated)
- `--attenuation <VALUE>` - Contrast curve (default: 1.0, auto-calculated)
- `--invert-luminance` - Invert brightness mapping

### Edge Detection
- `--edge-threshold <VALUE>` - Edge detection sensitivity (default: 0.1)
- `--sigma1 <VALUE>` - First Gaussian sigma for DoG (default: 1.0)
- `--sigma2 <VALUE>` - Second Gaussian sigma for DoG (default: 1.6)

### Color Options
- `--color` - Enable color output
- `--color-intensity <VALUE>` - Color intensity factor (default: 1.0)
- `--fg-color <HEX>` - Foreground color (default: #2d2d2d)
- `--bg-color <HEX>` - Background color (default: #15091b)

## 💡 Examples

```bash
# High-quality ASCII with dithering
./target/release/pixas -i photo.jpg --charset-type full --color --terminal

# Block-style output with custom colors
./target/release/pixas -i image.png --charset-type block --fg-color "#ffffff" --bg-color "#000000"

# Edge-enhanced processing
./target/release/pixas -i input.jpg --no-dither --edge-threshold 0.05 --gamma 1.2

# Custom cell size for detailed output
./target/release/pixas -i large_image.jpg --cell-size 4 -o detailed_output.png
```

## 🔧 Development

```bash
# Development build
cargo build

# Optimized release build  
cargo build --release
```

## 📄 License

Check LICENSE file