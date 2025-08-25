# Image Adjustment Tool

An interactive Python GUI application for cropping and color correction of images with real-time preview. Built with Tkinter and OpenCV, it allows users to select document corners, apply perspective correction, adjust colors (brightness, contrast, saturation, etc.), and export results as PDF or JPG.

## Features

- **Interactive Cropping**: Select 4 corners to crop documents or images
- **A4 Mode**: Crop to standard A4 paper dimensions
- **Panning & Zoom**: Navigate large images with mouse controls
- **Color Adjustments**:
  - Brightness, Contrast, Exposure
  - Temperature & Tint
  - Saturation, Vibrance, Hue
  - Shadows & Highlights
  - Sharpness & Noise Reduction
- **Auto-Adjust**: One-click automatic image enhancement
- **Real-time Preview**: See changes instantly in side-by-side view
- **Export Options**:
  - Save adjusted image as JPG or PNG
  - Export cropped result as PDF or JPG

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Pillow (`PIL`)
- NumPy
- tkinter (usually included with Python)

Optional:
- `tkinterdnd2` for drag-and-drop support

Install dependencies with:
```bash
pip install opencv-python pillow numpy
```

## Usage

1. Run the script:
   ```bash
   python image_adjustment.py
   ```

2. **Crop Image**:
   - Click "📂 Open" to load an image
   - Select 4 corners of the document
   - Toggle "A4 Mode" for standard paper size
   - Click "✂️ Crop" to proceed to adjustments

3. **Adjust Colors**:
   - Use sliders to modify image properties
   - Click "🤖 Auto" for automatic enhancement
   - "Reset All" clears all adjustments

4. **Save Results**:
   - In cropping window: "📄 Save PDF" or "🖼️ Save JPG"
   - In adjustment window: "Save" button

## Interface

### Main Cropping Window
- Canvas for image display with zoom/pan
- Corner selection with visual overlay
- A4 mode toggle for standard dimensions

### Adjustment Window
- Side-by-side original vs adjusted preview
- Individual sliders for each parameter
- Entry fields for precise value input
- Reset buttons for individual or all adjustments

## Technical Details

- Processes cropped region in real-time
- Maintains aspect ratio during display
- Uses conservative automatic adjustments
- Supports high-DPI output (300 DPI for PDF)
- Handles both landscape and portrait orientations

## License

MIT License
