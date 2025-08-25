import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageStat, ImageEnhance
import os
from math import atan2, degrees
import time

class ImageProcessor:
    def __init__(self):
        self.adjustments = {
            'brightness': 0,
            'contrast': 1,
            'saturation': 1,
            'exposure': 0,
            'temperature': 0,
            'tint': 0,
            'shadows': 0,
            'highlights': 0,
            'vibrance': 0,
            'hue': 0,
            'sharpness': 0,
            'noise_reduction': 0
        }
    
    def analyze_image(self, image):
        """Analyze image and suggest automatic adjustments"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image
        
        # Calculate histogram statistics
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Calculate percentiles for better analysis
        p1, p5, p95, p99 = np.percentile(gray, [1, 5, 95, 99])
        mean_brightness = np.mean(gray)
        
        # Calculate contrast using histogram spread
        contrast_measure = p95 - p5
        
        # Analyze color channels
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        saturation_mean = np.mean(hsv[:,:,1])
        
        # Suggest conservative adjustments
        adjustments = {}
        
        # Brightness adjustment - more conservative
        if mean_brightness < 80:  # Very dark
            adjustments['brightness'] = min(0.15, (100 - mean_brightness) / 400.0)
        elif mean_brightness > 200:  # Very bright
            adjustments['brightness'] = max(-0.15, (150 - mean_brightness) / 400.0)
        else:
            adjustments['brightness'] = 0
            
        # Contrast adjustment - enhance if low contrast
        if contrast_measure < 100:  # Low contrast
            adjustments['contrast'] = min(0.2, (120 - contrast_measure) / 500.0)
        else:
            adjustments['contrast'] = 0
            
        # Shadow lifting - only if very dark shadows
        if p5 < 30:  # Very dark shadows
            adjustments['shadows'] = min(0.15, (40 - p5) / 200.0)
        else:
            adjustments['shadows'] = 0
            
        # Highlight recovery - only if blown highlights
        if p95 > 240:  # Blown highlights
            adjustments['highlights'] = max(-0.15, (220 - p95) / 200.0)
        else:
            adjustments['highlights'] = 0
            
        # Saturation - subtle enhancement
        if saturation_mean < 0.4:  # Low saturation
            adjustments['saturation'] = min(0.1, (0.5 - saturation_mean) / 2.0)
        else:
            adjustments['saturation'] = 0
            
        # Set other adjustments to 0
        for key in self.adjustments:
            if key not in adjustments:
                adjustments[key] = 0
                
        return adjustments
        
    def apply_adjustments(self, image):
        img = image.copy()
        img = img.astype(np.float32) / 255.0
        
        # Exposure
        img = img * (2 ** self.adjustments['exposure'])
        
        # Temperature
        if self.adjustments['temperature'] != 0:
            temp = self.adjustments['temperature']
            blue = 1 - temp if temp > 0 else 1
            red = 1 + temp if temp > 0 else 1
            img[:,:,2] *= blue
            img[:,:,0] *= red
            
        # Tint
        if self.adjustments['tint'] != 0:
            tint = self.adjustments['tint']
            green = 1 + tint if tint > 0 else 1
            magenta = 1 - tint if tint > 0 else 1
            img[:,:,1] *= green
            img[:,:,[0,2]] *= magenta
            
        # Brightness and Contrast
        img = cv2.addWeighted(
            img,
            1 + self.adjustments['contrast'],
            np.zeros_like(img),
            0,
            self.adjustments['brightness']
        )
        
        # Saturation
        if self.adjustments['saturation'] != 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:,:,1] = hsv[:,:,1] * (1 + self.adjustments['saturation'])
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Shadows and Highlights
        if self.adjustments['shadows'] != 0 or self.adjustments['highlights'] != 0:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            shadows_mask = (gray < 0.5).astype(np.float32)
            highlights_mask = (gray >= 0.5).astype(np.float32)
            
            # Expand masks to match image dimensions
            shadows_mask = np.expand_dims(shadows_mask, axis=2)
            highlights_mask = np.expand_dims(highlights_mask, axis=2)
            
            img = img * (1 + shadows_mask * self.adjustments['shadows'])
            img = img * (1 + highlights_mask * self.adjustments['highlights'])
        
        # Vibrance
        if self.adjustments['vibrance'] != 0:
            saturation = np.max(img, axis=2) - np.min(img, axis=2)
            saturation_mask = (1 - saturation) * self.adjustments['vibrance']
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:,:,1] = hsv[:,:,1] * (1 + saturation_mask)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
        # Hue rotation
        if self.adjustments['hue'] != 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:,:,0] = (hsv[:,:,0] + self.adjustments['hue']) % 1.0
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
        # Sharpness
        if self.adjustments['sharpness'] > 0:
            blur = cv2.GaussianBlur(img, (0,0), 3)
            img = cv2.addWeighted(img, 1 + self.adjustments['sharpness'], 
                                blur, -self.adjustments['sharpness'], 0)
            
        # Noise reduction
        if self.adjustments['noise_reduction'] > 0:
            img = cv2.fastNlMeansDenoisingColored(
                (img * 255).astype(np.uint8),
                None,
                self.adjustments['noise_reduction'] * 10,
                self.adjustments['noise_reduction'] * 10,
                7,
                21
            ) / 255.0
            
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)
    
class ColorAdjustmentInterface(tk.Toplevel):
    def __init__(self, parent, image):
        super().__init__(parent)
        self.title("Color Correction - Cropped Image")
        self.geometry("1100x800")  # Increased height to prevent buttons from being cut off
        
        self.original_image = image  # This is now the cropped image
        self.processor = ImageProcessor()
        self.pending_update = False
        self.processing = True
        
        self.create_interface()
        # Schedule the original image display after the window is properly initialized
        self.after(100, self.display_original_image)
        
    def create_interface(self):
        # Main container
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image preview frame
        preview_frame = ttk.Frame(main_container)
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add labels for the preview canvases
        original_label = ttk.Label(preview_frame, text="Original (Cropped)", font=('Arial', 10, 'bold'))
        original_label.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=(0,5))
        
        # Original image preview
        self.original_canvas = tk.Canvas(preview_frame, bg='#2c2c2c')
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        adjusted_label = ttk.Label(preview_frame, text="Adjusted", font=('Arial', 10, 'bold'))
        adjusted_label.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=(0,5))
        
        # Adjusted image preview
        self.adjusted_canvas = tk.Canvas(preview_frame, bg='#2c2c2c')
        self.adjusted_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Controls frame
        controls_frame = ttk.Frame(main_container, padding="10")
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Slider configurations
        sliders_config = [
            ("Brightness", -100, 100, 0),
            ("Contrast", -100, 100, 0),
            ("Exposure", -100, 100, 0),
            ("Temperature", -100, 100, 0),
            ("Tint", -100, 100, 0),
            ("Saturation", -100, 100, 0),
            ("Vibrance", -100, 100, 0),
            ("Shadows", -100, 100, 0),
            ("Highlights", -100, 100, 0),
            ("Hue", -180, 180, 0),
            ("Sharpness", 0, 100, 0),
            ("Noise Reduction", 0, 100, 0)
        ]
        
        # Create sliders
        self.sliders = {}
        self.slider_values = {}
        
        for name, min_val, max_val, default in sliders_config:
            frame = ttk.Frame(controls_frame)
            frame.pack(fill=tk.X, pady=3)
            
            # Top row with just the label
            top_frame = ttk.Frame(frame)
            top_frame.pack(fill=tk.X)
            
            label = ttk.Label(top_frame, text=name)
            label.pack(side=tk.LEFT)
            
            # Bottom row with slider, clickable value, and reset button
            bottom_frame = ttk.Frame(frame)
            bottom_frame.pack(fill=tk.X)
            
            slider = ttk.Scale(
                bottom_frame,
                from_=min_val,
                to=max_val,
                orient=tk.HORIZONTAL,
                command=lambda v, n=name: self.update_slider(n, v, None)
            )
            slider.set(default)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
            
            # Clickable value entry
            value_var = tk.StringVar(value="0")
            value_entry = ttk.Entry(bottom_frame, textvariable=value_var, width=6, justify='center')
            value_entry.pack(side=tk.LEFT, padx=(0,5))
            value_entry.bind('<Return>', lambda e, n=name, var=value_var: self.update_from_entry(n, var))
            value_entry.bind('<FocusOut>', lambda e, n=name, var=value_var: self.update_from_entry(n, var))
            
            # Individual reset button to the right of the value entry
            reset_btn = ttk.Button(bottom_frame, text="↺", width=3,
                                 command=lambda n=name: self.reset_individual(n))
            reset_btn.pack(side=tk.LEFT)
            
            self.sliders[name.lower()] = slider
            self.slider_values[name.lower()] = value_var
        
        # Buttons frame
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(buttons_frame, text="🤖 Auto",
                  command=self.auto_adjust).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Reset All",
                  command=self.reset_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Save",
                  command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        self.display_adjusted_image()
        
    def update_slider(self, name, value, label):
        value = float(value)
        
        # Update the corresponding entry field
        param_key = name.lower()
        if param_key in self.slider_values:
            self.slider_values[param_key].set(f"{int(value)}")
        
        param = name.lower().replace(' ', '_')
        if param == 'noise_reduction':
            param = 'noise_reduction'
        elif param == 'sharpness':
            param = 'sharpness'
        
        if param in self.processor.adjustments:
            if param == 'hue':
                normalized_value = value / 180.0  # Normalize hue to -1 to 1 range
            else:
                normalized_value = value / 100.0  # All parameters use same scale now
            
            self.processor.adjustments[param] = normalized_value
            self.schedule_preview_update()
            
    def update_from_entry(self, name, value_var):
        """Update slider from entry field input"""
        try:
            value = float(value_var.get())
            
            # Get slider limits
            param_key = name.lower()
            if param_key in self.sliders:
                slider = self.sliders[param_key]
                min_val = slider.cget('from')
                max_val = slider.cget('to')
                
                # Clamp value to slider range
                value = max(min_val, min(max_val, value))
                
                # Update slider and entry
                slider.set(value)
                value_var.set(f"{int(value)}")
                
                # Update processor
                param = name.lower().replace(' ', '_')
                if param == 'noise_reduction':
                    param = 'noise_reduction'
                elif param == 'sharpness':
                    param = 'sharpness'
                
                if param in self.processor.adjustments:
                    if param == 'hue':
                        normalized_value = value / 180.0
                    else:
                        normalized_value = value / 100.0
                    
                    self.processor.adjustments[param] = normalized_value
                    self.schedule_preview_update()
        except ValueError:
            # Reset to current slider value if invalid input
            param_key = name.lower()
            if param_key in self.sliders:
                current_value = self.sliders[param_key].get()
                value_var.set(f"{int(current_value)}")
                
    def reset_individual(self, name):
        """Reset individual adjustment to default"""
        param_key = name.lower()
        if param_key in self.sliders:
            self.sliders[param_key].set(0)
            self.slider_values[param_key].set("0")
            
            # Update processor
            param = name.lower().replace(' ', '_')
            if param == 'noise_reduction':
                param = 'noise_reduction'
            elif param == 'sharpness':
                param = 'sharpness'
            
            if param in self.processor.adjustments:
                self.processor.adjustments[param] = 0
                
            self.schedule_preview_update()
            
    def schedule_preview_update(self):
        # If there's already a pending update, skip scheduling a new one
        if self.pending_update:
            return
            
        # Schedule the update for 30ms from now (about 30 FPS)
        self.after(30, self.display_adjusted_image)
        self.pending_update = True
        
    def auto_adjust(self):
        """Apply automatic adjustments based on image analysis"""
        if self.original_image is None:
            return
            
        # Analyze the image and get suggested adjustments
        original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        suggested_adjustments = self.processor.analyze_image(self.original_image)
        
        # Apply the suggested adjustments to sliders and processor
        for param, value in suggested_adjustments.items():
            if param in self.processor.adjustments:
                self.processor.adjustments[param] = value
                
                # Update corresponding slider
                slider_name = param.replace('_', ' ').title()
                if param == 'noise_reduction':
                    slider_name = 'Noise Reduction'
                elif param == 'hue':
                    slider_name = 'Hue'
                    
                # Find the correct slider
                for name, slider in self.sliders.items():
                    if name.replace(' ', '_').lower() == param:
                        # Convert back to slider scale
                        if param == 'hue':
                            slider_value = value * 180.0
                        else:
                            slider_value = value * 100.0
                            
                        slider.set(slider_value)
                        # Update the entry field
                        if name in self.slider_values:
                            self.slider_values[name].set(f"{int(slider_value)}")
                        break
        
        # Update the preview
        self.schedule_preview_update()
        
    def display_original_image(self):
        if self.original_image is None:
            return
            
        # Wait for canvas to be properly sized
        self.after(10, self._display_original_delayed)
        
    def _display_original_delayed(self):
        if self.original_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.original_canvas.winfo_width()
        canvas_height = self.original_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet, try again
            self.after(50, self._display_original_delayed)
            return
            
        # Calculate scaling to maintain aspect ratio of the cropped image
        img_height, img_width = self.original_image.shape[:2]
        
        # Calculate scale to fit within canvas while maintaining aspect ratio
        scale_w = (canvas_width * 0.85) / img_width  # Slightly more margin for cropped images
        scale_h = (canvas_height * 0.85) / img_height
        scale = min(scale_w, scale_h)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize cropped image for display
        resized_original = cv2.resize(self.original_image, (new_width, new_height))
        original_rgb = cv2.cvtColor(resized_original, cv2.COLOR_BGR2RGB)
        self.original_photo = ImageTk.PhotoImage(image=Image.fromarray(original_rgb))
        
        # Display cropped original image
        self.original_canvas.delete("all")
        self.original_canvas.create_image(
            canvas_width//2, canvas_height//2,
            image=self.original_photo,
            anchor="center"
        )
        
    def display_adjusted_image(self):
        # Reset the pending update flag
        self.pending_update = False
        
        if self.original_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.adjusted_canvas.winfo_width()
        canvas_height = self.adjusted_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet, try again
            self.after(30, self.display_adjusted_image)
            return
            
        # Calculate scaling to maintain aspect ratio of the cropped image
        img_height, img_width = self.original_image.shape[:2]
        
        # Calculate scale to fit within canvas while maintaining aspect ratio
        scale_w = (canvas_width * 0.85) / img_width  # Slightly more margin for cropped images
        scale_h = (canvas_height * 0.85) / img_height
        scale = min(scale_w, scale_h)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize cropped image for processing
        resized_original = cv2.resize(self.original_image, (new_width, new_height))
        original_rgb = cv2.cvtColor(resized_original, cv2.COLOR_BGR2RGB)
        
        # Process the cropped image with adjustments
        adjusted = self.processor.apply_adjustments(original_rgb)
        self.adjusted_photo = ImageTk.PhotoImage(image=Image.fromarray(adjusted))
        
        # Display the adjusted cropped image
        self.adjusted_canvas.delete("all")
        self.adjusted_canvas.create_image(
            canvas_width//2, canvas_height//2,
            image=self.adjusted_photo,
            anchor="center"
        )
        
    def reset_all(self):
        # Reset all processor adjustments to default values
        self.processor.adjustments = {
            'brightness': 0,
            'contrast': 0,  # Changed from 1 to 0 for proper reset
            'saturation': 0,  # Changed from 1 to 0 for proper reset
            'exposure': 0,
            'temperature': 0,
            'tint': 0,
            'shadows': 0,
            'highlights': 0,
            'vibrance': 0,
            'hue': 0,
            'sharpness': 0,
            'noise_reduction': 0
        }
        
        # Reset all sliders and entry fields to 0
        for name, slider in self.sliders.items():
            slider.set(0)
            self.slider_values[name].set("0")
            
        self.schedule_preview_update()
        
    def save_image(self):
        if self.original_image is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )
        
        if file_path:
            # Process full resolution image
            original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            adjusted = self.processor.apply_adjustments(original_rgb)
            
            # Save based on file extension
            if file_path.lower().endswith('.jpg'):
                cv2.imwrite(file_path, cv2.cvtColor(adjusted, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(file_path, cv2.cvtColor(adjusted, cv2.COLOR_RGB2BGR))
            
            messagebox.showinfo("Success", "Image saved successfully!")
            
    def destroy(self):
        self.processing = False
        super().destroy()

class ImageCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("Page Cropper")
        
        # Initialize variables
        self.image = None
        self.photo = None
        self.points = []
        self.scale = 1.0
        self.original_image = None
        self.pan_start = None
        self.offset_x = 0
        self.offset_y = 0
        self.cropped_image = None

        # Configure style
        style = ttk.Style()
        style.configure('Custom.TButton', padding=10)
        
        # Create main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas
        self.canvas = tk.Canvas(main_frame, cursor="cross", bg='#2c2c2c')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create bottom frame with modern styling
        button_frame = ttk.Frame(root, padding="10")
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Create buttons with Unicode symbols and tooltips
        self.create_button(button_frame, "📂 Open", self.load_image, "Open an image file")
        self.create_button(button_frame, "🔄 Reset", self.reset_points, "Reset selection points")
        
        # A4 mode checkbox
        self.a4_mode_var = tk.BooleanVar()
        a4_checkbox = ttk.Checkbutton(button_frame, text="A4 Mode", variable=self.a4_mode_var)
        a4_checkbox.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(a4_checkbox, "Crop to A4 dimensions (210mm x 297mm)")
        
        self.create_button(button_frame, "✂️ Crop", self.crop_and_adjust, "Crop and proceed to adjustments")
        self.create_button(button_frame, "📄 Save PDF", self.save_as_pdf, "Save as PDF")
        self.create_button(button_frame, "🖼️ Save JPG", self.save_as_jpg, "Save as JPG")
        
        # Status label
        self.status_label = ttk.Label(button_frame, text="Select 4 corners of the document")
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Bind events
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", self.zoom)  # Linux support
        self.canvas.bind("<Button-5>", self.zoom)  # Linux support
        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)
        
        # Enable drag and drop (optional - will work without it)
        try:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.drop_image)
        except:
            pass  # Drag and drop not available

    def create_button(self, parent, text, command, tooltip):
        btn = ttk.Button(parent, text=text, command=command, style='Custom.TButton')
        btn.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(btn, tooltip)

    def create_tooltip(self, widget, text):
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 20
            
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(self.tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1)
            label.pack()

        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
                
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    def crop_and_adjust(self):
        if len(self.points) != 4 or self.original_image is None:
            messagebox.showerror("Error", "Please select 4 corners first!")
            return
            
        # Perform the crop
        cropped = self.perform_crop()
        if cropped is not None:
            # Store the cropped image
            self.cropped_image = cropped
            # Open the color adjustment interface
            color_adjuster = ColorAdjustmentInterface(self.root, cropped)
            color_adjuster.transient(self.root)  # Make it modal
            color_adjuster.grab_set()  # Make it modal

    def perform_crop(self):
        if len(self.points) != 4 or self.original_image is None:
            return None
            
        pts = np.array(self.points, dtype="float32")
        rect = self.order_points(pts)

        # Calculate dimensions using average of opposite sides for more reliable results
        top_width = np.linalg.norm(rect[1] - rect[0])      # top edge
        bottom_width = np.linalg.norm(rect[2] - rect[3])   # bottom edge
        left_height = np.linalg.norm(rect[3] - rect[0])    # left edge
        right_height = np.linalg.norm(rect[2] - rect[1])   # right edge
        
        # Use average of opposite sides for more accurate dimensions
        avg_width = (top_width + bottom_width) / 2
        avg_height = (left_height + right_height) / 2
        
        # Check if A4 mode is enabled
        if self.a4_mode_var.get():
            # A4 dimensions in mm
            a4_width_mm = 210
            a4_height_mm = 297
            
            # Determine if the selected area is landscape or portrait based on the average dimensions
            if avg_width > avg_height:
                # Landscape - A4 landscape (297mm x 210mm)
                target_width_mm = a4_height_mm
                target_height_mm = a4_width_mm
            else:
                # Portrait - A4 portrait (210mm x 297mm)
                target_width_mm = a4_width_mm
                target_height_mm = a4_height_mm
                
            # Convert mm to pixels (assuming 300 DPI)
            # 1 inch = 25.4 mm, 300 DPI = 300 pixels per inch
            mm_to_pixels = 300 / 25.4
            target_width_px = int(target_width_mm * mm_to_pixels)
            target_height_px = int(target_height_mm * mm_to_pixels)
            
            # Set target dimensions for output
            target_width = target_width_px
            target_height = target_height_px
        else:
            # Use the actual proportions but scale to a reasonable resolution
            # Aim for around 2000-3000 pixels on the longer side for good quality
            max_dimension = 2500
            aspect_ratio = avg_width / avg_height
            
            if aspect_ratio > 1:  # Landscape (width > height)
                target_width = int(max_dimension)
                target_height = int(max_dimension / aspect_ratio)
            else:  # Portrait (height > width)
                target_height = int(max_dimension)
                target_width = int(max_dimension * aspect_ratio)
            
            # Ensure minimum dimensions
            target_width = max(target_width, 600)
            target_height = max(target_height, 600)
        
        dst = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.original_image, M, (target_width, target_height))
        
        return warped

    def add_point(self, event):
        if self.image is None:
            return
            
        if len(self.points) >= 4:
            self.points = []
            
        # Convert canvas coordinates to image coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        height, width = self.image.shape[:2]
        
        ratio = min(canvas_width/width, canvas_height/height)
        new_width = int(width * ratio * self.scale)
        new_height = int(height * ratio * self.scale)
        
        center_x = canvas_width//2 + self.offset_x
        center_y = canvas_height//2 + self.offset_y
        
        # Convert click coordinates to original image coordinates
        img_x = (event.x - center_x + new_width//2) / (ratio * self.scale)
        img_y = (event.y - center_y + new_height//2) / (ratio * self.scale)
        
        # Clamp to image bounds
        img_x = max(0, min(width-1, img_x))
        img_y = max(0, min(height-1, img_y))
        
        self.points.append([img_x, img_y])
        self.display_image()
        
        # Update status
        if len(self.points) == 4:
            self.status_label.config(text="4 corners selected. Click 'Crop' to proceed.")
        else:
            self.status_label.config(text=f"Select corner {len(self.points)+1} of 4")

    def reset_points(self):
        self.points = []
        self.cropped_image = None
        self.status_label.config(text="Select 4 corners of the document")
        self.display_image()

    def order_points(self, pts):
        # Sort points based on their x-coordinates
        x_sorted = pts[np.argsort(pts[:, 0]), :]
        
        # Get left-most and right-most points
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]
        
        # Sort left-most points by y-coordinate (top-left, bottom-left)
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        (tl, bl) = left_most
        
        # Calculate distance from top-left to right-most points
        D = np.linalg.norm(tl[np.newaxis] - right_most, axis=1)
        (br, tr) = right_most[np.argsort(D)[::-1], :]
        
        return np.array([tl, tr, br, bl], dtype="float32")

    def save_as_pdf(self):
        if self.cropped_image is None:
            if len(self.points) != 4 or self.original_image is None:
                messagebox.showerror("Error", "Please select 4 corners first!")
                return
            self.cropped_image = self.perform_crop()
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        
        if file_path:
            warped_rgb = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(warped_rgb)
            img.save(file_path, "PDF", resolution=300.0)
            self.reset_points()

    def save_as_jpg(self):
        if self.cropped_image is None:
            if len(self.points) != 4 or self.original_image is None:
                messagebox.showerror("Error", "Please select 4 corners first!")
                return
            self.cropped_image = self.perform_crop()
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg")]
        )
        
        if file_path:
            cv2.imwrite(file_path, self.cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.reset_points()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.open_image(file_path)

    def open_image(self, file_path):
        self.original_image = cv2.imread(file_path)
        self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.reset_points()
        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0
        self.display_image()

    def display_image(self):
        if self.image is None:
            return
            
        height, width = self.image.shape[:2]
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        ratio = min(canvas_width/width, canvas_height/height)
        new_width = int(width * ratio * self.scale)
        new_height = int(height * ratio * self.scale)
        
        resized = cv2.resize(self.image, (new_width, new_height))
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
        
        self.canvas.delete("all")
        
        center_x = canvas_width//2 + self.offset_x
        center_y = canvas_height//2 + self.offset_y
        
        self.canvas.create_image(center_x, center_y, image=self.photo, anchor="center")
        
        if len(self.points) > 1:
            for i in range(len(self.points)):
                p1 = self.points[i]
                p2 = self.points[(i + 1) % len(self.points)]
                
                x1 = p1[0] * ratio * self.scale + center_x - new_width//2
                y1 = p1[1] * ratio * self.scale + center_y - new_height//2
                x2 = p2[0] * ratio * self.scale + center_x - new_width//2
                y2 = p2[1] * ratio * self.scale + center_y - new_height//2
                
                self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2)
        
        for point in self.points:
            scaled_x = point[0] * ratio * self.scale + center_x - new_width//2
            scaled_y = point[1] * ratio * self.scale + center_y - new_height//2
            self.canvas.create_oval(scaled_x-5, scaled_y-5, scaled_x+5, scaled_y+5, fill="red")

    def start_pan(self, event):
        self.pan_start = (event.x, event.y)

    def pan(self, event):
        if self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.pan_start = (event.x, event.y)
            self.display_image()

    def end_pan(self, event):
        self.pan_start = None

    def zoom(self, event):
        if self.image is None:
            return
            
        if event.num == 5 or event.delta < 0:  # zoom out
            self.scale = max(0.1, self.scale - 0.1)
        else:  # zoom in
            self.scale = min(5.0, self.scale + 0.1)
            
        self.display_image()

    def drop_image(self, event):
        file_path = event.data
        if file_path.strip('{}').lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            self.open_image(file_path.strip('{}'))
        else:
            messagebox.showerror("Error", "Please drop a valid image file")

if __name__ == "__main__":
    try:
        from tkinterdnd2 import *
        root = TkinterDnD.Tk()
    except ImportError:
        # Fall back to regular Tk if tkinterdnd2 is not available
        root = tk.Tk()
    
    root.geometry("1200x800")
    app = ImageCropper(root)
    root.mainloop()
