import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import json
import os

class Annotator:
    def __init__(self, root):
        """
        Initialize the Annotator class and set up the GUI and event bindings.
        """
        self.root = root
        self.root.title("OCR Annotation Tool")

        # Set the window to full screen
        self.root.attributes('-fullscreen', True)
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.quit_fullscreen)

        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image = None
        self.tk_image = None
        self.original_image = None  # To store the image in its original size
        self.image_id = None
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.rectangles = []
        self.rotation_count = 0  # Track how many times the image has been rotated

        self.image_folder = r"C:\Users\Hami\Downloads\Input"
        self.images = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_index = 0

        self.output_file = r"C:\Users\Hami\Downloads\output.txt"
        self.annotations = {}

        self.load_image()

        # Bind canvas events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Button-3>", self.on_right_click)  # Bind right-click event

        # Bind keyboard events
        self.root.bind("<Return>", self.on_enter_press)
        self.root.bind("<space>", self.on_space_press)
        self.root.bind("<Left>", self.on_left_press)
        self.root.bind("<Right>", self.on_right_press)

    def toggle_fullscreen(self, event=None):
        """
        Toggle fullscreen mode on F11 key press.
        """
        self.root.attributes('-fullscreen', True)

    def quit_fullscreen(self, event=None):
        """
        Exit fullscreen mode on Escape key press.
        """
        self.root.attributes('-fullscreen', False)

    def load_image(self):
        """
        Load the current image, resize it to fit the window dimensions if necessary, and display it on the canvas.
        """
        if self.image_index < len(self.images):
            image_path = self.images[self.image_index]
            self.original_image = Image.open(image_path)  # Store original image
            self.rotation_count = 0  # Reset rotation count
            self.image = self.resize_image_to_fit_window(self.original_image)
            
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.image.width, height=self.image.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.image_id = image_path
            self.root.image = self.tk_image  # Retain reference to avoid garbage collection

        else:
            print("All images annotated")
            self.root.quit()

    def resize_image_to_fit_window(self, image):
        """
        Resize the image to fit within the Tkinter window while maintaining its aspect ratio.
        """
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()

        if window_width <= 0 or window_height <= 0:
            # If the window dimensions are invalid, use a default size.
            window_width = 800  # Example default width
            window_height = 600  # Example default height

        img_width, img_height = image.size
        scaling_factor = min(window_width / img_width, window_height / img_height)
        
        # Ensure scaling_factor is positive and the new dimensions are positive
        if scaling_factor <= 0:
            scaling_factor = 1  # Prevent scaling_factor from being zero or negative
        
        new_width = int(img_width * scaling_factor)
        new_height = int(img_height * scaling_factor)
        
        # Ensure new_width and new_height are positive
        if new_width <= 0:
            new_width = 1
        if new_height <= 0:
            new_height = 1

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def on_button_press(self, event):
        """
        Start drawing a rectangle for annotation when the mouse button is pressed.
        """
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_mouse_drag(self, event):
        """
        Update the rectangle's dimensions as the mouse is dragged.
        """
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        """
        Finalize the rectangle and prompt for transcription after mouse button release.
        """
        end_x, end_y = (event.x, event.y)
        self.rectangles.append((self.start_x, self.start_y, end_x, end_y))
        transcription = self.prompt_transcription()
        if transcription:
            self.save_annotation(self.start_x, self.start_y, end_x, end_y, transcription)
        self.canvas.delete(self.rect)

    def prompt_transcription(self):
        """
        Prompt the user for transcription input.
        """
        transcription = simpledialog.askstring("Input", "Enter transcription:")
        return transcription

    def save_annotation(self, start_x, start_y, end_x, end_y, transcription):
        """
        Save the annotation details for the current image.
        """
        if self.image_id not in self.annotations:
            self.annotations[self.image_id] = []
        points = [[start_x, start_y], [end_x, start_y], [end_x, end_y], [start_x, end_y]]
        self.annotations[self.image_id].append({"transcription": transcription, "points": points})

    def on_enter_press(self, event):
        """
        Move to the next image and save the rotated image if annotations exist.
        """
        if self.rectangles:  # Check if there are any annotations
            self.save_image_if_rotated()
        self.image_index += 1
        self.rectangles = []
        if self.image_index < len(self.images):
            self.load_image()
        else:
            print("All images annotated")
            self.root.quit()

    def on_space_press(self, event):
        """
        Save annotations to file and save the rotated image if annotations exist.
        """
        self.save_to_file()
        self.save_image_if_rotated()  # Save rotated image if needed
        print(f"Annotations saved in {os.path.abspath(self.output_file)}")

    def on_left_press(self, event):
        """
        Move to the previous image and save the rotated image if annotations exist.
        """
        if self.image_index > 0:
            if self.rectangles:  # Check if there are any annotations
                self.save_image_if_rotated()
            self.image_index -= 1
            self.rectangles = []
            self.load_image()

    def on_right_press(self, event):
        """
        Move to the next image and save the rotated image if annotations exist.
        """
        if self.image_index < len(self.images) - 1:
            if self.rectangles:  # Check if there are any annotations
                self.save_image_if_rotated()
            self.image_index += 1
            self.rectangles = []
            self.load_image()

    def on_right_click(self, event):
        """
        Rotate the image 90 degrees clockwise when the right mouse button is clicked.
        """
        self.rotate_image()

    def rotate_image(self):
        """
        Rotate the image 90 degrees clockwise and update the canvas to fit the window dimensions.
        """
        if self.original_image:
            # Rotate image in increments of 90 degrees
            self.rotation_count = (self.rotation_count + 1) % 4
            self.image = self.original_image.rotate(90 * self.rotation_count, expand=True)
            self.image = self.resize_image_to_fit_window(self.image)
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.image.width, height=self.image.height)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.root.image = self.tk_image  # Retain reference to avoid garbage collection

    def save_image_if_rotated(self):
        """
        Save the rotated image to disk with its original dimensions if it has been rotated.
        """
        if self.rotation_count > 0 and self.rectangles:  # Save only if rotated and annotations exist
            # Save the rotated image with the original dimensions
            rotated_image = self.original_image.rotate(90 * self.rotation_count, expand=True)
            rotated_image.save(self.images[self.image_index])
            self.rotation_count = 0  # Reset rotation count after saving

    def save_to_file(self):
        """
        Save annotations to a text file in tab-separated format.
        """
        with open(self.output_file, 'w') as f:
            for image_id, annotation in self.annotations.items():
                f.write(f"{image_id}\t{json.dumps(annotation)}\n")

if __name__ == "__main__":
    root = tk.Tk()
    annotator = Annotator(root)
    root.mainloop()
