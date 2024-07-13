import tkinter as tk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ASCII import ascii_filter
from default_parameters import DEFAULT_PARAMETERS


class ImageProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processing Application")
        self.geometry("1200x800")

        # Initialize variables
        self.current_image = None
        self.processing_label = None
        self.parameter_entries = {}
        self.image_filename = None

        # Define parameters
        self.PARAMETERS = DEFAULT_PARAMETERS

        # Create main canvas to display image
        self.canvas = tk.Canvas(self, bg="gray")
        self.canvas.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Placeholder for the image
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW)

        # Frame for parameter controls
        self.parameters_frame = tk.Frame(self)
        self.parameters_frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.Y)

        # Label for processing indication

        # Create parameter inputs
        self.create_parameter_inputs()

        # Button to load a new image
        self.load_button = tk.Button(
            self.parameters_frame, text="Load Image", command=self.open_file
        )
        self.load_button.grid(
            row=len(self.PARAMETERS),
            column=0,
            columnspan=2,
            padx=2,
            pady=2,
            sticky=tk.W + tk.E,
        )

        # Button to save optimal parameters
        self.save_button = tk.Button(
            self.parameters_frame,
            text="Save Optimal Parameters",
            command=self.save_parameters,
        )
        self.save_button.grid(
            row=len(self.PARAMETERS) + 1,
            column=0,
            columnspan=2,
            padx=2,
            pady=2,
            sticky=tk.W + tk.E,
        )

        # Button to process image
        self.process_button = tk.Button(
            self.parameters_frame,
            text="Process Image",
            command=self.process_image_with_parameters,
        )
        self.process_button.grid(
            row=len(self.PARAMETERS) + 2,
            column=0,
            columnspan=2,
            padx=2,
            pady=2,
            sticky=tk.W + tk.E,
        )

        # Label for processing indication
        self.processing_label = tk.Label(
            self.parameters_frame,
            text="Processing...",
            font=("Arial", 20),
            fg="red",
            bg="gray",
        )
        self.processing_label.grid(
            row=len(self.PARAMETERS) + 3,
            column=0,
            columnspan=2,
            padx=2,
            pady=2,
            sticky=tk.W + tk.E,
        )

    def create_parameter_inputs(self):
        row = 0
        col = 0
        for i, (param, default_value) in enumerate(self.PARAMETERS.items()):
            if i % (len(self.PARAMETERS) // 1.25) == 0 and i != 0:
                row = 0
                col += 2

            label = tk.Label(
                self.parameters_frame, text=param.replace("_", " ").title()
            )
            label.grid(row=row, column=col, padx=2, pady=2, sticky=tk.W)

            if isinstance(default_value, bool):
                var = tk.IntVar(value=int(default_value))
                entry = tk.Checkbutton(self.parameters_frame, variable=var)
            else:
                var = tk.StringVar(value=str(default_value))
                entry = tk.Entry(self.parameters_frame, textvariable=var)

            entry.grid(row=row, column=col + 1, padx=2, pady=2, sticky=tk.W)
            self.parameter_entries[param] = var
            row += 1

    def open_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        if filename:
            image = cv2.imread(filename)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.update_image_display(Image.fromarray(image))
                self.image_filename = filename
                for param, var in self.parameter_entries.items():
                    if isinstance(var, tk.StringVar):
                        var.set(str(self.PARAMETERS[param]))
                    elif isinstance(var, tk.IntVar):
                        var.set(int(self.PARAMETERS[param]))
            else:
                messagebox.showerror("Error", "Failed to open the image file.")

    def update_image_display(self, image):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = image.size

        scale = min(canvas_width / img_width, canvas_height / img_height)
        resized_image = image.resize((int(img_width * scale), int(img_height * scale)))

        self.current_image = ImageTk.PhotoImage(resized_image)
        self.canvas.itemconfig(self.image_item, image=self.current_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def save_parameters(self):
        optimal_params = {}
        for param, var in self.parameter_entries.items():
            if isinstance(var, tk.StringVar):
                optimal_params[param] = var.get()
            elif isinstance(var, tk.IntVar):
                optimal_params[param] = bool(int(var.get()))

        # Save optimal_params to a file or perform any other operation

        with open("optimal_params.txt", "w") as f:
            for key, value in optimal_params.items():
                f.write(f"{key}={value}\n")
        messagebox.showinfo("Saved!")

    def process_image_with_parameters(self):
        try:
            self.processing_label.grid()

            kwargs = {}
            for param, var in self.parameter_entries.items():
                if isinstance(var, tk.StringVar):
                    value = var.get()
                    if value.isdigit():
                        kwargs[param] = int(value)
                    else:
                        try:
                            kwargs[param] = float(value)
                        except ValueError:
                            kwargs[param] = value
                elif isinstance(var, tk.IntVar):
                    kwargs[param] = bool(int(var.get()))

            if self.image_filename:
                processed_image = ascii_filter(self.image_filename, **kwargs)
                self.update_image_display(Image.fromarray(processed_image))

        except Exception as e:
            messagebox.showerror("Error-process img with para", str(e))

        finally:
            self.processing_label.grid_remove()


if __name__ == "__main__":
    app = ImageProcessingApp()
    app.mainloop()
