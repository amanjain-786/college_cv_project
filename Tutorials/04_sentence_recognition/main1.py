from checkingInferenceModel1 import image_to_predicted_text
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import sys
import pyperclip  # Import the pyperclip module
import os

file_path = ""
script_dir = os.getcwd()
img_sh = r"mltu\Tutorials\04_sentence_recognition\default.png"
img_sh = os.path.join(script_dir,img_sh)
img_to_be_displayed = Image.open(img_sh)
text = ""
button_config = {
        "bg": "LightGray",
        "fg": "Red",
        "font": ("Arial", 8),
    }

def copy_to_clipboard():
    if text == "" :
        messagebox.showerror("Error", "No text to copy.")
    else:
        messagebox.showinfo("Done", "Text Copied.")
    pyperclip.copy(text)


def browse_file():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

def process_image():
    global text, img_to_be_displayed
    if file_path:
        wo_correction, w_correction, img_rect = image_to_predicted_text(file_path)
        img_to_be_displayed = img_rect
        if checkbox_var.get():
            to_be_displayed = "With Correction : \n\n"
            text = w_correction
        else:
            to_be_displayed = "Without Correction : \n\n"
            text = wo_correction
        to_be_displayed += text
        output_text.delete(1.0, tk.END)  # Clear previous content
        output_text.insert(tk.END, to_be_displayed)
        display_image()
    else:
        messagebox.showerror("Error", "No image selected.")

def display_image():
    global img_to_be_displayed
    img_to_be_displayed.thumbnail((350, 350))
    photo = ImageTk.PhotoImage(img_to_be_displayed)
    image_label.config(image=photo)
    image_label.image = photo


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()


root = tk.Tk()
root.title("Image Processing")
root.protocol("WM_DELETE_WINDOW", on_closing)  # Handle window closing event

main_frame = ttk.Frame(root)
main_frame.pack(padx=10, pady=10)


top_frame = ttk.Frame(main_frame)
top_frame.pack()

bottom_frame = ttk.Frame(main_frame)
bottom_frame.pack()

left_frame = ttk.Frame(bottom_frame)
left_frame.pack(side="left")

right_frame = ttk.Frame(bottom_frame)
right_frame.pack(side="left")

checkbox_var = tk.IntVar()
checkbox = ttk.Checkbutton(top_frame, text="With Correction", variable=checkbox_var)
checkbox.pack(padx=5, pady=5)

label = ttk.Label(top_frame, text="Select an image:")
label.pack()

global entry
entry = ttk.Entry(top_frame, width=40)
entry.pack()

browse_button = ttk.Button(top_frame, text="Browse", command=browse_file)
browse_button.pack(padx=5, pady=5)

run_button = ttk.Button(top_frame, text="Run", command=process_image)
run_button.pack(padx=5, pady=5)

global image_label
image_label = ttk.Label(left_frame)
image_label.pack(padx=5, pady=5)

img_to_be_displayed.thumbnail((350, 350))
photo = ImageTk.PhotoImage(img_to_be_displayed)
image_label.config(image=photo)
image_label.image = photo

global output_text
output_text = tk.Text(right_frame, height=15, width=50)
output_text.pack(padx=5, pady=5)

copy_button = ttk.Button(bottom_frame, text="Copy to Clipboard", command=copy_to_clipboard)
copy_button.place(relx=0.97, rely=0.97, anchor="se")
#copy_button.pack(side='bottom', padx=5, pady=10)

root.geometry("900x500")  # Set your desired size here

root.mainloop()
sys.exit()