# Integrated screens (home page, file upload for sarcasm, & file upload for mp)

"""
Program Title: XLM-R: Sarcasm and Mock Politeness Detector

Overview:
    1. Main Window (Home Page):
        a. Displays a title, logo, and two model selection buttons ("Sarcasm Detection" and "Sarcasm and Mock Politeness Detection").
        b. Each button leads to a respective frame for file selection and classification.

UI Design and Navigation:
    1. Sarcasm Detection Frame:
        a. Contains a file selection button for uploading a CSV file.
        b. Includes an entry field to display the selected file path.
        c. A "Classify" button initiates classification.
        d. A "Back" button navigates to the home page.
    2. Sarcasm and Mock Politeness Detection Frame:
        a. Similar to the Sarcasm Detection frame but for the "Sarcasm and Mock Politeness Detection" model.
        b. Includes file selection, entry field, classify, and back buttons.
    3. UI Elements:
        a. Rounded rectangle buttons and labels organize the interface with hover effects for a polished experience.

Utility Functions and Preprocessing:
    1. File Selection: Users can select CSV files using the filedialog.askopenfilename function.
    2. Emoticon and Emoji Processing: 
        a. replace_emoticons and convert_emoji_and_emoticon functions convert emoticons and emojis to textual descriptions.
    
Data Management:
    1. Lists & Dictionaries - Store labels, detected emoticons, and classification reports.
    2. Pandas DataFrame - Manages CSV file data, including student comments and classification labels.

Machine Learning Model:
    1. Model Architecture:
        a. Multi-Task Learning Model - Uses shared encoder layers with separate classification heads for sarcasm and mock politeness.
    2. Tokenization & Embedding:
        a. Uses XLMRobertaTokenizer for text processing and feature extraction.
    3. PyTorch Tensors & Neural Network:
        a. Handles tokenized input and model predictions using XLM-RoBERTa.

Control Flow and Logic:
    1. Event-Driven Programming:
        a. Tkinter handles UI interaction and event flow.
    2. Looping & Conditional Logic:
        a. Loops for text processing and if-else conditions for classification decisions.
    3. Multithreading:
        a. Ensures UI remains responsive while processing large datasets.

Performance Evaluation:
    1. Confusion Matrix Calculation:
        a. Confusion matrix is computed and displayed after classification.
    2. Visualization:
        a. Confusion matrix results are displayed in a visual format using Seaborn and Matplotlib.

"""

import tkinter as tk
from tkinter import Canvas, filedialog
from tkinter import ttk
import threading
import emoji
import emot
import pandas as pd
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch.nn as nn
from tkinter import messagebox
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sys


def create_rounded_rectangle(canvas, x1, y1, x2, y2, radius=25, **kwargs):
    points = [x1+radius, y1, x1+radius, y1, x2-radius, y1, x2-radius, y1,
              x2, y1, x2, y1+radius, x2, y1+radius, x2, y2-radius, x2, y2-radius,
              x2, y2, x2-radius, y2, x2-radius, y2, x1+radius, y2, x1+radius, y2,
              x1, y2, x1, y2-radius, x1, y2-radius, x1, y1+radius, x1, y1+radius, x1, y1]
    return canvas.create_polygon(points, **kwargs, smooth=True)

def choose_file(entry):
    filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filename:
        entry.delete(0, tk.END)
        entry.insert(0, filename)

def on_hover(canvas, button_bg, button_text):
    canvas.itemconfig(button_bg, fill="#6c7ce0")
    canvas.itemconfig(button_text, fill="white")

def on_leave(canvas, button_bg, button_text):
    canvas.itemconfig(button_bg, fill="#3d52d5")
    canvas.itemconfig(button_text, fill="white")

def bind_button_states(canvas, button_bg, button_text):
    canvas.tag_bind(button_bg, "<Enter>", lambda e: on_hover(canvas, button_bg, button_text))
    canvas.tag_bind(button_bg, "<Leave>", lambda e: on_leave(canvas, button_bg, button_text))
    canvas.tag_bind(button_text, "<Enter>", lambda e: on_hover(canvas, button_bg, button_text))
    canvas.tag_bind(button_text, "<Leave>", lambda e: on_leave(canvas, button_bg, button_text))

def show_page(page_frame):
    home_frame.place_forget()
    sarcasm_frame.place_forget()
    mock_politeness_frame.place_forget()
    Sarcasm_and_MP_Frame.place_forget()
    Sarcasm_Frame.place_forget()
    page_frame.place(x=0, y=0, width=1440, height=1000)
    page_frame.update_idletasks()
    page_frame.place(x=0, y=0, width=1440, height=1000)      
    page_frame.lift()  # Bring the frame to the front



# Initialize the emoticon detector
emoticon_detector = emot.core.emot()

def replace_emoticons(text):
    # Detect emoticons in the text
    detected_emoticons = emoticon_detector.emoticons(text)
    # Get the lists of emoticons and their meanings
    emoticons = detected_emoticons['value']
    meanings = detected_emoticons['mean']
    
    # Loop over each emoticon and its meaning
    for i in range(len(emoticons)):
        emoticon = emoticons[i]
        description = meanings[i]
        # Replace emoticon with its description
        text = text.replace(emoticon, description)
    return text

# Integrating into the overall function
def convert_emoji_and_emoticon(text):
    text = emoji.demojize(text, delimiters=("", ""))  # Convert emojis to text
    text = replace_emoticons(text)  # Replace emoticons with descriptions
    return text


# Home page / Main window
root = tk.Tk()
root.title("XLM-RoBERTa Model Selection")
root.geometry("1440x1000")
root.resizable(False, False)
root.configure(bg='white')


#####################################################################################################################################################

# Home Page Frame
home_frame = tk.Frame(root, bg="white")
home_frame.place(x=0, y=0, width=1440, height=1000)

# Indigo rectangle (left)
indigo_frame = tk.Frame(home_frame, bg="#090c9b")
indigo_frame.place(x=0, y=0, width=475, height=1024)

# Logo
logo_image = tk.PhotoImage(file="assets/xlm_logo.png")
logo_label = tk.Label(home_frame, image=logo_image, bg="#090c9b")
logo_label.place(x=10, y=367, width=450, height=350)

# Title for the home page
title_label = tk.Label(home_frame, text="Choose a Model", font=("Inter", 60, "bold", "underline"), bg="white", fg="#3c3744")
title_label.place(x=660, y=136)

# White rectangle (right)
canvas_home = Canvas(home_frame, bg="white", highlightthickness=0)
canvas_home.place(x=490, y=280, width=900, height=550)

# Frames for labels 
create_rounded_rectangle(canvas_home, 50, 50, 375, 375, radius=30, fill="white", outline="#3d52d5", width=4)  # Frame 1
create_rounded_rectangle(canvas_home, 475, 50, 875, 375, radius=30, fill="white", outline="#3d52d5", width=4)  # Frame 2

# Labels
canvas_home.create_text(210, 200, text="Sarcasm\nDetection", font=("Inter", 30, "bold"), fill="#3c3744", justify='center')
canvas_home.create_text(675, 210, text="Sarcasm and\nMock Politeness\nDetection", font=("Inter", 30, "bold"), fill="#3c3744", justify='center')

# Button 1 (for Sarcasm Detection)
button1_bg = create_rounded_rectangle(canvas_home, 45, 470, 385, 530, radius=20, fill="#3d52d5", outline="#3d52d5")
button1_text = canvas_home.create_text(212, 500, text="Select", font=("Inter", 25, "bold"), fill="white")
bind_button_states(canvas_home, button1_bg, button1_text)

# Bind Select button (for Sarcasm Detection) 
canvas_home.tag_bind(button1_bg, "<Button-1>", lambda e: show_page(sarcasm_frame))
canvas_home.tag_bind(button1_text, "<Button-1>", lambda e: show_page(sarcasm_frame))

# Button 2 (for Sarcasm and Mock Politeness Detection)
button2_bg = create_rounded_rectangle(canvas_home, 475, 470, 875, 530, radius=20, fill="#3d52d5", outline="#3d52d5")
button2_text = canvas_home.create_text(670, 500, text="Select", font=("Inter", 25, "bold"), fill="white")
bind_button_states(canvas_home, button2_bg, button2_text)

# Bind Select button (for Sarcasm and Mock Politeness Detection)
canvas_home.tag_bind(button2_bg, "<Button-1>", lambda e: show_page(mock_politeness_frame))
canvas_home.tag_bind(button2_text, "<Button-1>", lambda e: show_page(mock_politeness_frame))



####################################################################################################################################################3


# Sarcasm Detection Frame
sarcasm_frame = tk.Frame(root, bg="white")

# Indigo rectangle (left)
indigo_frame_sarcasm = tk.Frame(sarcasm_frame, bg="#090c9b")
indigo_frame_sarcasm.place(x=0, y=0, width=475, height=1024)

# Logo 
logo_label_sarcasm = tk.Label(sarcasm_frame, image=logo_image, bg="#090c9b")
logo_label_sarcasm.place(x=10, y=367, width=450, height=350)

canvas_sarcasm = Canvas(sarcasm_frame, bg="white", highlightthickness=0)
canvas_sarcasm.place(x=490, y=150, width=900, height=800)

# Title Label 
canvas_sarcasm.create_text(480, 250, text="Sarcasm Detection", font=("Inter", 40, "bold", "underline"), fill="#3d52d5", justify='center')

# File selection button for Sarcasm Detection
choose_file_bg_sarcasm = create_rounded_rectangle(canvas_sarcasm, 200, 320, 400, 370, radius=20, fill="#3d52d5", outline="#3d52d5")
choose_file_text_sarcasm = canvas_sarcasm.create_text(300, 345, text="Choose CSV File", font=("Inter", 14, "bold"), fill="white")
bind_button_states(canvas_sarcasm, choose_file_bg_sarcasm, choose_file_text_sarcasm)

# Entry field 
entry_bg_sarcasm = create_rounded_rectangle(canvas_sarcasm, 420, 320, 820, 370, radius=20, fill="white", outline="#3d52d5")
file_entry_sarcasm = tk.Entry(canvas_sarcasm, font=("Inter", 14), width=35, bd=0, relief="flat")
file_path_sarcasm = file_entry_sarcasm.get()  
canvas_sarcasm.create_window(620, 345, window=file_entry_sarcasm)  

# Choose file bind button (Sarcasm Detection)
canvas_sarcasm.tag_bind(choose_file_bg_sarcasm, "<Button-1>", lambda e: choose_file(file_entry_sarcasm))
canvas_sarcasm.tag_bind(choose_file_text_sarcasm, "<Button-1>", lambda e: choose_file(file_entry_sarcasm))

# Classify button 
classify_bg_sarcasm = create_rounded_rectangle(canvas_sarcasm, 350, 450, 600, 500, radius=20, fill="#3d52d5", outline="#3d52d5")
classify_text_sarcasm = canvas_sarcasm.create_text(475, 475, text="Classify", font=("Inter", 16, "bold"), fill="white")
bind_button_states(canvas_sarcasm, classify_bg_sarcasm, classify_text_sarcasm)

# Back button 
back_bg_sarcasm = create_rounded_rectangle(canvas_sarcasm, 700, 700, 800, 750, radius=20, fill="#3d52d5", outline="#3d52d5")
back_text_sarcasm = canvas_sarcasm.create_text(750, 725, text="Back", font=("Inter", 14, "bold"), fill="white")
bind_button_states(canvas_sarcasm, back_bg_sarcasm, back_text_sarcasm)


canvas_sarcasm.tag_bind(back_bg_sarcasm, "<Button-1>", lambda e: show_page(home_frame))
canvas_sarcasm.tag_bind(back_text_sarcasm, "<Button-1>", lambda e: show_page(home_frame))



##############################################################################################################################################################


# Sarcasm and Mock Politeness Detection Frame
mock_politeness_frame = tk.Frame(root, bg="white")

# Indigo rectangle (left)
indigo_frame_mock = tk.Frame(mock_politeness_frame, bg="#090c9b")
indigo_frame_mock.place(x=0, y=0, width=475, height=1024)

# Logo 
logo_label_mock = tk.Label(mock_politeness_frame, image=logo_image, bg="#090c9b")
logo_label_mock.place(x=10, y=367, width=450, height=350)

canvas_mock = Canvas(mock_politeness_frame, bg="white", highlightthickness=0)
canvas_mock.place(x=490, y=150, width=900, height=800)

# Title Label 
canvas_mock.create_text(480, 220, text="Sarcasm and\nMock Politeness Detection", font=("Inter", 40, "bold", "underline"), fill="#3d52d5", justify='center')

# File selection button for Sarcasm and Mock Politeness Detection
choose_file_bg_mock = create_rounded_rectangle(canvas_mock, 200, 320, 400, 370, radius=20, fill="#3d52d5", outline="#3d52d5")
choose_file_text_mock = canvas_mock.create_text(300, 345, text="Choose CSV File", font=("Inter", 14, "bold"), fill="white")
bind_button_states(canvas_mock, choose_file_bg_mock, choose_file_text_mock)

# Entry field 
entry_bg_mock = create_rounded_rectangle(canvas_mock, 420, 320, 820, 370, radius=20, fill="white", outline="#3d52d5")
file_entry_mock = tk.Entry(canvas_mock, font=("Inter", 14), width=35, bd=0, relief="flat")  
file_path_mock = file_entry_mock.get()

canvas_mock.create_window(620, 345, window=file_entry_mock)  

# Choose file bind button (Sarcasm and Mock Politeness Detection)
canvas_mock.tag_bind(choose_file_bg_mock, "<Button-1>", lambda e: choose_file(file_entry_mock))
canvas_mock.tag_bind(choose_file_text_mock, "<Button-1>", lambda e: choose_file(file_entry_mock))

# Classify button 
classify_bg_mock = create_rounded_rectangle(canvas_mock, 350, 450, 600, 500, radius=20, fill="#3d52d5", outline="#3d52d5")
classify_text_mock = canvas_mock.create_text(475, 475, text="Classify", font=("Inter", 16, "bold"), fill="white")
bind_button_states(canvas_mock, classify_bg_mock, classify_text_mock)

# Back button 
back_bg_mock = create_rounded_rectangle(canvas_mock, 700, 700, 800, 750, radius=20, fill="#3d52d5", outline="#3d52d5")
back_text_mock = canvas_mock.create_text(750, 725, text="Back", font=("Inter", 14, "bold"), fill="white")
bind_button_states(canvas_mock, back_bg_mock, back_text_mock)

# Bind back button event to go back to the home page
canvas_mock.tag_bind(back_bg_mock, "<Button-1>", lambda e: show_page(home_frame))
canvas_mock.tag_bind(back_text_mock, "<Button-1>", lambda e: show_page(home_frame))

####################################################################################################################################
# Loading screen
def show_page_loading_screen(frame):
    # Hide all frames by using place_forget, then show the specified frame
    for widget in root.winfo_children():
        if isinstance(widget, tk.Frame):
            widget.place_forget()  # Hide all frames

    frame.place(x=0, y=0, width=1440, height=1000)  # Show the desired frame


def LoadingFrame():
    global loading_frame1 # Make these global

    # Frame 1
    loading_frame1 = tk.Frame(root, bg="white")
    loading_frame1.place(x=0, y=0, width=1440, height=1000)

    # Indigo rectangle (left) for Frame 1
    indigo_frame1 = tk.Frame(loading_frame1, bg="#090c9b")
    indigo_frame1.place(x=0, y=0, width=475, height=1024)

    # Logo in Frame 1
    logo_image = tk.PhotoImage(file="assets/xlm_logo.png")  # Make sure this path is correct
    logo_label = tk.Label(loading_frame1, image=logo_image, bg="#090c9b")
    logo_label.image = logo_image
    logo_label.place(x=10, y=367, width=450, height=350)

    canvas_loading1 = Canvas(loading_frame1, bg="white", highlightthickness=0)
    canvas_loading1.place(x=490, y=150, width=900, height=800)

    # Title for Frame 1
    title_label1 = tk.Label(loading_frame1, text="Predicting Labels...", font=("Inter", 50, "bold"), bg="white", fg="#3d52d5")
    title_label1.place(x=660, y=400)

    
    # Bind back button event to go back to the previous page

    progress_bar = ttk.Progressbar(loading_frame1, orient="horizontal", length=300, mode="indeterminate")
    progress_bar.place(x=660, y=530, width=640, height=20)

    style_loading = ttk.Style()
    style_loading.theme_use('clam')  # Use a theme that supports color changes
    style_loading.configure("custom.Horizontal.TProgressbar", 
                    troughcolor="blue",  # Background color of the progress bar
                    background="white")       # Color of the progress
    progress_bar['style'] = "custom.Horizontal.TProgressbar"

    progress_bar.start()  # Start the progress bar animation



    return loading_frame1


####################################################################################################################################
# Result Page for Sarcasm and NonSarcasm

Sarcasm_Frame = tk.Frame(root, bg="white")

# Indigo rectangle (left) with proper padding for margins
indigo_frame_Result_Sarcasm = tk.Frame(Sarcasm_Frame, bg="#090c9b")
indigo_frame_Result_Sarcasm.place(x=20, y=20, width=860, height=960)  # Adding margin to each side

# White rectangle (right)
canvas_Sarcasm_Result = Canvas(Sarcasm_Frame, bg="white", highlightthickness=0)
canvas_Sarcasm_Result.place(x=900, y=150, width=510, height=800)

# Confusion matrix button 
cm_bg_sarcasm = create_rounded_rectangle(canvas_Sarcasm_Result, 90, 400, 410, 500, radius=20, fill="#3d52d5", outline="#3d52d5")
cm_text_sarcasm = canvas_Sarcasm_Result.create_text(250, 448, text="Confusion Matrix", font=("Inter", 25 , "bold"), fill="white")
bind_button_states(canvas_Sarcasm_Result, cm_bg_sarcasm, cm_text_sarcasm)

# Bind the button click to save the file path and show the page
canvas_Sarcasm_Result.tag_bind(cm_bg_sarcasm, "<Button-1>", lambda e: stl_cm())
canvas_Sarcasm_Result.tag_bind(cm_text_sarcasm, "<Button-1>", lambda e: stl_cm())

# Title Label
canvas_Sarcasm_Result.create_text(250, 160, text="Results", font=("Inter", 30, "bold"), fill="#000000", justify='center')

# Placeholder text that will be updated dynamically
total_comments_text_sarcasm = canvas_Sarcasm_Result.create_text(250, 220, text="Total Comments = 0", font=("Inter", 20), fill="#000000", justify='center')
sarcasm_count_text_sarcasm = canvas_Sarcasm_Result.create_text(250, 260, text="Sarcasm Detection = 0", font=("Inter", 20), fill="#000000", justify='center')
non_sarcasm_count_text_sarcasm = canvas_Sarcasm_Result.create_text(250, 300, text="Non Sarcasm Detection = 0", font=("Inter", 20), fill="#000000", justify='center')

# Create the Back to Home button 
back_home_bg = create_rounded_rectangle(canvas_Sarcasm_Result, 90, 520, 410, 620, radius=20,
                                         fill="#3d52d5", outline="#3d52d5")
back_home_text = canvas_Sarcasm_Result.create_text(250, 570, text="Back to Home", font=("Inter", 25, "bold"), fill="white")

# Bind button states for hover effects and interactions
bind_button_states(canvas_Sarcasm_Result, back_home_bg, back_home_text)

# Bind the Back to Home button to show the home page
canvas_Sarcasm_Result.tag_bind(back_home_bg, "<Button-1>", lambda e: show_page(home_frame))
canvas_Sarcasm_Result.tag_bind(back_home_text, "<Button-1>", lambda e: show_page(home_frame))

# Display the result frame
Sarcasm_Frame.pack(fill=tk.BOTH, expand=True)

def sarcasm_model(file):
    

    # Load your dataset
    dataset = pd.read_csv(file)
    global df, pred_sarcasm_labels
    df = dataset

    # Apply the conversion function to the 'Value' column in your dataset
    df['Student_Comment'] = df['Student_Comment'].apply(convert_emoji_and_emoticon)

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer and model for XLM-Roberta
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    

    # Define your custom model class for sarcasm detection
    class CustomXLMRobertaModel(nn.Module):
        def __init__(self, dropout_prob=0.3):
            super(CustomXLMRobertaModel, self).__init__()
            self.xlm_roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")

            # Dropout layer
            self.dropout = nn.Dropout(dropout_prob)

            # Hidden layer size (adjust as needed)
            hidden_layer_size = 128  # Adjust the size of the hidden layer as you see fit

            # Sarcasm classification head with an extra hidden layer
            self.classifier_sarcasm = nn.Sequential(
                nn.Linear(self.xlm_roberta.config.hidden_size, hidden_layer_size),  # First hidden layer
                nn.ReLU(),  # Activation function
                nn.Linear(hidden_layer_size, 1)  # Output layer for binary classification
            )

            # Loss functions
            self.criterion_sarcasm = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification

        def forward(self, input_ids, attention_mask, labels_sarcasm=None):
            outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)

            # Get the CLS token output (first token's hidden state)
            cls_token_output = outputs[0][:, 0, :]  # Shape: [batch_size, hidden_size]

            # Apply dropout to CLS token output
            cls_token_output = self.dropout(cls_token_output)

            # Sarcasm classification head
            logits_sarcasm = self.classifier_sarcasm(cls_token_output).squeeze(-1)  # Shape: [batch_size]

            loss = None
            if labels_sarcasm is not None :
                # Compute the loss for both tasks
                loss_sarcasm = self.criterion_sarcasm(logits_sarcasm, labels_sarcasm.float())  # Binary cross-entropy loss

                # Combine losses (you can adjust the weights of the losses if needed)
                loss = loss_sarcasm

            return loss, logits_sarcasm

    # Initialize the model
    model = CustomXLMRobertaModel(dropout_prob=0.3)

    # Load the tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # Load your trained model state
    try:
        model.load_state_dict(torch.load(r"best_models\best_model_stl_7.pt", map_location=device))
    except RuntimeError as e:
        print(f"Error loading model state: {e}")

    # Move model to the defined device
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    global true_sarcasm_labels, pred_sarcasm_labels, pred_sarcasm_label
    # Initialize lists to hold predicted sarcasm labels
    pred_sarcasm_labels = []
    true_sarcasm_labels= df['Sarcasm_Label'].tolist()  # True labels for mock politeness

    # Iterate over the DataFrame and get predictions
    with torch.no_grad():  # Disable gradient calculation
        for index, row in df.iterrows():
            # Get the input text
            input_text = row['Student_Comment']

            # Tokenize the input
            inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)

            # Move inputs to the device
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Forward pass to get model output (loss and logits_sarcasm)
            _, logits_sarcasm = model(**inputs)

            # Get predictions (1 for sarcasm, 0 for non-sarcasm)
            predictions = (torch.sigmoid(logits_sarcasm) > 0.5).int().cpu().numpy()
            pred_sarcasm_labels.extend(predictions)

    # Loop through each row in your DataFrame to print the labels
    for index, row in df.iterrows():
        if index < len(pred_sarcasm_labels):
            
            pred_sarcasm_label = 'Sarcasm' if pred_sarcasm_labels[index] == 1 else 'Non-Sarcasm'

            # Print the sentence with the predicted label
            print(f"Row No: {row['Row No.']}")
            print(f"Sentence: {row['Student_Comment']}")
            print(f"Predicted Sarcasm Label: {pred_sarcasm_label}")
            print('-' * 80)  # separator line for readability
        else:
            print(f"Prediction not available for index {index}.")

    # Save the updated DataFrame to a CSV file
    #output_file = "classified_output_STL.csv"
    #df.to_csv(output_file, index=False)
    #print(f"Classified output saved to {output_file}")


    # Result Page for Sarcasm and Mock Politeness

class TreeviewTooltip:
    def __init__(self, tree):
        self.tree = tree
        self.tooltip = None
        self.tree.bind("<Motion>", self.on_hover)
        self.hover_id = None  # ID for the delayed tooltip

    def on_hover(self, event):
        # Get the item under the mouse
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)

        if item and column:
            item_values = self.tree.item(item, "values")
            col_index = int(column[1:]) - 1  # Convert from "#1", "#2" to index 0, 1
            if 0 <= col_index < len(item_values):
                text = item_values[col_index]

                # Schedule the tooltip with a 1-second delay
                if self.hover_id:
                    self.tree.after_cancel(self.hover_id)  # Reset delay if moving
                self.hover_id = self.tree.after(300, lambda: self.show_tooltip(event, text))
        else:
            self.hide_tooltip()  # Hide if not hovering over a valid cell

    def show_tooltip(self, event, text):
        if self.tooltip:
            self.tooltip.destroy()  # Remove previous tooltip
        self.tooltip = tk.Toplevel(self.tree)
        self.tooltip.wm_overrideredirect(True)  # Remove window decorations
        self.tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")  # Position near cursor

        # Define maximum width before wrapping (adjust as needed)
        max_width = 300

        # Styling 
        label = tk.Label(
            self.tooltip,
            text=text,
            background="white",
            foreground="black",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=4,
            font=("Inter", 14, "bold"),
            wraplength=max_width,  # Enables text wrapping
            justify="left",  # Align text for readability
        )
        label.pack(ipadx=6, ipady=3)
        self.tooltip.configure(bg="gray")  # Simulated drop shadow
        self.tree.bind("<Leave>", self.hide_tooltip)  # Hide on mouse leave

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None
        if self.hover_id:
            self.tree.after_cancel(self.hover_id)
            self.hover_id = None

# Function to load and display the CSV content in the Treeview
def load_csv_Sarcasm():
    global tree  # Make tree global for access in tooltip
    # Create the Treeview widget
    tree = ttk.Treeview(indigo_frame_Result_Sarcasm, columns=("Value", "Label"), show='headings', height=10)

    # Define the columns
    tree.heading("Value", text="Value")
    tree.heading("Label", text="Label")

    # Set the column widths
    tree.column("Value", width=400, anchor='center')  # Set width for the Value column
    tree.column("Label", width=100, anchor='center')   # Set width for the Label column

    # Custom styling
    style = ttk.Style()
    style.configure("Treeview",
                    background="#b4c5e4",       # Background color for the cells
                    foreground="black",     # Text color for the cells
                    rowheight=50,             # Row height (increased for visibility)
                    fieldbackground="#b4c5e4",  # Field background color
                    borderwidth=2,
                    font=('Inter', 14))            # Width of the cell border

    style.configure("Treeview.Heading",
                    padding=[11, 11, 11, 11],
                    background="white",  # Header background color
                    foreground="black",     # Header text color
                    font=('Inter', 18, 'bold', 'underline'))  # Header font style

    '''    #style.map("Treeview",
              background=[('selected', '#d1e8ff')],  # Color when row is selected
              foreground=[('selected', 'black')])  # Text color when row is selected'''

    # Counters for sarcasm and non-sarcasm
    total_comments = 0
    sarcasm_count = 0
    non_sarcasm_count = 0

    # Create a Scrollbar
    scrollbar = ttk.Scrollbar(indigo_frame_Result_Sarcasm, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    # Apply padding (margins) to the Treeview frame
    indigo_frame_Result_Sarcasm.grid_columnconfigure(0, weight=1, pad=20)
    indigo_frame_Result_Sarcasm.grid_rowconfigure(0, weight=1, pad=20)

    # Place the Treeview and Scrollbar using grid for centering and alignment
    tree.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    scrollbar.grid(row=0, column=1, sticky="ns")

    for index, row in df.iterrows():
        if index < len(pred_sarcasm_labels):
            pred_sarcasm_label = (
                'Sarcasm' if pred_sarcasm_labels[index] == 1 
                else 'Non-Sarcasm'
            )

            # Insert the data row by row
            tree.insert("", "end", values=(row['Student_Comment'], pred_sarcasm_label))
            total_comments += 1
            if pred_sarcasm_label == 'Sarcasm':
                sarcasm_count += 1
            else:
                non_sarcasm_count += 1
    
    canvas_Sarcasm_Result.itemconfig(total_comments_text_sarcasm, text="Total Comments = " + str(total_comments))
    canvas_Sarcasm_Result.itemconfig(sarcasm_count_text_sarcasm, text="Sarcasm Detection = " + str(sarcasm_count))
    canvas_Sarcasm_Result.itemconfig(non_sarcasm_count_text_sarcasm, text="Non-Sarcasm Detection = " + str(non_sarcasm_count))
    
    TreeviewTooltip(tree)  # Attach tooltip functionality

def followfunction_sarcasm():
    # Show the loading frame first
    loadingframe_sarcasm = LoadingFrame()  # Create the loading frame and keep a reference to it
    
    # Use threading to run the next steps without blocking the UI
    threading.Thread(target=Sarcasm_save_file_path_and_show_page, args=(loadingframe_sarcasm,)).start()

def Sarcasm_save_file_path_and_show_page(loadingframe_sarcasm):
    try:
        file_path_sarcasm = file_entry_sarcasm.get().strip()  # Get the file path and remove extra spaces

        if not file_path_sarcasm:  # Check if the file path is empty
            raise ValueError("File path cannot be empty.")

        # Now that the path is saved, call the sarcasm model
        sarcasm_model(file_path_sarcasm)
        
        # Load the CSV for Sarcasm and Non-Sarcasm
        load_csv_Sarcasm()
        
        # Destroy the loading frame once the process is complete
        loadingframe_sarcasm.destroy()  
        
        # Call the function to show the Sarcasm_Frame page
        show_page(Sarcasm_Frame)

    except ValueError as e:
        loadingframe_sarcasm.destroy()  # Ensure the loading frame is removed if there's an error
        messagebox.showerror("Error", str(e))  # Show an error message to the user


# Bind the button click to save the file path and show the page
canvas_sarcasm.tag_bind(classify_bg_sarcasm, "<Button-1>", lambda e: followfunction_sarcasm())
canvas_sarcasm.tag_bind(classify_text_sarcasm, "<Button-1>", lambda e: followfunction_sarcasm())
# Bind the button click to save the file path and show the page
canvas_Sarcasm_Result.tag_bind(cm_bg_sarcasm, "<Button-1>", lambda e: stl_cm())
canvas_Sarcasm_Result.tag_bind(cm_text_sarcasm, "<Button-1>", lambda e: stl_cm())

####################################################################################################################################

# Result Page for Sarcasm and Mock Politeness

Sarcasm_and_MP_Frame = tk.Frame(root, bg="white")

# Indigo rectangle (left) with proper padding for margins
indigo_frame_Result_MP = tk.Frame(Sarcasm_and_MP_Frame, bg="#090c9b")
indigo_frame_Result_MP.place(x=20, y=20, width=860, height=960)  # Adding margin to each side

# White rectangle (right)
canvas_MP_Result= Canvas(Sarcasm_and_MP_Frame, bg="white", highlightthickness=0)
canvas_MP_Result.place(x=900, y=150, width=510, height=800)

# Confusion matrix button 
cm_bg_mock = create_rounded_rectangle(canvas_MP_Result,90, 400, 410, 500, radius=20, fill="#3d52d5", outline="#3d52d5")
cm_text_mock = canvas_MP_Result.create_text(250, 448, text="Confusion Matrix", font=("Inter", 25, "bold"), fill="white")
bind_button_states(canvas_MP_Result, cm_bg_mock, cm_text_mock)

# Bind the button click to save the file path and show the page
canvas_MP_Result.tag_bind(cm_bg_mock, "<Button-1>", lambda e: mtl_cm())
canvas_MP_Result.tag_bind(cm_text_mock, "<Button-1>", lambda e: mtl_cm())

# Title Label
canvas_MP_Result.create_text(250, 160, text="Results", font=("Inter", 30, "bold"), fill="#000000", justify='center')

# Placeholder text that will be updated dynamically
total_comments_text_mp = canvas_MP_Result.create_text(250, 220, text="Total Comments = 0", font=("Inter", 20), fill="#000000", justify='center')
sarcasm_count_text_mp = canvas_MP_Result.create_text(250, 260, text="Sarcasm Detection = 0", font=("Inter", 20), fill="#000000", justify='center')
mock_politeness_count_text_mp = canvas_MP_Result.create_text(250, 300, text="Mock Politeness Detection = 0", font=("Inter", 20), fill="#000000", justify='center')
non_sarcasm_count_text_mp = canvas_MP_Result.create_text(250, 340, text="Non-Sarcasm Detection = 0", font=("Inter", 20), fill="#000000", justify='center')

# Create the Back to Home button 
back_home_bg = create_rounded_rectangle(canvas_MP_Result, 90, 520, 410, 620, radius=20,
                                         fill="#3d52d5", outline="#3d52d5")
back_home_text = canvas_MP_Result.create_text(250, 570, text="Back to Home", font=("Inter", 25, "bold"), fill="white")

# Bind button states for hover effects and interactions
bind_button_states(canvas_MP_Result, back_home_bg, back_home_text)

# Bind the Back to Home button to show the home page
canvas_MP_Result.tag_bind(back_home_bg, "<Button-1>", lambda e: show_page(home_frame))
canvas_MP_Result.tag_bind(back_home_text, "<Button-1>", lambda e: show_page(home_frame))

# Display the result frame
Sarcasm_Frame.pack(fill=tk.BOTH, expand=True)

def MockPoliteness_Model(file):
    # Load your dataset

    dataset = pd.read_csv(file)
    global df, pred_mock_politeness_labels, pred_mock_politeness_label
    df = dataset
    # Apply the conversion function to the 'Value' column in your dataset
    df['Student_Comment'] = df['Student_Comment'].apply(convert_emoji_and_emoticon)
    
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer and model from the pre-trained model
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    

    # Define your custom model class
    class CustomXLMRobertaModel(nn.Module):
        def __init__(self, dropout_prob=0.3):
            super(CustomXLMRobertaModel, self).__init__()
            self.xlm_roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")

            # Dropout layer
            self.dropout = nn.Dropout(dropout_prob)

            # Hidden layer size (adjust as needed)
            hidden_layer_size = 128  # Adjust the size of the hidden layer as you see fit

            # Sarcasm classification head with an extra hidden layer
            self.classifier_sarcasm = nn.Sequential(
                nn.Linear(self.xlm_roberta.config.hidden_size, hidden_layer_size),  # First hidden layer
                nn.ReLU(),  # Activation function
                nn.Linear(hidden_layer_size, 1)  # Output layer for binary classification
            )

            # Mock Politeness classification head with an extra hidden layer
            self.classifier_mockpoliteness = nn.Sequential(
                nn.Linear(self.xlm_roberta.config.hidden_size, hidden_layer_size),  # First hidden layer
                nn.ReLU(),  # Activation function
                nn.Linear(hidden_layer_size, 3)  # Output layer for 3-class classification
            )

            # Loss functions
            self.criterion_sarcasm = nn.BCEWithLogitsLoss()  # For binary classification
            self.criterion_mockpoliteness = nn.CrossEntropyLoss()  # For multiclass classification

        def forward(self, input_ids, attention_mask, labels_sarcasm=None, labels_mockpoliteness=None):
            # XLM-RoBERTa forward pass
            outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)

            # Get the CLS token output (first token's hidden state)
            cls_token_output = outputs[0][:, 0, :]  # Shape: [batch_size, hidden_size]

            # Apply dropout to CLS token output
            cls_token_output = self.dropout(cls_token_output)

            # Sarcasm classification head
            logits_sarcasm = self.classifier_sarcasm(cls_token_output).squeeze(-1)  # Shape: [batch_size]

            # Mock Politeness classification head
            logits_mockpoliteness = self.classifier_mockpoliteness(cls_token_output)  # Shape: [batch_size, 3]

            loss = None
            if labels_sarcasm is not None and labels_mockpoliteness is not None:
                # Compute the loss for both tasks
                loss_sarcasm = self.criterion_sarcasm(logits_sarcasm, labels_sarcasm.float())  # Binary cross-entropy loss
                loss_mockpoliteness = self.criterion_mockpoliteness(logits_mockpoliteness, labels_mockpoliteness)  # Cross-entropy loss

                # Weighted combination of losses
                loss  = (loss_sarcasm + loss_mockpoliteness) / 2

            return loss, logits_sarcasm, logits_mockpoliteness

    # Initialize the model
    model = CustomXLMRobertaModel(dropout_prob=0.3)

    
    # Load your trained model state
    try:
        model.load_state_dict(torch.load(r"best_models\best_model_mtl_7.pt", map_location=device))
    except RuntimeError as e:
        print(f"Error loading model state: {e}")
        # Handle the error or exit

    # Move model to the defined device
    model.to(device)
    model.eval()
    global pred_mock_politeness_labels, true_mock_politeness_labels
    # Initialize lists to hold true and predicted mock politeness labels
    pred_mock_politeness_labels = []
    true_mock_politeness_labels = df['MP_Label'].tolist()  # True labels for mock politeness

    with torch.no_grad():  # Disable gradient calculation
        for index, row in df.iterrows():
            # Get the input text
            input_text = row['Student_Comment']
            
            # Tokenize the input
            inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
            
            # Move inputs to the device
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Forward pass to get model output (loss, logits_sarcasm, logits_mockpoliteness)
            _, logits_sarcasm, logits_mockpoliteness = model(**inputs)

            # Get predictions by taking the argmax of the mock politeness logits
            predictions = torch.argmax(logits_mockpoliteness, dim=1).cpu().numpy()
            pred_mock_politeness_labels.extend(predictions)
    # Loop through each row in your DataFrame to print the labels
    for index, row in df.iterrows():
        # Check if the index exists in predictions
        if index < len(pred_mock_politeness_labels):            
            # Predicted Mock Politeness Label
            pred_mock_politeness_label = (
                'Mock Politeness' if pred_mock_politeness_labels[index] == 2 
                else 'Sarcasm' if pred_mock_politeness_labels[index] == 1 
                else 'Non-Sarcasm'
            )

            # Print the sentence with the true and predicted labels
            print(f"Row No: {row['Row No.']}")
            print(f"Sentence: {row['Student_Comment']}")
            print("Predicted Mock Politeness Label: " + pred_mock_politeness_label)
            print('-' * 80)  # separator line for readability
        else:
            print(f"Prediction not available for index {index}.")

    # Add predicted labels to the DataFrame
    df['True_Label'] = [
        'Mock Politeness' if label == 2 else 'Sarcasm' if label == 1 else 'Non-Sarcasm'
        for label in true_mock_politeness_labels
    ]
    df['Predicted_Label'] = [
        'Mock Politeness' if label == 2 else 'Sarcasm' if label == 1 else 'Non-Sarcasm'
        for label in pred_mock_politeness_labels
    ]
    # Save the updated DataFrame to a CSV file
    #output_file = "classified_output_MTL.csv"
    #df.to_csv(output_file, index=False)
    #print(f"Classified output saved to {output_file}")


    # Result Page for Sarcasm and Mock Politeness
    
# Function to load and display the CSV content in the Treeview
def load_csv_Sarcasm_and_MP():
    # Create the Treeview widget
    tree = ttk.Treeview(indigo_frame_Result_MP, columns=("Value", "Label"), show='headings', height=10)

    # Define the columns
    tree.heading("Value", text="Value")
    tree.heading("Label", text="Label")

    # Set the column widths
    tree.column("Value", width=460, anchor='center')  # Set width for the Value column
    tree.column("Label", width=40, anchor='center')   # Set width for the Label column

    # Custom styling
    style = ttk.Style()
    style.configure("Treeview",
                    background="#b4c5e4",       # Background color for the cells
                    foreground="black",     # Text color for the cells
                    rowheight=50,             # Row height (increased for visibility)
                    fieldbackground="#b4c5e4",  # Field background color
                    borderwidth=2,
                    font=('Inter', 14))            # Width of the cell border

    style.configure("Treeview.Heading",
                    padding=[11, 11, 11, 11],
                    background="white",  # Header background color
                    foreground="black",     # Header text color
                    font=('Inter', 18, 'bold', 'underline'))  # Header font style
    
    '''    #style.map("Treeview",
                background=[('selected', '#d1e8ff')],  # Color when row is selected
                foreground=[('selected', 'black')])  # Text color when row is selected'''

    # Counters for sarcasm and non-sarcasm
    total_comments = 0
    sarcasm_count = 0
    mock_politeness_count = 0
    non_sarcasm = 0


    # Read the CSV and insert the rows into the Treeview
   

    # Create a Scrollbar
    scrollbar = ttk.Scrollbar(indigo_frame_Result_MP, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    # Apply padding (margins) to the Treeview frame
    indigo_frame_Result_MP.grid_columnconfigure(0, weight=1, pad=20)
    indigo_frame_Result_MP.grid_rowconfigure(0, weight=1, pad=20)

    # Place the Treeview and Scrollbar using grid for centering and alignment
    tree.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    scrollbar.grid(row=0, column=1, sticky="ns")
    for index, row in df.iterrows():
        if index < len(pred_mock_politeness_labels):
            
            pred_mock_politeness_label = (
                'Mock Politeness' if pred_mock_politeness_labels[index] == 2 
                else 'Sarcasm' if pred_mock_politeness_labels[index] == 1 
                else 'Non-Sarcasm'
            )

            # Insert the data row by row
            tree.insert("", "end", values=(row['Student_Comment'], pred_mock_politeness_label))
            total_comments += 1
            if pred_mock_politeness_labels[index] == 2:
                mock_politeness_count += 1
            elif pred_mock_politeness_labels[index] == 1:
                sarcasm_count += 1
            else:
                non_sarcasm += 1    
        

    # Update the canvas_MP_Result text with the counts
    canvas_MP_Result.itemconfig(total_comments_text_mp, text="Total Comments = " + str(total_comments))
    canvas_MP_Result.itemconfig(sarcasm_count_text_mp, text="Sarcasm Detection = " + str(sarcasm_count))
    canvas_MP_Result.itemconfig(mock_politeness_count_text_mp, text="Mock Politeness Detection = " + str(mock_politeness_count))
    canvas_MP_Result.itemconfig(non_sarcasm_count_text_mp, text="Non-Sarcasm Detection = " + str(non_sarcasm))
    
    TreeviewTooltip(tree)  # Attach tooltip functionality


def followfunction_mock():
    # Show the loading frame first
    loadingframe_mock = LoadingFrame()  # Create the loading frame and keep a reference to it
    
    # Use threading to run the next steps without blocking the UI
    threading.Thread(target=MP_save_file_path_and_show_page, args=(loadingframe_mock,)).start()

    

def MP_save_file_path_and_show_page(loadingframe_mock):
    try:
        file_path_mock = file_entry_mock.get()  # Get the file path entered by the user

        # Show the loading window
        if not file_path_mock:
            raise ValueError("File path cannot be empty.")

        # Process the file with the selected path
        MockPoliteness_Model(file_path_mock)

        # Load CSV with Sarcasm and Mock Politeness data
        load_csv_Sarcasm_and_MP()

        loading_frame1.destroy()


        # Show the new page or frame
        show_page(Sarcasm_and_MP_Frame)
    except ValueError as e:
        loadingframe_mock.destroy()  # Ensure the loading frame is removed if there's an error
        messagebox.showerror("Error", str(e))  # Show an error message to the user

    

# Bind the button click to save the file path and show the page
canvas_mock.tag_bind(classify_bg_mock, "<Button-1>", lambda e: followfunction_mock())
canvas_mock.tag_bind(classify_text_mock, "<Button-1>", lambda e: followfunction_mock())

#######################################################################################################################################################


# Confusion matrix and metrics for sarcasm classification
def stl_cm():
    
    
    # Check if true labels are provided and are not NaN
    if not true_sarcasm_labels or np.any(pd.isna(true_sarcasm_labels)):
        messagebox.showerror("Error", "A confusion matrix can't be generated due to lack of true labels.")
        return  # Stop further execution if labels are missing or invalid

    # STL Frame (Sarcasm Detection)
    global stl_frame
    stl_frame = tk.Frame(root, bg="white", width=1440, height=1000)

    # Indigo rectangle (left) for STL Frame
    stl_left_frame = tk.Frame(stl_frame, bg="#090c9b", width=720, height=1000)  
    stl_left_frame.pack(side=tk.LEFT, fill=tk.Y)  
    stl_canvas_2 = Canvas(stl_left_frame, bg="#090c9b", highlightthickness=0)
    stl_canvas_2.place(x=0, y=0, width=720, height=1000)
    stl_left_label = tk.Label(stl_left_frame, text="Sarcasm Detection\nConfusion Matrix",
                            font=("Inter", 28, "bold"), fg="white", bg="#090c9b", justify="center")
    stl_left_label.place(x=180, y=90)
      

    # White rectangle (right) for STL Frame
    stl_right_frame = tk.Frame(stl_frame, bg="white", width=720, height=1000)
    stl_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    stl_canvas_1 = Canvas(stl_right_frame, bg="white", highlightthickness=0)
    stl_canvas_1.place(x=0, y=0, width=720, height=1000)
    stl_right_label = tk.Label(stl_right_frame, text="Performance Metrics", font=("Inter", 28, "bold"), fg="black", bg="white")
    stl_right_label.place(x=170, y=125) 
    
    

     # Create confusion matrix and classification report
    cm = confusion_matrix(true_sarcasm_labels, pred_sarcasm_labels)
    report = classification_report(true_sarcasm_labels, pred_sarcasm_labels, output_dict=True)

    # Plot heatmap for confusion matrix on the left side
    fig, ax = plt.subplots(figsize=(6, 6))  # Create a figure and axis for the heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                annot_kws={"size": 16},  # Increase annotation font size
                xticklabels=['Non-Sarcasm', 'Sarcasm'], 
                yticklabels=['Non-Sarcasm', 'Sarcasm'], ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=18)
    ax.set_ylabel('True Label', fontsize=18)

    # Set larger font for tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Embed Matplotlib figure in Tkinter's canvas
    canvas = FigureCanvasTkAgg(fig, master=stl_left_frame)
    canvas.draw()
    canvas.get_tk_widget().place(x=60, y=200)

    # Create a Treeview for the classification report
    tree_frame = tk.Frame(stl_right_frame)
    tree_frame.place(x=50, y=200, width=620, height=600)

    # Create Treeview without style
    tree2 = ttk.Treeview(tree_frame, columns=("Label", "Precision", "Recall", "F1 Score", "Support"), show='headings')
    
    # Define column headings
    tree2.heading("Label", text="Label")
    tree2.heading("Precision", text="Precision")
    tree2.heading("Recall", text="Recall")
    tree2.heading("F1 Score", text="F1 Score")
    tree2.heading("Support", text="Support")

    # Set column widths for better visibility
    tree2.column("Label", width=100, anchor='center')            # Label column width
    tree2.column("Precision", width=130, anchor='center')       # Precision column width
    tree2.column("Recall", width=130, anchor='center')          # Recall column width
    tree2.column("F1 Score", width=130, anchor='center')        # F1 Score column width
    tree2.column("Support", width=100, anchor='center')         # Support column width

    label_map = {
    '0': 'Non-Sarcasm',
    '1': 'Sarcasm',
    
    }
    
    # Insert data into the Treeview, converting scores to percentages
    for label, metrics in report.items():
        if label in label_map:  # Only include the individual classes
            precision = "{:.2f}%".format(metrics['precision'] * 100)
            recall = "{:.2f}%".format(metrics['recall'] * 100)
            f1_score = "{:.2f}%".format(metrics['f1-score'] * 100)
            support = metrics['support']
            tree2.insert("", "end", values=(label_map[label], precision, recall, f1_score, support))

    # Add macro and weighted averages
    macro_precision = "{:.2f}%".format(report['macro avg']['precision'] * 100)
    macro_recall = "{:.2f}%".format(report['macro avg']['recall'] * 100)
    macro_f1 = "{:.2f}%".format(report['macro avg']['f1-score'] * 100)
    macro_support = report['macro avg']['support']
    
    weighted_precision = "{:.2f}%".format(report['weighted avg']['precision'] * 100)
    weighted_recall = "{:.2f}%".format(report['weighted avg']['recall'] * 100)
    weighted_f1 = "{:.2f}%".format(report['weighted avg']['f1-score'] * 100)
    weighted_support = report['weighted avg']['support']

    tree2.insert("", "end", values=("Macro avg", macro_precision, macro_recall, macro_f1, macro_support))
    tree2.insert("", "end", values=("Weighted avg", weighted_precision, weighted_recall, weighted_f1, weighted_support))

    # Add Scrollbars
    scrollbar_y = ttk.Scrollbar(tree_frame, orient="vertical", command=tree2.yview)
    scrollbar_y.pack(side='right', fill='y')
    tree2.configure(yscrollcommand=scrollbar_y.set)

    scrollbar_x = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree2.xview)
    scrollbar_x.pack(side='bottom', fill='x')
    tree2.configure(xscrollcommand=scrollbar_x.set)

    tree2.pack(fill=tk.BOTH, expand=True)

    
    mtlbutton_bg = create_rounded_rectangle(stl_canvas_1, 230, 840, 490, 920, radius=20, fill="#3d52d5", outline="#3d52d5")
    mtlbutton_text = stl_canvas_1.create_text(360, 880, text="Back to Results", font=("Inter", 20, "bold"), fill="white")
    # Bind the hover effects for the button (for MTL transition)
    bind_button_states(stl_canvas_1, mtlbutton_bg, mtlbutton_text)
    
    
    # Bind the mtlbutton to switch from STL to MTL frame
    stl_canvas_1.tag_bind(mtlbutton_bg, "<Button-1>", lambda e: show_page(Sarcasm_Frame))
    stl_canvas_1.tag_bind(mtlbutton_text, "<Button-1>", lambda e: show_page(Sarcasm_Frame))

    bbmtlbutton_bg = create_rounded_rectangle(stl_canvas_2, 230, 840, 490, 920, radius=20, fill="#3d52d5", outline="#3d52d5")
    bbmtlbutton_text = stl_canvas_2.create_text(360, 880, text="Back to Home", font=("Inter", 20, "bold"), fill="white")
    # Bind the hover effects for the button (for MTL transition)
    bind_button_states(stl_canvas_2, bbmtlbutton_bg, bbmtlbutton_text)
    
    
    # Bind the mtlbutton to switch from STL to MTL frame
    stl_canvas_2.tag_bind(bbmtlbutton_bg, "<Button-1>", lambda e: show_page(home_frame))
    stl_canvas_2.tag_bind(bbmtlbutton_text, "<Button-1>", lambda e: show_page(home_frame))

    show_page(stl_frame)


def mtl_cm():
    
    # Check if true labels are provided and are not NaN
    if not true_mock_politeness_labels or np.any(pd.isna(true_mock_politeness_labels)):
        messagebox.showerror("Error", "A Confusion Matrix can't be generated due to lack of True Labels.")
        return  # Stop further execution if labels are missing or invalid
    
    global mtl_frame
    
    # MTL Frame (Mock Politeness Detection)
    mtl_frame = Canvas(root, bg="white", width=1440, height=1000)
    

    # Indigo rectangle (left) for MTL Frame
    mtl_left_frame = tk.Frame(mtl_frame, bg="#090c9b", width=720, height=1000)
    mtl_left_frame.pack(side=tk.LEFT, fill=tk.Y)
    # White rectangle (right) for MTL Frame
    mtl_right_frame = Canvas(mtl_frame, bg="white", width=720, height=1000)
    mtl_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
     # Button to go back to STL 
    mtl_canvas_1 = Canvas(mtl_right_frame, bg="white", highlightthickness=0)
    mtl_canvas_1.place(x=0, y=0, width=720, height=1000)
    mtl_canvas_2 = Canvas(mtl_left_frame, bg="#090c9b", highlightthickness=0)
    mtl_canvas_2.place(x=0, y=0, width=720, height=1000)
    mtl_left_label = tk.Label(mtl_left_frame, text="Sarcasm and Mock\nPoliteness Detection\nConfusion Matrix",
                            font=("Inter", 28, "bold"), fg="white", bg="#090c9b", justify="center")
    mtl_left_label.place(x=180, y=40)
    mtl_right_label = tk.Label(mtl_right_frame, text="Performance Metrics", font=("Inter", 28, "bold"), fg="black", bg="white")
    mtl_right_label.place(x=170, y=125)
        
    # Create confusion matrix and classification report
    cm = confusion_matrix(true_mock_politeness_labels, pred_mock_politeness_labels)
    report = classification_report(true_mock_politeness_labels, pred_mock_politeness_labels, output_dict=True)

    # Plot heatmap for confusion matrix on the left side
    fig, ax = plt.subplots(figsize=(6, 6))  # Create a figure and axis for the heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                annot_kws={"size": 16},  # Increase annotation font size
                xticklabels=['Non-Sarcasm', 'Sarcasm', 'Mock Politeness'], 
                yticklabels=['Non-Sarcasm', 'Sarcasm', 'Mock Politeness'], ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=18)
    ax.set_ylabel('True Label', fontsize=18)

    # Set larger font for tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Embed Matplotlib figure in Tkinter's canvas
    canvas = FigureCanvasTkAgg(fig, master=mtl_left_frame)
    canvas.draw()
    canvas.get_tk_widget().place(x=60, y=200)

    # Create a Treeview for the classification report
    tree_frame = tk.Frame(mtl_right_frame)
    tree_frame.place(x=50, y=200, width=620, height=600)

    # Create Treeview without style
    tree2 = ttk.Treeview(tree_frame, columns=("Label", "Precision", "Recall", "F1 Score", "Support"), show='headings')
    
    # Define column headings
    tree2.heading("Label", text="Label")
    tree2.heading("Precision", text="Precision")
    tree2.heading("Recall", text="Recall")
    tree2.heading("F1 Score", text="F1 Score")
    tree2.heading("Support", text="Support")

    # Set column widths for better visibility
    tree2.column("Label", width=100, anchor='center')            # Label column width
    tree2.column("Precision", width=130, anchor='center')       # Precision column width
    tree2.column("Recall", width=130, anchor='center')          # Recall column width
    tree2.column("F1 Score", width=130, anchor='center')        # F1 Score column width
    tree2.column("Support", width=100, anchor='center')         # Support column width

    label_map = {
    '0': 'Non-Sarcasm',
    '1': 'Sarcasm',
    '2': 'Mock Politeness'
    }
    
    # Insert data into the Treeview, converting scores to percentages
    for label, metrics in report.items():
        if label in label_map:  # Only include the individual classes
            precision = "{:.2f}%".format(metrics['precision'] * 100)
            recall = "{:.2f}%".format(metrics['recall'] * 100)
            f1_score = "{:.2f}%".format(metrics['f1-score'] * 100)
            support = metrics['support']
            tree2.insert("", "end", values=(label_map[label], precision, recall, f1_score, support))

    # Add macro and weighted averages
    macro_precision = "{:.2f}%".format(report['macro avg']['precision'] * 100)
    macro_recall = "{:.2f}%".format(report['macro avg']['recall'] * 100)
    macro_f1 = "{:.2f}%".format(report['macro avg']['f1-score'] * 100)
    macro_support = report['macro avg']['support']
    
    weighted_precision = "{:.2f}%".format(report['weighted avg']['precision'] * 100)
    weighted_recall = "{:.2f}%".format(report['weighted avg']['recall'] * 100)
    weighted_f1 = "{:.2f}%".format(report['weighted avg']['f1-score'] * 100)
    weighted_support = report['weighted avg']['support']

    tree2.insert("", "end", values=("Macro avg", macro_precision, macro_recall, macro_f1, macro_support))
    tree2.insert("", "end", values=("Weighted avg", weighted_precision, weighted_recall, weighted_f1, weighted_support))

    # Add Scrollbars
    scrollbar_y = ttk.Scrollbar(tree_frame, orient="vertical", command=tree2.yview)
    scrollbar_y.pack(side='right', fill='y')
    tree2.configure(yscrollcommand=scrollbar_y.set)

    scrollbar_x = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree2.xview)
    scrollbar_x.pack(side='bottom', fill='x')
    tree2.configure(xscrollcommand=scrollbar_x.set)

    
    tree2.pack(fill=tk.BOTH, expand=True)
    # Create rounded rectangles and text on the Canvas
    
    stlbutton_bg_mtl = create_rounded_rectangle(mtl_canvas_1, 230, 840, 490, 920, radius=20, fill="#3d52d5", outline="#3d52d5")
    stlbutton_text_mtl = mtl_canvas_1.create_text(360, 880, text="Back To Results", font=("Inter", 20, "bold"), fill="white")
    # Bind the hover effects for the button (to return to STL)
    bind_button_states(mtl_canvas_1, stlbutton_bg_mtl, stlbutton_text_mtl)
    # Bind the stlbutton in MTL frame to switch back to STL frame
    mtl_canvas_1.tag_bind(stlbutton_bg_mtl, "<Button-1>", lambda e: show_page(Sarcasm_and_MP_Frame))
    mtl_canvas_1.tag_bind(stlbutton_text_mtl, "<Button-1>", lambda e: show_page(Sarcasm_and_MP_Frame))

    bbutton_bg_mtl = create_rounded_rectangle(mtl_canvas_2, 230, 840, 490, 920, radius=20, fill="#3d52d5", outline="#3d52d5")
    bbutton_text_mtl = mtl_canvas_2.create_text(360, 880, text="Back To Home", font=("Inter", 20, "bold"), fill="white")
    # Bind the hover effects for the button (to return to STL)
    bind_button_states(mtl_canvas_2, bbutton_bg_mtl, bbutton_text_mtl)
    # Bind the stlbutton in MTL frame to switch back to STL frame
    mtl_canvas_2.tag_bind(bbutton_bg_mtl, "<Button-1>", lambda e: show_page(home_frame))
    mtl_canvas_2.tag_bind(bbutton_text_mtl, "<Button-1>", lambda e: show_page(home_frame))
    
    
    show_page(mtl_frame)






show_page(home_frame)
root.protocol("WM_DELETE_WINDOW", sys.exit)
root.mainloop()
