AI-Powered Medical Diagnosis Using CNN for
Precision Healthcare
Source Code
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow\.keras.models import load\_model
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from gtts import gTTS
from playsound import playsound
from googletrans import Translator
import webbrowser
import uuid
import threading
import tkinter.font as tkFont

# Enable GPU memory growth

gpu\_devices = tf.config.list\_physical\_devices('GPU')
if gpu\_devices:
for device in gpu\_devices:
tf.config.experimental.set\_memory\_growth(device, True)

# Model configuration

MODELS = {
"Pneumonia": {"path": "models/pneumonia\_model.h5",
"input\_size": (224, 224), "grayscale": False},
"MURA": {"path": "models/mura\_model.h5", "input\_size":
(224, 224), "grayscale": False},
"Tuberculosis": {"path": "models/tb\_model.h5", "input\_size":
(224, 224), "grayscale": False}
}

# Load models

LOADED\_MODELS = {
disease: {"model": load\_model(info\["path"]), "input\_size":
info\["input\_size"], "grayscale": info\["grayscale"]}
for disease, info in MODELS.items()
}

# Disease Information with working Google Maps links

DISEASE\_INFO = {
"Pneumonia": {
"description": "Pneumonia is an infection that inflames air
sacs in one or both lungs, caused by bacteria, viruses, or fungi.",
"consultation": "Consult a pulmonologist.",
"curing\_method": "Hydration, rest, and antibiotics.",
"hospitals": \[
{"name": "Apollo Hospitals, Chennai", "location": "21,
Greams Lane, Greams Rd, Chennai",
"contact": "+91 44 2829 3333",
"link":
"[https://www.google.com/maps/place/Apollo+Hospitals,+Gream](https://www.google.com/maps/place/Apollo+Hospitals,+Gream)
s+Rd,+Chennai"},
{"name": "Fortis Malar Hospital", "location": "52, 1st
Main Rd, Gandhi Nagar, Adyar, Chennai",
"contact": "+91 44 4289 2222", "link":
"[https://www.google.com/maps/place/Fortis+Malar+Hospital](https://www.google.com/maps/place/Fortis+Malar+Hospital)"},
{"name": "Global Hospital", "location": "439, Cheran
Nagar, Perumbakkam, Chennai",
"contact": "+91 44 4477 7000", "link":
"[https://www.google.com/maps/place/Global+Hospitals,+Perum](https://www.google.com/maps/place/Global+Hospitals,+Perum)
bakkam"}
]
},
"MURA": {
"description": "MURA refers to abnormalities in
musculoskeletal radiographs such as fractures, dislocations, or
deformities.",
"consultation": "Consult an orthopedic specialist.",
"curing\_method": "Physical therapy, pain management.",
"hospitals": \[
{"name": "MIOT International", "location": "4/112,
Mount Poonamallee Rd, Manapakkam, Chennai",
"contact": "+91 44 4200 2288", "link":
"[https://www.google.com/maps/place/MIOT+Hospitals](https://www.google.com/maps/place/MIOT+Hospitals)"},
{"name": "Orthomed Hospital", "location": "21,
Royapettah High Rd, Royapettah, Chennai",
"contact": "+91 44 4061 4061", "link":
"[https://www.google.com/maps/place/Orthomed+Hospital](https://www.google.com/maps/place/Orthomed+Hospital)"},
{"name": "VS Hospitals",
"location": "1, New No. 6, Old No. 174, Valluvarkottam
High Rd, Nungambakkam, Chennai",
"contact": "+91 44 4909 4909", "link":
"[https://www.google.com/maps/place/VS+Hospitals](https://www.google.com/maps/place/VS+Hospitals)"}
]
},
"Tuberculosis": {
"description": "Tuberculosis is a bacterial infection that
mainly affects the lungs and spreads through airborne particles.",
"consultation": "Consult a pulmonologist or infectious
disease specialist.",
"curing\_method": "Long-term antibiotic treatment.",
"hospitals": \[
{"name": "Government TB Hospital", "location":
"Tambaram Sanatorium, Chennai",
"contact": "+91 44 2536 1000", "link":
"[https://www.google.com/maps/place/Government+TB+Hospital](https://www.google.com/maps/place/Government+TB+Hospital)
"},
{"name": "Institute of Thoracic Medicine", "location":
"Chetpet, Chennai",
"contact": "+91 44 2836 1044", "link":
"[https://www.google.com/maps/place/Institute+of+Thoracic+Me](https://www.google.com/maps/place/Institute+of+Thoracic+Me)
dicine"},
{"name": "Stanley Medical College Hospital", "location":
"Old Jail Rd, Old Washermanpet, Chennai",
"contact": "+91 44 2528 2044", "link":
"[https://www.google.com/maps/place/Stanley+Medical+College](https://www.google.com/maps/place/Stanley+Medical+College)
"}
]
}
}
languages = {
"English": "en", "Hindi": "hi", "Tamil": "ta", "Malayalam":
"ml",
"Telugu": "te", "Kannada": "kn", "Chinese": "zh-cn",
"Korean": "ko"
}
translator = Translator()

# Custom colors with medical theme

BG\_COLOR = "#f8f9fa"
HEADER\_COLOR = "#005b96"
TEXT\_COLOR = "#343a40"
LINK\_COLOR = "#0066cc"
BUTTON\_COLOR = "#0077b6"
SPEAKER\_COLOR = "#1e88e5"
DIVIDER\_COLOR = "#dee2e6"
IMAGE\_BG\_COLOR = "#ffffff"
SECTION\_HEADER\_COLOR = "#d32f2f"
def preprocess\_image(image\_path, input\_size, grayscale=False):
img = cv2.imread(image\_path)
img = cv2.cvtColor(img, cv2.COLOR\_BGR2RGB)
if grayscale:
img = cv2.cvtColor(img, cv2.COLOR\_RGB2GRAY)
img = np.expand\_dims(img, axis=-1)
img = cv2.resize(img, input\_size)
img = img / 255.0
return np.expand\_dims(img, axis=0)
def predict\_disease(image\_path):
predictions = {}
for disease, model\_info in LOADED\_MODELS.items():
try:
img = preprocess\_image(image\_path,
model\_info\["input\_size"], model\_info\["grayscale"])
pred = model\_info\["model"].predict(img)\[0]\[0]
predictions\[disease] = float(pred)
except Exception:
predictions\[disease] = -1
confident = {k: v for k, v in predictions.items() if v >= 0.85}
if confident:
best = max(confident, key=confident.get)
return {best: True, "Confidence": round(confident\[best] \*
100, 2)}
return {"No Disease Detected": False, "Confidence":
round(max(predictions.values()) \* 100, 2)}
def open\_directions(url):
webbrowser.open(url)
def format\_results(results):
report = ""
for disease, detected in results.items():
if disease != "Confidence" and detected:
info = DISEASE\_INFO\[disease]
hospital\_list = "\n".join(\[
f"{i + 1}. {h\['name']}, {h\['location']}\n
{h\['contact']}\n   Directions: {h\['link']}"
for i, h in enumerate(info\["hospitals"])
])
report = f"""DIAGNOSIS REPORT\n\n{disease}
Detected (Confidence: {results\['Confidence']}%)\n\n\[ABOUT
DISEASE]\n{info\['description']}\n\n\[CONSULTATION]\n{info\[
'consultation']}\n\n\[CURING
METHOD]\n{info\['curing\_method']}\n\n\[NEARBY
HOSPITALS]\n{hospital\_list}"""
break
if not report:
report = f"DIAGNOSIS REPORT\n\nNo Disease Detected
(Confidence: {results\['Confidence']}%)"
return report
def text\_to\_speech(text, language="en"):
def play():
try:
filename = f"tts\_{uuid.uuid4().hex}.mp3"
tts = gTTS(text=text, lang=language, slow=False)
tts.save(filename)
playsound(filename)
os.remove(filename)
except Exception as e:
print("TTS Error:", e)
t = threading.Thread(target=play)
t.daemon = True
t.start()
def speak\_report():
lang\_code = languages\[selected\_language.get()]
text = diagnosis\_box.get("1.0", tk.END)
if text.strip():
speak\_button.config(text="Speaking...", bg="#64b5f6",
state="disabled")
text\_to\_speech(text, language=lang\_code)
root.after(100, lambda: speak\_button.config(text="
Speak Report", bg=SPEAKER\_COLOR, state="normal"))
def upload\_image():
global original\_report, current\_image
file\_path = filedialog.askopenfilename(
filetypes=\[("Image Files", ".png;.jpg;\*.jpeg")],
title="Select Medical Scan Image"
)
if file\_path:
image\_canvas.delete("all")
image\_canvas.create\_text(100, 100, text="Processing...",
fill="gray", font=label\_font)
root.update()
try:
img = Image.open(file\_path)
width, height = img.size
aspect = width / height
new\_width = 200
new\_height = int(new\_width / aspect)
if new\_height > 200:
new\_height = 200
new\_width = int(new\_height \* aspect)
img = img.resize((new\_width, new\_height),
Image.Resampling.LANCZOS)
bg = Image.new('RGB', (200, 200), (255, 255, 255))
x = (200 - new\_width) // 2
y = (200 - new\_height) // 2
bg.paste(img, (x, y))
current\_image = ImageTk.PhotoImage(bg)
image\_canvas.delete("all")
image\_canvas.create\_image(100, 100,
image=current\_image)
results = predict\_disease(file\_path)
original\_report = format\_results(results)
translate\_and\_display()
speak\_button.config(state="normal")
except Exception as e:
image\_canvas.delete("all")
image\_canvas.create\_text(100, 100, text="Error loading
image", fill="red", font=label\_font)
diagnosis\_box.config(state="normal")
diagnosis\_box.delete("1.0", tk.END)
diagnosis\_box.insert(tk.END, f"Error: {str(e)}")
diagnosis\_box.config(state="disabled")
speak\_button.config(state="disabled")
def translate\_and\_display():
lang\_code = languages\[selected\_language.get()]
try:
translated = translator.translate(original\_report,
dest=lang\_code).text
except:
translated = original\_report
diagnosis\_box.config(state="normal")
diagnosis\_box.delete("1.0", tk.END)

# Configure text styles

diagnosis\_box.tag\_config("title",
foreground=HEADER\_COLOR, font=("Helvetica", 14, "bold"))
diagnosis\_box.tag\_config("header",
foreground=SECTION\_HEADER\_COLOR, font=("Helvetica",
12, "bold"))
diagnosis\_box.tag\_config("normal",
foreground=TEXT\_COLOR, font=("Helvetica", 11))
diagnosis\_box.tag\_config("link", foreground=LINK\_COLOR,
font=("Helvetica", 11, "underline"))
lines = translated.split('\n')
for line in lines:
if not line.strip():
diagnosis\_box.insert(tk.END, "\n")
continue
if line == "DIAGNOSIS REPORT":
diagnosis\_box.insert(tk.END, line + "\n\n", "title")
elif line.startswith("\["):
diagnosis\_box.insert(tk.END, line + "\n", "header")
elif line.strip().startswith("Directions:"):
parts = line.split("Directions:")
diagnosis\_box.insert(tk.END, parts\[0], "normal")
if len(parts) > 1:
url = parts\[1].strip()
start\_index = diagnosis\_box.index(tk.INSERT)
diagnosis\_box.insert(tk.END, "Directions", "link")
end\_index = diagnosis\_box.index(tk.INSERT)
diagnosis\_box.tag\_add("link", start\_index, end\_index)
diagnosis\_box.tag\_bind("link", "<Button-1>", lambda
e, url=url: open\_directions(url))
diagnosis\_box.insert(tk.END, "\n")
elif line.strip() and line\[0].isdigit():
diagnosis\_box.insert(tk.END, line + "\n", "normal")
else:
diagnosis\_box.insert(tk.END, line + "\n", "normal")
diagnosis\_box.config(state="disabled")
def update\_ui\_language(\_=None):
lang\_code = languages\[selected\_language.get()]
try:
title\_label.config(text=translator.translate("AI-Powered
Medical Diagnostics", dest=lang\_code).text)
upload\_button.config(text=translator.translate("Upload
Image", dest=lang\_code).text)
language\_label.config(text=translator.translate("Select
Language:", dest=lang\_code).text)
translate\_and\_display()
except:
pass

# GUI Setup

root = tk.Tk()
root.title("AI-Powered Medical Diagnostics")
root.geometry("700x750")
root.configure(bg=BG\_COLOR)
original\_report = ""
selected\_language = tk.StringVar(value="English")
current\_image = None

# Fonts

title\_font = tkFont.Font(family="Helvetica", size=16,
weight="bold")
label\_font = tkFont.Font(family="Helvetica", size=10)
button\_font = tkFont.Font(family="Helvetica", size=10,
weight="bold")
report\_font = tkFont.Font(family="Helvetica", size=10)

# Header

header\_frame = tk.Frame(root, bg=HEADER\_COLOR,
padx=15, pady=10)
header\_frame.pack(fill="x")

title\_label = tk.Label(header\_frame,
text="AI-Powered Medical Diagnostics",
font=title\_font,
fg="white",
bg=HEADER\_COLOR)
title\_label.pack(side="left", padx=5)

language\_frame = tk.Frame(header\_frame,
bg=HEADER\_COLOR)
language\_frame.pack(side="right", padx=5)

language\_label = tk.Label(language\_frame,
text="Select Language:",
font=label\_font,
fg="white",
bg=HEADER\_COLOR)
language\_label.pack(side="left")

language\_spinner = ttk.Combobox(language\_frame,
values=list(languages.keys()),
textvariable=selected\_language,
state="readonly",
font=label\_font,
width=10)
language\_spinner.pack(side="left", padx=5)
language\_spinner.bind("<<ComboboxSelected>>",
update\_ui\_language)

# Main content

main\_frame = tk.Frame(root, bg=BG\_COLOR, padx=20,
pady=10)
main\_frame.pack(fill="both", expand=True)

# Upload section

upload\_frame = tk.Frame(main\_frame, bg=BG\_COLOR)
upload\_frame.pack(fill="x", pady=5)

button\_frame = tk.Frame(upload\_frame, bg=BG\_COLOR)
button\_frame.pack()

upload\_button = tk.Button(button\_frame,
text="Upload Image",
command=upload\_image,
font=button\_font,
bg=BUTTON\_COLOR,
fg="white",
activebackground="#0066a1",
activeforeground="white",
relief="flat",
padx=20,
pady=5)
upload\_button.pack(side="left", padx=5)

speak\_button = tk.Button(button\_frame,
text="    Speak Report",
command=speak\_report,
font=button\_font,
bg=SPEAKER\_COLOR,
fg="white",
activebackground="#64b5f6",
activeforeground="white",
relief="flat",
padx=20,
pady=5,
state="disabled")
speak\_button.pack(side="left")

# Divider

divider = tk.Frame(main\_frame, height=1,
bg=DIVIDER\_COLOR)
divider.pack(fill="x", pady=5)

# Image display

image\_frame = tk.Frame(main\_frame, bg=BG\_COLOR)
image\_frame.pack(pady=5)

image\_canvas = tk.Canvas(image\_frame, width=200,
height=200, bg=IMAGE\_BG\_COLOR, highlightthickness=1,
highlightbackground="#ced4da")
image\_canvas.pack()
image\_canvas.create\_text(100, 100, text="No image selected",
fill="gray", font=label\_font)

# Report section

report\_frame = tk.Frame(main\_frame, bg=BG\_COLOR)
report\_frame.pack(fill="both", expand=True)
report\_container = tk.Frame(report\_frame, bg="white",
relief="solid", bd=1)
report\_container.pack(fill="both", expand=True, pady=(0, 5))
diagnosis\_box = tk.Text(report\_container,
font=report\_font,
wrap="word",
padx=10,
pady=10,
bg="white",
borderwidth=0,
highlightthickness=0,
height=12)
diagnosis\_box.pack(side="left", fill="both", expand=True)
root.mainloop()
