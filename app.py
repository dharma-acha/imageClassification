import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 256 * 4 * 4)
        x = self.fc_layers(x)
        return x

model = VGG13()
model.load_state_dict(torch.load("C:/Users/accha/OneDrive/Desktop/2ndSem/DL/project/VGG13.pth", map_location=torch.device('cpu')))
model.eval()


classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def transform_image(image_file):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_file).convert('RGB')
    return transform(image).unsqueeze(0)






st.set_page_config(page_title="Interactive VGG Image Classifier", page_icon=":camera:")

st.markdown(
    "<h1 style='text-align: center; color: black;'>Interactive VGG Image Classifier</h1>", 
    unsafe_allow_html=True
)

uploads = st.file_uploader("Choose images:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploads:
    st.write("### Uploaded Images:")
    cols = st.columns(4)  
    predictions = []  
    for i, uploaded_file in enumerate(uploads):
        if i % 4 == 0:
            st.write(" ")
        with cols[i % 4]:  
            image = transform_image(uploaded_file)  
            st.image(uploaded_file, caption='Uploaded Image', width=150)

            if st.button('Predict', key=f'predict_button_{i}'): 
                with st.spinner("Predicting..."):
                    output = model(image)
                    probs = F.softmax(output, dim=1)
                    pred_class = torch.argmax(probs, dim=1)
                    class_name = classes[pred_class.item()]
                    predictions.append(class_name)

    if predictions:
        st.write("### Predictions:")
        for j, prediction in enumerate(predictions, 1):
            st.write(f"Image: **{prediction}**")

    if st.button("Clear All"):
        uploads.clear()  
        st.empty()  
        predictions = []  






