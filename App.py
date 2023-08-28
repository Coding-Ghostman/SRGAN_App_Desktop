import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


model = Generator()
model.load_state_dict(torch.load(
    './Models/generator_model_epoch_35.pth', map_location=torch.device('cpu')))
model.eval()


class ImageSuperResolutionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Super Resolution App")

        self.root.geometry("900x400")  # Set an initial window size

        self.upload_button = ttk.Button(
            root, text="Upload Image", command=self.upload_image, style="Upload.TButton")
        # Add padding between button and images
        self.upload_button.pack(pady=10)

        self.style = ttk.Style()
        self.style.configure("Upload.TButton", font=(
            "Helvetica", 12), background="#4CAF50", foreground="black")

        self.image_frame = tk.Frame(root)
        self.image_frame.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        input_image = Image.open(file_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(input_image).unsqueeze(0)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_image = output_tensor.squeeze().clamp(0, 1).cpu().numpy()
        output_image = (output_image * 255).astype('uint8')
        output_image = Image.fromarray(output_image.transpose(1, 2, 0), 'RGB')
        self.display_images(input_image, output_image)

    def display_images(self, input_image, output_image):
        self.image_frame.destroy()  # Clear previous images

        self.image_frame = tk.Frame(root)
        self.image_frame.pack()

        transform_LR = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        transform_HR = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        input_lr = Image.fromarray((((transform_LR(input_image).clamp(
            0, 1).cpu().numpy()) * 255).astype('uint8')).transpose(1, 2, 0), 'RGB')

        input_hr = Image.fromarray((((transform_HR(input_image).clamp(
            0, 1).cpu().numpy()) * 255).astype('uint8')).transpose(1, 2, 0), 'RGB')

        input_LR_photo = ImageTk.PhotoImage(
            input_lr.resize((256, 256), Image.BILINEAR))
        output_photo = ImageTk.PhotoImage(output_image)
        input_HR_photo = ImageTk.PhotoImage(input_hr)

        input_label_lr = tk.Label(self.image_frame, image=input_LR_photo)
        input_label_lr.image = input_LR_photo
        # Add padding between images
        input_label_lr.grid(row=0, column=0, padx=10)

        input_label_hr = tk.Label(self.image_frame, image=input_HR_photo)
        input_label_hr.image = input_HR_photo
        # Add padding between images
        input_label_hr.grid(row=0, column=1, padx=10)

        output_label = tk.Label(self.image_frame, image=output_photo)
        output_label.image = output_photo
        output_label.grid(row=0, column=2, padx=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSuperResolutionApp(root)
    root.mainloop()
