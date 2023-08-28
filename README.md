# Image Super-Resolution Application

This is a simple Tkinter application that allows you to upload an image, process it using a super-resolution PyTorch model, and display the results.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)
- [License](#license)

## Overview

This application is built using Python's Tkinter library for creating the user interface and PyTorch for implementing the super-resolution functionality. The uploaded image is first transformed to 64x64 pixels, then passed through the super-resolution model to generate a 256x256 high-resolution output image.

## Getting Started

### Prerequisites

To run this application, you'll need the following:

- Python (3.6 or higher)
- PyTorch (and torchvision)
- PIL (Python Imaging Library)
- tkinter (included with most Python installations)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/image-super-resolution-app.git
   cd image-super-resolution-app
   pip install torch torchvision pillow
   ```
### Usage

1. Run the application:
  ```bash
  python app.py
  ```
3. Click on the "Upload Image" button to select an image file from your local machine.
4. The uploaded image will be transformed to 64x64 pixels and then processed using the super-resolution model.
5. The original 64x64 image and the 256x256 high-resolution output image will be displayed on the application window.

### Customization

- To use your own super-resolution PyTorch model, replace 'model.pth' with the actual path to your model checkpoint in the app.py file.
- You can customize the architecture of the super-resolution model by editing the SuperResolutionModel class in the app.py file.
- Feel free to modify the user interface and enhance the application's features according to your requirements.
