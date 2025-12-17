#!/bin/bash

echo "=========================================="
echo "Pensieve Project Setup for WSL/Ubuntu"
echo "=========================================="

# Update system
echo "Step 1: Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python and pip
echo "Step 2: Installing Python and pip..."
sudo apt-get install -y python3 python3-pip python3-dev

# Install system dependencies
echo "Step 3: Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    unzip \
    xvfb \
    xserver-xephyr \
    tightvncserver

# Install Mahimahi (network emulator)
echo "Step 4: Installing Mahimahi..."
sudo sysctl -w net.ipv4.ip_forward=1
sudo add-apt-repository -y ppa:keithw/mahimahi
sudo apt-get update -y
sudo apt-get install -y mahimahi

# Install Apache server
echo "Step 5: Installing Apache2..."
sudo apt-get install -y apache2

# Install Python packages
echo "Step 6: Installing Python packages..."
pip3 install --upgrade pip

# Install TensorFlow 2.x (compatible with Python 3)
pip3 install tensorflow==2.13.0

# Install other Python dependencies
pip3 install tflearn
pip3 install numpy
pip3 install matplotlib
pip3 install scipy
pip3 install h5py
pip3 install selenium
pip3 install pyvirtualdisplay

# Install Node.js and npm
echo "Step 7: Installing Node.js..."
sudo apt-get install -y nodejs npm

# Install Chrome (for Selenium)
echo "Step 8: Installing Google Chrome..."
wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
sudo apt-get -f install -y
rm google-chrome-stable_current_amd64.deb

# Install ChromeDriver
echo "Step 9: Installing ChromeDriver..."
CHROME_DRIVER_VERSION=$(curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE)
wget -q https://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip
unzip -q chromedriver_linux64.zip
sudo mv chromedriver /usr/local/bin/
sudo chmod +x /usr/local/bin/chromedriver
rm chromedriver_linux64.zip

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Installed versions:"
python3 --version
pip3 --version
node --version
npm --version
google-chrome --version
echo ""
echo "Next steps:"
echo "1. Run: bash setup_project.sh"
echo "2. Add network trace files to cooked_traces/"
echo "3. Start training or testing"
