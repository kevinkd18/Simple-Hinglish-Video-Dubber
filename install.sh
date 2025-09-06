#!/bin/bash
# One-command installation script for Hinglish Video Dubber

echo "🚀 Starting Hinglish Video Dubber Setup..."

# Detect OS and install system dependencies
echo "📦 Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    # Debian/Ubuntu
    sudo apt update && sudo apt install -y ffmpeg python3 python3-pip python3-venv
elif command -v brew &> /dev/null; then
    # macOS
    brew install ffmpeg python
elif command -v choco &> /dev/null; then
    # Windows
    choco install ffmpeg python -y
else
    echo "❌ Unsupported package manager. Please install ffmpeg manually."
    exit 1
fi

# Create and activate virtual environment
echo "🔧 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate || . venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Installation complete!"
echo "🎬 To start the application, run:"
echo "source venv/bin/activate  # If not already activated"
echo "python ok.py"
