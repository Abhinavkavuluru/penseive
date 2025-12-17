#!/bin/bash

echo "=========================================="
echo "Pensieve Project Configuration"
echo "=========================================="

# Get the project directory
PROJECT_DIR=$(pwd)
echo "Project directory: $PROJECT_DIR"

# Create required directories
echo "Step 1: Creating required directories..."
mkdir -p cooked_traces
mkdir -p rl_server/results
mkdir -p run_exp/results
mkdir -p real_exp/results
mkdir -p sim/cooked_traces
mkdir -p sim/cooked_test_traces
mkdir -p sim/results
mkdir -p sim/test_results
mkdir -p test/cooked_test_traces
mkdir -p test/results
mkdir -p test/models

echo "Directories created successfully!"

# Install Node.js dependencies for dash.js
echo "Step 2: Installing dash.js dependencies..."
if [ -d "dash.js" ]; then
    cd dash.js
    npm install
    cd ..
    echo "dash.js dependencies installed!"
else
    echo "Warning: dash.js directory not found"
fi

# Install Node.js dependencies for dash_client
echo "Step 3: Installing dash_client dependencies..."
if [ -d "dash_client/player_code_noMPC" ]; then
    cd dash_client/player_code_noMPC
    npm install
    cd ../..
    echo "dash_client dependencies installed!"
else
    echo "Warning: dash_client directory not found"
fi

# Copy video server files to Apache
echo "Step 4: Setting up Apache server..."
if [ -d "video_server" ]; then
    sudo cp video_server/myindex_*.html /var/www/html/ 2>/dev/null || echo "No myindex files found"
    sudo cp video_server/dash.all.min.js /var/www/html/ 2>/dev/null || echo "No dash.all.min.js found"
    sudo cp -r video_server/video* /var/www/html/ 2>/dev/null || echo "No video files found"
    sudo cp video_server/Manifest.mpd /var/www/html/ 2>/dev/null || echo "No Manifest.mpd found"
    echo "Apache server configured!"
else
    echo "Warning: video_server directory not found"
fi

# Start Apache
echo "Step 5: Starting Apache server..."
sudo service apache2 start
echo "Apache server started!"

# Create sample trace file if none exist
echo "Step 6: Checking for trace files..."
if [ ! "$(ls -A cooked_traces 2>/dev/null)" ]; then
    echo "No trace files found. You need to add network trace files to:"
    echo "  - cooked_traces/"
    echo "  - sim/cooked_traces/"
    echo "  - sim/cooked_test_traces/"
    echo "  - test/cooked_test_traces/"
    echo ""
    echo "You can generate synthetic traces using: python3 sim/synthetic_traces.py"
fi

echo "=========================================="
echo "Project Configuration Complete!"
echo "=========================================="
echo ""
echo "Project structure:"
tree -L 2 -d . 2>/dev/null || ls -la
echo ""
echo "Next steps:"
echo "1. Add network trace files"
echo "2. Generate video sizes: cd sim && python3 get_video_sizes.py"
echo "3. Start training: python3 sim/multi_agent.py"
echo "4. Or test: cd test && python3 rl_no_training.py"
