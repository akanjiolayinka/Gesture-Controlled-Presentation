# 🎮 Gesture-Controlled Presentation System

Control PowerPoint, Keynote, or Google Slides presentations using hand gestures detected via webcam - **WORKING AND TESTED!**

## ✅ Status: FULLY FUNCTIONAL

The system is ready to use! All dependencies are installed and the hand tracking model is configured.

## 🚀 Quick Start

### Method 1: Double-Click (Easiest)
- **Windows**: Double-click `run.bat`
- **PowerShell**: Right-click `run.ps1` → Run with PowerShell

### Method 2: Command Line
```bash
# Activate virtual environment first
.venv\Scripts\activate

# Run the application
python gesture_presentation.py
```

## 🖐️ Supported Gestures

| Gesture | Action | How To |
|---------|--------|--------|
| Swipe Right → | Next Slide | Move hand quickly to the right |
| Swipe Left ← | Previous Slide | Move hand quickly to the left |
| Swipe Down ↓ | Start Presentation (F5) | Move hand down |
| Swipe Up ↑ | Stop Presentation (ESC) | Move hand up |
| Pinch 🤏 | Black Screen | Touch thumb and index finger |
| Open Palm 🖐 | White Screen | Show open palm with all fingers |

## ⌨️ Keyboard Shortcuts

While the application is running:

- `q` - Quit application
- `s` - Toggle debug display on/off
- `l` - Toggle hand landmarks visualization
- `r` - Reset gesture tracking
- `1` - Manual next slide (for testing)
- `2` - Manual previous slide (for testing)

## 🎯 Usage Tips

### Best Practices
1. **Good Lighting**: Ensure your room is well-lit
2. **Hand Position**: Keep your hand in the camera frame
3. **Deliberate Gestures**: Make clear, intentional movements
4. **Distance**: Stay 1-2 feet from the camera
5. **Focus**: Have your presentation app in focus before gesturing

### For PowerPoint Users
```bash
python gesture_presentation.py --app powerpoint
```

### Adjust Sensitivity
```bash
# Less sensitive (fewer false triggers)
python gesture_presentation.py --sensitivity 0.7

# More sensitive (quicker response)
python gesture_presentation.py --sensitivity 1.5
```

### Use Different Camera
```bash
python gesture_presentation.py --camera 1
```

## 📦 What's Installed

All dependencies are pre-installed in `.venv`:
- ✅ OpenCV 4.12.0 - Camera and image processing
- ✅ MediaPipe 0.10.31 - Hand tracking AI
- ✅ PyAutoGUI 0.9.54 - Keyboard control
- ✅ NumPy 2.2.6 - Numerical operations

## 🔧 System Requirements

- **Python**: 3.12+ (Already configured in `.venv`)
- **Webcam**: Built-in or external
- **OS**: Windows (tested), macOS, Linux
- **RAM**: 2GB minimum, 4GB recommended
- **Presentation Software**: PowerPoint, Keynote, Google Slides, or browser

## 📁 Project Structure

```
Gesture Controller/
├── .venv/                      # Virtual environment (DO NOT DELETE)
├── models/                     # Hand tracking model
│   └── hand_landmarker.task   # MediaPipe model (auto-downloaded)
├── gesture_presentation.py    # Main application
├── test_camera.py             # Camera testing utility
├── run.bat                    # Windows launcher
├── run.ps1                    # PowerShell launcher
├── requirements.txt           # Python dependencies
├── install_requirements.bat   # Dependency installer
└── README.md                  # This file
```

## 🐛 Troubleshooting

### Camera Not Detected
```bash
python test_camera.py
```
This will show available cameras and test hand detection.

### Low Frame Rate
- Close other camera applications
- Reduce window size
- Update camera drivers

### Gestures Not Working
1. Ensure presentation app is in focus
2. Try adjusting sensitivity: `--sensitivity 0.8`
3. Check lighting conditions
4. Press `r` to reset tracking

### Hand Not Detected
- Move closer to camera
- Improve lighting
- Try different hand positions
- Check debug overlay (press `s`)

## 🎓 Advanced Options

### Run Without Debug Overlay
```bash
python gesture_presentation.py --no-debug
```

### Custom Application Mode

## Performance Tips

- **Close unnecessary applications** to free up CPU
- **Use adequate lighting** for better hand detection
- **Keep hand movements deliberate** to avoid false triggers
- **Adjust sensitivity** based on your environment
- **Use lower camera resolution** if experiencing lag

## Project Structure

```
Gesture Controller/
├── gesture_presentation.py    # Main application
├── test_camera.py             # Camera testing utility
├── requirements.txt           # Python dependencies
├── install_requirements.bat   # Windows installer
├── install_requirements.sh    # Mac/Linux installer
└── README.md                  # This file
```

## How It Works

1. **Camera Capture** - Captures video frames from webcam
2. **Hand Detection** - MediaPipe detects hand landmarks (21 points)
3. **Gesture Recognition** - Analyzes finger positions and hand movement
4. **Smoothing** - Buffers gestures to prevent false triggers
5. **Action Mapping** - Maps gestures to keyboard commands
6. **Presentation Control** - Sends commands to presentation software

## Future Enhancements

- [ ] Laser pointer simulation with point gesture
- [ ] Zoom controls with two-finger gestures
- [ ] Voice feedback for gesture confirmation
- [ ] Machine learning for personalized gesture learning
- [ ] Multi-hand support for advanced controls
- [ ] Web-based configuration interface
- [ ] Gesture recording and playback
- [ ] Support for remote presentations

## License

This project is provided as-is for educational and personal use.

## Credits

- **MediaPipe** - Google's hand tracking solution
- **OpenCV** - Computer vision library
- **PyAutoGUI** - GUI automation

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Test your camera with `test_camera.py`
3. Verify all dependencies are installed
4. Check that presentation software is running

---

**Happy Presenting! 🎤✋**
