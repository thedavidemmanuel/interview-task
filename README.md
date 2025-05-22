# English Accent Classifier for Hiring

> **AI-powered English proficiency assessment tool for modern hiring teams**

An intelligent solution that evaluates candidates' English communication skills through accent analysis from video recordings. Perfect for remote hiring, global teams, and assessing English proficiency for customer-facing roles.


## 🚀 Quick Start

### 📱 **Try the Live Demo**
**Deployed App**: [🔗 **Launch Accent Classifier**](https://your-app-url.streamlit.app/)

### 🎬 **How to Test (3 Simple Steps)**
1. **Get a video URL**: 
   - Loom recording 
   - Use any YouTube video with clear English speech
   - Upload video to Google Drive and get shareable link

2. **Analyze**: 
   - Paste URL into the tool
   - Click "🚀 Analyze English Accent"
   - Wait 30-60 seconds for processing

3. **Get Results**:
   - View accent classification with confidence score
   - Read hiring recommendation
   - Download assessment for candidate file

### 🎯 **Sample Test URLs**
```
✅ Good for testing:
- Loom introduction videos
- Professional YouTube presentations  
- Clear interview recordings
- Educational content with single speakers

❌ Avoid:
- Music videos or songs
- Multiple speakers talking over each other
- Heavy background noise or music
- Videos shorter than 10 seconds
```

## 🛠️ Local Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/accent-classifier.git
cd accent-classifier
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the app**:
```bash
streamlit run app.py
```

4. **Open in browser**: Navigate to `http://localhost:8501`

### Required Files
- `app.py` - Main Streamlit application
- `english_indian_accent_classifier.h5` - Trained CNN model
- `accent_mapping.pkl` - Label mappings
- `requirements.txt` - Python dependencies

## 📁 Project Structure

```
accent-classifier/
├── app.py                                    # Main application
├── english_indian_accent_classifier.h5       # CNN model (4MB)
├── accent_mapping.pkl                        # Label mappings
├── requirements.txt                          # Dependencies
└── README.md                                 # This file
```

## 🎥 Supported Video Sources

- **Loom**: loom.com recordings
- **YouTube**: youtube.com videos  
- **Vimeo**: vimeo.com videos
- **Direct links**: .mp4, .mov, .avi files
- **Most platforms**: Supported by yt-dlp

## 🧠 Technical Details

### Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Features**: 13 MFCC coefficients over 499 time frames
- **Training**: AccentDB dataset with 5 accent categories

### Accent Categories
1. **American English** (Native)
2. **British English** (Native)
3. **Australian English** (Native)
4. **Welsh English** (Native)
5. **Indian English** (Non-native but fluent)

### Processing Pipeline
1. **Audio Extraction**: yt-dlp downloads audio from video
2. **Feature Extraction**: Librosa extracts MFCC features
3. **Classification**: CNN model predicts accent category
4. **Confidence Scoring**: Softmax probability as confidence

## 📊 Understanding Results

### Confidence Levels
- **90-100%**: Very reliable classification
- **80-89%**: High confidence, good for hiring decisions
- **70-79%**: Moderate confidence, generally reliable
- **60-69%**: Lower confidence, consider additional samples
- **<60%**: Low confidence, may need clearer audio

