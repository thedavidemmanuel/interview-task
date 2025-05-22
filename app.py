import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import pickle
import tempfile
import os
import subprocess
import time

# Page config
st.set_page_config(page_title="Accent Classifier", page_icon="🎤")

st.title("🎤 English Accent Classifier")
st.markdown("### AI-powered accent analysis from video URLs")

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('english_indian_accent_classifier.h5')
        with open('accent_mapping.pkl', 'rb') as f:
            accent_mapping = pickle.load(f)
        return model, accent_mapping
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, accent_mapping = load_model()

if model is None:
    st.error("❌ Model not found. Please ensure these files are in your directory:")
    st.code("english_indian_accent_classifier.h5\naccent_mapping.pkl")
    st.stop()

st.success("✅ Model loaded successfully!")

def download_and_extract_audio(url):
    """Download audio and convert to WAV format"""
    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Get FFmpeg path
        ffmpeg_path = os.path.join('ffmpeg', 'ffmpeg-7.1.1-essentials_build', 'bin', 'ffmpeg.exe')
        ffprobe_path = os.path.join('ffmpeg', 'ffmpeg-7.1.1-essentials_build', 'bin', 'ffprobe.exe')
        
        # Simple yt-dlp command - download best audio
        cmd = [
            "yt-dlp",
            "--format", "bestaudio",
            "--output", os.path.join(temp_dir, "audio.%(ext)s"),
            url
        ]
        
        # Add FFmpeg location if it exists
        if os.path.exists(ffmpeg_path):
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            cmd.extend(["--ffmpeg-location", ffmpeg_dir])
        
        st.info("Downloading audio...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            st.error(f"Download failed: {result.stderr}")
            return None
        
        # Find downloaded file
        files = os.listdir(temp_dir)
        audio_files = [f for f in files if f.startswith('audio.')]
        
        if not audio_files:
            st.error("No audio file downloaded")
            return None
            
        original_audio_path = os.path.join(temp_dir, audio_files[0])
        st.success(f"Downloaded: {audio_files[0]}")
        
        # Convert to WAV for better librosa compatibility
        wav_path = os.path.join(temp_dir, "converted_audio.wav")
        
        # Use FFmpeg to convert to WAV format
        if os.path.exists(ffmpeg_path):
            st.info("Converting audio to WAV format...")
            convert_cmd = [
                ffmpeg_path,
                "-i", original_audio_path,
                "-ar", "22050",  # Match sample rate used in feature extraction
                "-ac", "1",      # Convert to mono
                "-y",            # Overwrite output file if it exists
                wav_path
            ]
            
            conv_result = subprocess.run(convert_cmd, capture_output=True, text=True)
            
            if conv_result.returncode == 0 and os.path.exists(wav_path):
                st.success("Audio converted successfully")
                return wav_path
            else:
                st.warning(f"Conversion failed, using original file: {conv_result.stderr}")
                return original_audio_path
        else:
            st.warning("FFmpeg not found for conversion, using original file")
            return original_audio_path
        
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def extract_features(audio_path):
    """Extract MFCC features"""
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            st.error(f"Audio file not found: {audio_path}")
            return None
            
        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            st.error("Audio file is empty (0 bytes)")
            return None
        elif file_size < 1000:  # Less than 1KB
            st.warning(f"Audio file is very small ({file_size} bytes), may not contain audio")
        
        st.info(f"Loading audio file: {os.path.basename(audio_path)} ({file_size/1024:.1f} KB)")
        
        # Load audio
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=5.0)
            if len(y) == 0:
                st.error("Audio loaded but contains no samples")
                return None
            st.info(f"Loaded {len(y)} audio samples at {sr} Hz")
        except Exception as e:
            st.error(f"Failed to load audio: {str(e)}")
            return None
        
        # Extract MFCC
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
            mfcc = mfcc.T  # Transpose to (time, features)
            st.info(f"Extracted MFCC features: shape {mfcc.shape}")
        except Exception as e:
            st.error(f"Failed to extract MFCC features: {str(e)}")
            return None
        
        # Pad/truncate to 499 frames
        if mfcc.shape[0] < 499:
            mfcc = np.pad(mfcc, ((0, 499 - mfcc.shape[0]), (0, 0)), mode='constant')
            st.info(f"Padded features to shape {mfcc.shape}")
        else:
            mfcc = mfcc[:499, :]
            st.info(f"Truncated features to shape {mfcc.shape}")
        
        # Normalize
        norm = np.linalg.norm(mfcc.flatten())
        if norm > 0:
            mfcc = mfcc / norm
            
        return mfcc
        
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def predict_accent(features):
    """Predict accent"""
    try:
        # Reshape for model
        features_input = features.reshape(1, 499, 13)
        
        # Predict
        predictions = model.predict(features_input, verbose=0)
        confidence = np.max(predictions) * 100
        predicted_class = np.argmax(predictions)
        accent = accent_mapping[predicted_class]
        
        # Check if native
        native_accents = ["australian", "british", "american", "welsh"]
        is_native = accent in native_accents
        
        return {
            'accent': accent,
            'confidence': confidence,
            'is_native': is_native,
            'all_probs': {accent_mapping[i]: predictions[0][i]*100 for i in range(len(accent_mapping))}
        }
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# Main interface
st.markdown("### Enter Video URL")
url = st.text_input("Video URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("🚀 Analyze Accent", type="primary"):
    if not url:
        st.warning("Please enter a URL")
    else:
        with st.spinner("Processing..."):
            # Step 1: Download
            audio_path = download_and_extract_audio(url)
            if not audio_path:
                st.stop()
            
            # Step 2: Extract features
            st.info("Extracting features...")
            features = extract_features(audio_path)
            if features is None:
                st.stop()
            
            # Step 3: Predict
            st.info("Classifying accent...")
            result = predict_accent(features)
            if not result:
                st.stop()
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Results")
        
        # Main result
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accent", result['accent'].title())
        
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1f}%")
        
        with col3:
            status = "Native" if result['is_native'] else "Non-Native"
            st.metric("Speaker Type", status)
        
        # Detailed breakdown
        st.markdown("### Probability Breakdown")
        for accent, prob in sorted(result['all_probs'].items(), key=lambda x: x[1], reverse=True):
            if accent == result['accent']:
                st.success(f"**{accent.title()}**: {prob:.1f}% ⭐")
            else:
                st.info(f"{accent.title()}: {prob:.1f}%")
        
        # Hiring recommendation
        st.markdown("### 💼 Hiring Assessment")
        if result['is_native']:
            st.success(f"""
            **✅ Native English Speaker**
            - {result['accent'].title()} accent detected
            - Confidence: {result['confidence']:.1f}%
            - Excellent for English communication roles
            """)
        else:
            confidence_level = "High" if result['confidence'] > 80 else "Medium" if result['confidence'] > 60 else "Low"
            st.info(f"""
            **ℹ️ Non-Native English Speaker**
            - {result['accent'].title()} accent detected
            - Confidence: {result['confidence']:.1f}% ({confidence_level})
            - Consider role requirements for verbal communication
            """)

# Instructions
with st.expander("📝 How to Use"):
    st.markdown("""
    1. **Paste a video URL** (YouTube, Loom, etc.)
    2. **Click Analyze Accent**
    3. **Get instant results** with confidence scores
    
    **Supported accents:**
    - 🇺🇸 American (Native)
    - 🇬🇧 British (Native)  
    - 🇦🇺 Australian (Native)
    - 🏴󠁧󠁢󠁷󠁬󠁳󠁿 Welsh (Native)
    - 🇮🇳 Indian English (Non-native)
    """)

st.markdown("---")
st.markdown("**Model Accuracy: 98.5%** | Built for hiring assessment")