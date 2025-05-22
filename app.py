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

st.title("🎤 English Accent Classifier for Hiring")
st.markdown("### AI-powered English accent analysis for candidate evaluation")

# Load model
@st.cache_resource
def load_model():
    try:
        # Custom loader for models with batch_shape parameter
        def load_model_with_custom_objects(model_path):
            try:
                # Try to load the model normally
                model = tf.keras.models.load_model(model_path)
                return model
            except Exception as e:
                if "batch_shape" in str(e):
                    # Define a custom layer that handles the batch_shape parameter
                    class CustomInputLayer(tf.keras.layers.InputLayer):
                        def __init__(self, batch_shape=None, **kwargs):
                            if batch_shape is not None:
                                input_shape = batch_shape[1:]  # Convert batch_shape to input_shape
                                kwargs['input_shape'] = input_shape
                            super().__init__(**kwargs)
                    
                    # Load model with custom objects
                    model = tf.keras.models.load_model(
                        model_path,
                        custom_objects={'InputLayer': CustomInputLayer}
                    )
                    return model
                else:
                    # Re-raise if it's a different error
                    raise e
        
        # Use the custom loader
        model = load_model_with_custom_objects('english_indian_accent_classifier.h5')
        
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

def normalize_like_training(mfcc):
    """Apply the exact same normalization as used in training"""
    # This matches the normalize function from the training notebook
    x = mfcc.shape[0]  # time frames
    y = mfcc.shape[1]  # features (13)
    
    # Flatten for normalization
    v = mfcc.reshape(-1, x * y)
    
    # Calculate norm and prevent division by zero
    nm = np.linalg.norm(v, axis=1)
    nm = nm.reshape(-1, 1)
    nm = np.where(nm == 0, 1, nm)  # Prevent division by zero
    
    # Normalize
    v = v / nm
    
    # Reshape back
    v = v.reshape(-1, x, y)
    
    return v[0]  # Return the single sample

def extract_features(audio_path):
    """Extract MFCC features matching the training preprocessing exactly"""
    try:
        # Check if file exists and has content
        if not os.path.exists(audio_path):
            st.error(f"Audio file not found: {audio_path}")
            return None
            
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            st.error("Audio file is empty")
            return None
        
        st.info(f"Processing audio file: {os.path.basename(audio_path)} ({file_size/1024:.1f} KB)")
        
        # Load audio with exact same parameters as training
        try:
            # Load 5 seconds at 22050 Hz (matching training)
            y, sr = librosa.load(audio_path, sr=22050, duration=5.0)
            
            if len(y) == 0:
                st.error("No audio samples loaded")
                return None
                
            # Check if audio is too quiet
            if np.max(np.abs(y)) < 0.01:
                st.warning("Audio signal is very weak, results may be unreliable")
            
            # Trim silence and normalize
            y, _ = librosa.effects.trim(y, top_db=20)
            y = librosa.util.normalize(y)
            
            # Ensure we have enough audio (pad if necessary)
            min_length = int(5.0 * sr)  # 5 seconds worth of samples
            if len(y) < min_length:
                y = np.pad(y, (0, min_length - len(y)), mode='constant')
            
            st.info(f"Loaded {len(y)} audio samples at {sr} Hz")
            
        except Exception as e:
            st.error(f"Failed to load audio: {str(e)}")
            return None
        
        # Extract MFCC with exact same parameters as training
        try:
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=13,      # Exactly 13 coefficients
                n_fft=2048,     # Same FFT size
                hop_length=512  # Same hop length
            )
            
            # Transpose to match training format: (time_frames, features)
            mfcc = mfcc.T
            st.info(f"Extracted MFCC features: shape {mfcc.shape}")
            
        except Exception as e:
            st.error(f"Failed to extract MFCC features: {str(e)}")
            return None
        
        # Ensure exactly 499 time frames (matching training)
        target_frames = 499
        if mfcc.shape[0] < target_frames:
            # Pad with zeros
            padding = target_frames - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode='constant', constant_values=0)
            st.info(f"Padded to {target_frames} frames")
        elif mfcc.shape[0] > target_frames:
            # Truncate to exact size
            mfcc = mfcc[:target_frames, :]
            st.info(f"Truncated to {target_frames} frames")
        
        # Apply the same normalization as training
        mfcc_normalized = normalize_like_training(mfcc)
        
        st.success(f"Features ready: shape {mfcc_normalized.shape}")
        return mfcc_normalized
        
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def predict_accent(features):
    """Predict accent using improved logic"""
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
st.markdown("### 📹 Enter Candidate Video URL")
st.markdown("*Paste any video URL (Loom, YouTube, direct video link, etc.)*")
url = st.text_input(
    "Video URL:", 
    placeholder="https://www.loom.com/share/... or https://youtube.com/watch?v=... or direct video link",
    help="Supports Loom, YouTube, Vimeo, and most video platforms"
)

if st.button("🚀 Analyze English Accent", type="primary"):
    if not url.strip():
        st.warning("Please enter a video URL")
    elif not (url.startswith('http://') or url.startswith('https://')):
        st.warning("Please enter a valid URL starting with http:// or https://")
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
        
        # Quality warnings
        if result['confidence'] < 60:
            st.warning("⚠️ Low confidence prediction. Results may be unreliable.")
        
        # Detailed breakdown
        st.markdown("### Probability Breakdown")
        for accent, prob in sorted(result['all_probs'].items(), key=lambda x: x[1], reverse=True):
            if accent == result['accent']:
                st.success(f"**{accent.title()}**: {prob:.1f}% ⭐")
            else:
                st.info(f"{accent.title()}: {prob:.1f}%")
          # Hiring recommendation
        st.markdown("### 💼 English Proficiency Assessment")
        if result['is_native']:
            english_proficiency = "Native"
            if result['confidence'] > 80:
                recommendation = "✅ **Excellent English Communication**\n- Native English speaker with clear accent\n- Highly recommended for English-speaking roles"
            else:
                recommendation = "✅ **Good English Communication**\n- Native English speaker\n- Suitable for most English-speaking roles"
            
            st.success(f"""
            **{english_proficiency} English Speaker**
            - Accent: {result['accent'].title()}
            - Confidence: {result['confidence']:.1f}%
            
            {recommendation}
            """)
        else:
            # For Indian English - still English proficient but non-native
            english_proficiency = "Non-Native (Fluent)"
            if result['confidence'] > 80:
                recommendation = "ℹ️ **Fluent English Speaker**\n- Non-native but clear English accent\n- Suitable for most roles with global teams"
            else:
                recommendation = "ℹ️ **English Speaker**\n- Non-native accent detected\n- Assess based on role requirements"
            
            st.info(f"""
            **{english_proficiency} English Speaker**
            - Accent: {result['accent'].title()}
            - Confidence: {result['confidence']:.1f}%
            
            {recommendation}
            """)

# Instructions
with st.expander("📝 How to Use This Tool"):
    st.markdown("""
    **Steps to Analyze Candidate's English Accent:**
    1. **Get candidate video**: Ask for Loom recording, YouTube upload, or direct video link
    2. **Paste the URL** in the field above
    3. **Click Analyze** and get instant English proficiency assessment
    4. **Use results** to evaluate English communication skills
    
    **Supported Video Sources:**
    - 🎬 **Loom recordings** (loom.com links)
    - 📹 **YouTube videos** (youtube.com/watch links)
    - 🔗 **Direct video links** (.mp4, .mov, .avi files)
    - 📺 **Most video platforms** supported by yt-dlp
    
    **English Accents Detected:**
    - 🇺🇸 **American** (Native English)
    - 🇬🇧 **British** (Native English)  
    - 🇦🇺 **Australian** (Native English)
    - 🏴 **Welsh** (Native English)
    - 🇮🇳 **Indian English** (Non-native but fluent)
    
    **Best Practices:**
    - Use videos with 30+ seconds of clear speech
    - Ensure minimal background noise
    - Single speaker preferred
    - Ask candidates to introduce themselves or describe their experience
    """)

with st.expander("⚙️ Technical Details"):
    st.markdown("""
    **Model Information:**
    - Architecture: Convolutional Neural Network (CNN)
    - Features: 13 MFCC coefficients over 499 time frames
    - Training: AccentDB dataset
    - Processing: Enhanced feature extraction matching training data
    
    **Quality Indicators:**
    - **Confidence**: How certain the model is about the prediction
    - Lower confidence may indicate mixed accents or unclear audio
    """)

with st.expander("📊 Understanding Confidence Scores"):
    st.markdown("""
    **Confidence Score Interpretation:**
    
    - **90-100%**: Very high confidence - reliable classification
    - **80-89%**: High confidence - good classification reliability  
    - **70-79%**: Moderate confidence - generally reliable
    - **60-69%**: Lower confidence - consider additional samples
    - **Below 60%**: Low confidence - may need clearer audio or longer sample
    
    **Factors Affecting Confidence:**
    - Audio quality and clarity
    - Background noise levels
    - Length of speech sample
    - Speaker's consistency
    - Audio compression/encoding
    
    **For Hiring Decisions:**
    - High confidence (80%+) results are reliable for assessment
    - Lower confidence may indicate mixed accent or unclear audio
    - Consider requesting additional voice samples if confidence is low
    """)

st.markdown("---")
st.markdown("**🎯 Built for Hiring Teams** | Evaluate English communication skills instantly")
st.markdown("*Model trained on diverse English accents for accurate candidate assessment*")