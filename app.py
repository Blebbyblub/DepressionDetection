# app.py
import streamlit as st
import pandas as pd
import numpy as np
from depression_predictor import DepressionSeverityPredictor
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Analisis Tingkat Depresi - Indonesia",
    page_icon="🧠",
    layout="wide"
)

# Initialize predictor with better error handling
@st.cache_resource
def load_predictor():
    try:
        predictor = DepressionSeverityPredictor()
        return predictor
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.info("""
        **Pastikan file-file berikut ada di direktori Anda:**
        - `depression_model.h5` (model yang sudah dilatih)
        - `tokenizer.pkl` (tokenizer)
        - `label_encoder.pkl` (label encoder)
        
        Jika file-file ini tidak ada, Anda perlu melatih model terlebih dahulu.
        """)
        return None

def main():
    st.title("🧠 Analisis Tingkat Depresi - Indonesia")
    st.markdown("Analisis teks untuk mendeteksi tingkat depresi menggunakan AI")
    
    # Language selector
    col1, col2 = st.columns([3, 1])
    with col2:
        language = st.selectbox("Bahasa / Language", ["Indonesia", "English"])
    
    # Load model
    predictor = load_predictor()
    
    if predictor is None:
        show_model_not_available_message(language)
        return
    
    st.success("✅ Model berhasil dimuat!" if language == "Indonesia" else "✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.header("Opsi Analisis" if language == "Indonesia" else "Analysis Options")
    analysis_mode = st.sidebar.radio(
        "Pilih Mode Analisis:" if language == "Indonesia" else "Select Analysis Mode:",
        ["Teks Tunggal", "Analisis Batch", "Unggah File"] if language == "Indonesia" 
        else ["Single Text", "Batch Analysis", "File Upload"]
    )
    
    if analysis_mode in ["Teks Tunggal", "Single Text"]:
        single_text_analysis(predictor, language)
    elif analysis_mode in ["Analisis Batch", "Batch Analysis"]:
        batch_analysis(predictor, language)
    else:
        file_upload_analysis(predictor, language)

def show_model_not_available_message(language):
    """Show message when model is not available"""
    if language == "Indonesia":
        st.warning("""
        ### 🚨 Model Tidak Tersedia
        
        Untuk menggunakan aplikasi ini, Anda perlu:
        
        1. **Latih model terlebih dahulu** menggunakan script training
        2. **Pastikan file-file ini ada di direktori Anda**:
           - `depression_model.h5`
           - `tokenizer.pkl` 
           - `label_encoder.pkl`
        """)
    else:
        st.warning("""
        ### 🚨 Model Not Available
        
        To use this app, you need to:
        
        1. **Train the model first** using the training script
        2. **Ensure these files are in your directory**:
           - `depression_model.h5`
           - `tokenizer.pkl` 
           - `label_encoder.pkl`
        """)

def single_text_analysis(predictor, language):
    """Single text analysis in Indonesian/English"""
    
    if language == "Indonesia":
        st.header("Analisis Teks Tunggal")
        placeholder = "Jelaskan perasaan Anda atau tempel teks untuk dianalisis..."
        button_text = "Analisis Teks"
        analyzing_text = "Menganalisis..."
    else:
        st.header("Single Text Analysis")
        placeholder = "Describe how you're feeling or paste text to analyze..."
        button_text = "Analyze Text"
        analyzing_text = "Analyzing..."
    
    text_input = st.text_area(
        "Masukkan teks untuk dianalisis:" if language == "Indonesia" else "Enter text to analyze:",
        height=150,
        placeholder=placeholder
    )
    
    if st.button(button_text, type="primary") and text_input:
        with st.spinner(analyzing_text):
            result = predictor.predict_severity(text_input)
            
            if 'error' in result:
                st.error(f"❌ Error analisis: {result['error']}" if language == "Indonesia" else f"❌ Analysis error: {result['error']}")
                return
            
            # Display language detection info
            if result['was_translated']:
                st.info(f"🌐 Teks terdeteksi: Bahasa Indonesia → Diterjemahkan ke Inggris" 
                       if language == "Indonesia" else f"🌐 Detected: Indonesian → Translated to English")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                # Severity with color coding
                severity = result['severity']
                confidence = result['confidence']
                
                severity_colors = {
                    'minimum': '🟢',
                    'mild': '🟡', 
                    'moderate': '🟠',
                    'severe': '🔴'
                }
                
                # Severity labels in Indonesian
                severity_labels_id = {
                    'minimum': 'MINIMAL',
                    'mild': 'RINGAN', 
                    'moderate': 'SEDANG',
                    'severe': 'BERAT'
                }
                
                st.subheader("Hasil Analisis" if language == "Indonesia" else "Analysis Result")
                display_severity = severity_labels_id.get(severity, severity.upper()) if language == "Indonesia" else severity.upper()
                st.markdown(f"### {severity_colors.get(severity, '⚪')} {display_severity}")
                st.metric("Tingkat Kepercayaan" if language == "Indonesia" else "Confidence", f"{confidence:.2%}")
                
                # Risk assessment in Indonesian/English
                if severity in ['severe', 'moderate'] and confidence > 0.7:
                    risk_msg = "🚨 RISIKO TINGGI - Pertimbangkan untuk mencari bantuan profesional" if language == "Indonesia" else "🚨 HIGH RISK - Consider seeking professional help"
                    st.error(risk_msg)
                elif severity == 'mild':
                    risk_msg = "⚠️  RISIKO SEDANG - Pantau dengan cermat" if language == "Indonesia" else "⚠️  MODERATE RISK - Monitor closely"
                    st.warning(risk_msg)
                else:
                    risk_msg = "✅ RISIKO RENDAH - Lanjutkan pemantauan" if language == "Indonesia" else "✅ LOW RISK - Continue monitoring"
                    st.success(risk_msg)
            
            with col2:
                # Probability chart
                prob_labels = {k: severity_labels_id.get(k, k) for k in result['probabilities'].keys()} if language == "Indonesia" else result['probabilities'].keys()
                
                prob_df = pd.DataFrame({
                    'Tingkat' if language == "Indonesia" else 'Severity': list(prob_labels.values()),
                    'Probabilitas' if language == "Indonesia" else 'Probability': list(result['probabilities'].values())
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Tingkat' if language == "Indonesia" else 'Severity', 
                    y='Probabilitas' if language == "Indonesia" else 'Probability',
                    color='Tingkat' if language == "Indonesia" else 'Severity',
                    color_discrete_map={
                        'MINIMAL' if language == "Indonesia" else 'minimum': 'green',
                        'RINGAN' if language == "Indonesia" else 'mild': 'yellow',
                        'SEDANG' if language == "Indonesia" else 'moderate': 'orange', 
                        'BERAT' if language == "Indonesia" else 'severe': 'red'
                    },
                    title="Probabilitas Tingkat Depresi" if language == "Indonesia" else "Severity Probabilities"
                )
                fig.update_layout(yaxis_tickformat='.0%')
                st.plotly_chart(fig, use_container_width=True)
            
            # Show translation details if applicable
            if result['was_translated']:
                with st.expander("📝 Detail Terjemahan" if language == "Indonesia" else "📝 Translation Details"):
                    st.write("**Teks Asli (Bahasa Indonesia):**")
                    st.text(result['original_text'])
                    st.write("**Teks Hasil Terjemahan (English):**")
                    st.text(result['translated_text'])

def batch_analysis(predictor, language):
    """Batch analysis in Indonesian/English"""
    
    if language == "Indonesia":
        st.header("Analisis Batch")
        placeholder = "Masukkan setiap teks pada baris baru..."
        button_text = "Analisis Batch"
    else:
        st.header("Batch Analysis")
        placeholder = "Enter each text on a new line..."
        button_text = "Analyze Batch"
    
    st.markdown("Masukkan beberapa teks untuk dianalisis (satu per baris):" if language == "Indonesia" else "Enter multiple texts to analyze (one per line):")
    batch_text = st.text_area(
        "Teks Batch:" if language == "Indonesia" else "Batch Texts:",
        height=200,
        placeholder=placeholder
    )
    
    if st.button(button_text) and batch_text:
        texts = [text.strip() for text in batch_text.split('\n') if text.strip()]
        
        with st.spinner(f"Menganalisis {len(texts)} teks..." if language == "Indonesia" else f"Analyzing {len(texts)} texts..."):
            results = predictor.predict_batch(texts)
            
            # Create results table
            results_data = []
            for result in results:
                results_data.append({
                    'Pratinjau Teks' if language == "Indonesia" else 'Text Preview': result['original_text'][:100] + '...' if len(result['original_text']) > 100 else result['original_text'],
                    'Tingkat' if language == "Indonesia" else 'Severity': result['severity'],
                    'Kepercayaan' if language == "Indonesia" else 'Confidence': f"{result['confidence']:.2%}",
                    'Bahasa' if language == "Indonesia" else 'Language': 'ID' if result.get('was_translated', False) else 'EN'
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            st.subheader("Statistik Ringkasan" if language == "Indonesia" else "Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            severity_counts = results_df['Tingkat' if language == "Indonesia" else 'Severity'].value_counts()
            
            with col1:
                st.metric("Total Teks" if language == "Indonesia" else "Total Texts", len(texts))
            with col2:
                st.metric("Paling Umum" if language == "Indonesia" else "Most Common", severity_counts.index[0] if not severity_counts.empty else "N/A")
            with col3:
                high_risk = len([r for r in results if r['severity'] in ['severe', 'moderate'] and r['confidence'] > 0.7])
                st.metric("Teks Berisiko Tinggi" if language == "Indonesia" else "High Risk Texts", high_risk)
            with col4:
                avg_confidence = np.mean([r['confidence'] for r in results])
                st.metric("Rata-rata Kepercayaan" if language == "Indonesia" else "Avg Confidence", f"{avg_confidence:.2%}")

def file_upload_analysis(predictor, language):
    """File upload analysis in Indonesian/English"""
    
    if language == "Indonesia":
        st.header("Analisis Unggah File")
        help_text = "File harus berisi kolom dengan teks untuk dianalisis"
        button_text = "Analisis File"
    else:
        st.header("File Upload Analysis")
        help_text = "File should contain a column with text to analyze"
        button_text = "Analyze File"
    
    uploaded_file = st.file_uploader(
        "Unggah file CSV atau Excel dengan data teks" if language == "Indonesia" else "Upload a CSV or Excel file with text data",
        type=['csv', 'xlsx'],
        help=help_text
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File berhasil dimuat! Ditemukan {len(df)} baris" if language == "Indonesia" else f"✅ File loaded successfully! Found {len(df)} rows")
            
            # Select text column
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            if text_columns:
                text_column = st.selectbox("Pilih kolom teks:" if language == "Indonesia" else "Select text column:", text_columns)
                
                if st.button(button_text):
                    with st.spinner("Menganalisis isi file..." if language == "Indonesia" else "Analyzing file contents..."):
                        results = []
                        for text in df[text_column]:
                            result = predictor.predict_severity(str(text))
                            results.append(result)
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(results)
                        
                        # Merge results with original data
                        final_df = pd.concat([df, results_df[['severity', 'confidence']]], axis=1)
                        
                        st.subheader("Hasil Analisis" if language == "Indonesia" else "Analysis Results")
                        st.dataframe(final_df, use_container_width=True)
                        
                        # Download results
                        csv = final_df.to_csv(index=False)
                        st.download_button(
                            label="Unduh Hasil sebagai CSV" if language == "Indonesia" else "Download Results as CSV",
                            data=csv,
                            file_name="hasil_analisis_depresi.csv" if language == "Indonesia" else "depression_analysis_results.csv",
                            mime="text/csv"
                        )
            else:
                st.error("Tidak ditemukan kolom teks dalam file yang diunggah." if language == "Indonesia" else "No text columns found in the uploaded file.")
                
        except Exception as e:
            st.error(f"Error memproses file: {e}" if language == "Indonesia" else f"Error processing file: {e}")

if __name__ == "__main__":
    main()