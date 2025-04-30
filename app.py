import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from gensim.models import KeyedVectors
from scipy.spatial import distance
import streamlit as st
import google.generativeai as genai
import plotly.express as px
import time
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from essayGradeHelper import essay_to_wordlist, get_model, getAvgFeatureVecs

# Configure Gemini API
gemini_key = 'AIzaSyCGJA-usGDbFY0n_BvhF3pyGQeANSzleDY'
genai.configure(api_key=gemini_key)
ocr = genai.GenerativeModel("gemini-2.0-flash")
gen_model = genai.GenerativeModel("gemini-2.0-flash")

# --- Page Config and Styling ---
st.set_page_config(
    page_title="EssayMind AI | Essay Grader", 
    layout="wide", 
    page_icon="📝",
    initial_sidebar_state="expanded"
)

# Modern UI styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #6C63FF;
    --primary-light: #8983FF;
    --secondary: #4A47A3;
    --dark: #2C2C54;
    --light: #F8F9FA;
    --success: #48BF91;
    --warning: #FFBE0B;
    --danger: #FF5757;
}

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: var(--light);
    color: #333;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 h4 {
    color: white;
    font-weight: 600;
}

.stButton>button {
    background-color: var(--primary);
    color: white;
    border: none;
    padding: 0.75em 1.5em;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
    box-shadow: 0 2px 5px rgba(108, 99, 255, 0.2);
}

.stButton>button:hover {
    background-color: var(--primary-light);
    box-shadow: 0 4px 8px rgba(108, 99, 255, 0.3);
    transform: translateY(-2px);
}

.stTextArea textarea, .stTextInput input {
    background-color: black;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
            
/* Global override for all text input and textarea elements */
textarea, input[type="text"], input[type="email"], input[type="password"] {
    background-color: #1E1E2E !important;  /* black-ish background */
    color: #FFFFFF !important;            /* white text */
    border: 1px solid #444 !important;    /* subtle border */
    border-radius: 6px !important;
    padding: 10px !important;
}

    /* Specifically target email history ones inside sidebar */
section[data-testid="stSidebar"] .email-history-container textarea {
    background-color: #1E1E2E !important;
    color: #FFFFFF !important;
    border: 1px solid #555 !important;
    border-radius: 6px !important;
}



.stTabs [role="tab"] {
    color: var(--dark);
    font-weight: 500;
}

.stTabs [role="tab"][aria-selected="true"] {
    color: var(--primary);
    border-bottom-color: var(--primary);
    font-weight: 600;
}

.card {
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    padding: 1.5rem;
    background-color: black;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    transform: translateY(-3px);
}

.card-header {
    border-bottom: 1px solid #f0f0f0;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
}

.metric-container {
    text-align: center;
    padding: 1rem;
    background: linear-gradient(145deg, #ffffff, #f0f0f0);
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}

.metric-container:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}

.metric-label {
    font-size: 0.9rem;
    font-weight: 500;
    color: #555;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--primary);
    line-height: 1;
}

.metric-subtitle {
    font-size: 0.8rem;
    color: #888;
    margin-top: 0.3rem;
}

.success-box {
    padding: 1rem;
    background-color: rgba(72, 191, 145, 0.1);
    border-left: 4px solid var(--success);
    border-radius: 4px;
    margin-bottom: 1rem;
}

.warning-box {
    padding: 1rem;
    background-color: rgba(255, 190, 11, 0.1);
    border-left: 4px solid var(--warning);
    border-radius: 4px;
    margin-bottom: 1rem;
}

.danger-box {
    padding: 1rem;
    background-color: rgba(255, 87, 87, 0.1);
    border-left: 4px solid var(--danger);
    border-radius: 4px;
    margin-bottom: 1rem;
}

.feedback-section {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin-top: 1rem;
}

.stProgress .st-ey {
    background-color: var(--primary);
}

/* Custom file uploader */
.file-uploader {
    border: 2px dashed #e0e0e0;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    background-color: #f9f9f9;
    transition: all 0.3s ease;
    cursor: pointer;
}

.file-uploader:hover {
    border-color: var(--primary);
    background-color: rgba(108, 99, 255, 0.05);
}

.history-item {
    padding: 0.75rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    background-color: white;
    border-left: 3px solid var(--primary);
    transition: all 0.2s ease;
}

.history-item:hover {
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

/* Animated loader */
@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}

.loading-pulse {
    animation: pulse 1.5s infinite ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'grading_history' not in st.session_state:
    st.session_state.grading_history = []

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# --- Essay Set Data ---
essay_set_data = {
    "essay_set": [1, 2, 3, 4, 5, 6, 7, 8],
    "type_of_essay": ["The Impact of Computers on Modern Society"] +["Censorship in the Libraries"] + ["The Role of Setting in the Cyclist’s Journey"] +["How Landscapes Trigger Identity and Belonging in Winter Hibiscus"]+ ["Gratitude and Warmth in Narciso Rodriguez's Memoir"]+["Dirigible Docking Challenges at the Empire State Building"]+["Patience: A Personal Story"]+["Laughter is a medicine"],
    "grade_level": [8, 10, 10, 10, 8, 10, 7, 10],
    "min_domain1_score": [2, 1, 0, 0, 0, 0, 0, 0],
    "max_domain1_score": [12, 6, 3, 3, 4, 4, 30, 60]
}
essay_set_df = pd.DataFrame(essay_set_data)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/null/brain.png", width=60)
    st.markdown("### EssayMind AI")
    st.markdown("🧠 **AI-Powered Essay Analysis & Grading**")
    
    st.markdown("---")
    
    # Settings section
    st.markdown("### ⚙️ Settings")
    if st.checkbox("Dark Mode", value=st.session_state.dark_mode):
        st.session_state.dark_mode = True
        st.markdown("""
        <style>
        :root {
            --primary: #8B80FF;
            --primary-light: #A39BFF;
            --secondary: #6259DB;
            --dark: #F8F9FA;
            --light: #1E1E2E;
        }
        
        html, body, [class*="css"] {
            background-color: #1E1E2E;
            color: #E0E0E0;
        }
        
        .stTextArea textarea, .stTextInput input {
            background-color: #2D2D44;
            border: 1px solid #3D3D5D;
            color: #E0E0E0;
        }
        
        .card {
            background-color: #2D2D44;
            border-color: #3D3D5D;
        }
        
        .stTabs [role="tab"] {
            color: #E0E0E0;
        }
        
        .history-item {
            background-color: #2D2D44;
        }
        
        h1, h2, h3, h4, h5 {
            color: #E0E0E0;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.session_state.dark_mode = False
    
    # Language model selection
    st.selectbox("Language Model", ["BERT", "LSTM"], index=0)
    
    # Grading history
    st.markdown("### 📜 Recent Activity")

    recent = []

# Collect recent graded essays
    if 'grading_history' in st.session_state:
        recent.extend([
        {
            "date": item["date"],
            "score": item["score"],
            "type": "Graded Essay",
            "detail": f"Essay Set {item['essay_set']} - {item['word_count']} words"
        } for item in st.session_state.grading_history
    ])

# Collect recent model comparisons
    if 'model_answer_history' in st.session_state:
        recent.extend([
        {
            "date": item["date"],
            "score": item["grade"],
            "type": "Model Comparison",
            "detail": f"Q: {item['question'][:30]}..."
        } for item in st.session_state.model_answer_history
    ])

# Sort by newest first
    recent.sort(key=lambda x: x["date"], reverse=True)

    if not recent:
        st.caption("No recent activity")
    else:
        for i, item in enumerate(recent[:5]):
            with st.expander(f"{item['date']} - {item['type']} - Score: {item['score']}"):
                st.caption(item["detail"])

                if st.button("View Details", key=f"history_{i}"):
                    # You would implement functionality to show details
                    pass

    st.markdown("---")
    st.caption("© 2025 EssayMind AI | v2.0.4")
    
# --- Grading Functions ---
def grade_essay(essay_content, essay_set_num):
    if len(essay_content) > 40:
        word_count = len(essay_content.split())
        
        # Show loading animation
        with st.spinner("Analyzing essay..."):
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulating work being done
                time.sleep(0.02)
                progress_bar.progress(i + 1)
        
        # Load models and predict
        word2vec_model = KeyedVectors.load_word2vec_format('word2vecmodel (1).bin', binary=True)
        lstm_model = get_model()
        lstm_model.load_weights('final_lstm (1).h5')
        clean_test = [essay_to_wordlist(essay_content, remove_stopwords=True)]
        vectors = getAvgFeatureVecs(clean_test, word2vec_model, 300)
        vectors = np.array(vectors).reshape((len(vectors), 1, 300))
        pred = lstm_model.predict(vectors)
        
        # Get the valid score range for this essay set
        essay_info = essay_set_df[essay_set_df["essay_set"] == essay_set_num].iloc[0]
        min_score = essay_info["min_domain1_score"]
        max_score = essay_info["max_domain1_score"]
        
        # Constrain the prediction to the valid range
        raw_grade = pred[0][0]
        grade = min(max_score, max(min_score, round(raw_grade)))
        
        # Generate AI feedback
        try:
            feedback_prompt = f"""
            Analyze this student essay and provide constructive feedback:
            Essay: {essay_content[:2000]}...
            
            Provide feedback in these areas:
            1. Main strengths (2-3 points)
            2. Areas for improvement (2-3 specific points)
            3. Writing style and clarity
            4. Organization and structure
            
            Keep your response under 200 words and focus on being helpful.
            """
            
            response = gen_model.generate_content(feedback_prompt)
            ai_feedback = response.text
        except:
            ai_feedback = "AI feedback generation failed. Please try again later."
        
        # Add to history
        st.session_state.grading_history.append({
        "date": datetime.now().strftime("%m/%d/%Y %H:%M"),
        "essay_set": essay_set_num,
        "score": grade,
        "word_count": word_count,
        "essay_snippet": essay_content[:100] + "...",
        "ai_feedback": ai_feedback  # ✅ Save feedback!
        })
        
        return grade, word_count, ai_feedback
    
    return None, 0, "Essay too short for analysis."

# --- Synoptic JSON Store ---
def load_question_synoptics():
    if os.path.exists("question_synoptics.json"):
        with open("question_synoptics.json", 'r') as f:
            return json.load(f)
    return {}

def save_question_synoptics(data):
    with open("question_synoptics.json", 'w') as f:
        json.dump(data, f, indent=4)

# --- Header ---
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <img src="https://img.icons8.com/fluency/96/null/brain.png" width="50">
    <div style="margin-left: 15px;">
        <h1 style="margin: 0; font-size: 2rem;">EssayMind AI</h1>
        <p style="margin: 0; opacity: 0.8;">Intelligent Essay Analysis & Grading</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📝 Essay Grading", 
    "🧠 Model Essay Comparison",
    "📧 Email Results"
])

# --- Tab 1: Essay Grading ---
with tab1:
    st.markdown("## Grade Your Essays")
    st.write("Submit an essay for AI-powered analysis and scoring")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        
        st.markdown('<div class="card-header"><h3>Essay Input</h3></div>', unsafe_allow_html=True)
        
        selected_question = st.selectbox(
            "Select Essay Set", 
            [f"Essay Set {i}" for i in range(1, 9)],
            help="Choose the appropriate essay set for grading"
        )
        question_number = int(selected_question.split(" ")[-1])
        selected = essay_set_df[essay_set_df["essay_set"] == question_number]
        min_score = selected["min_domain1_score"].values[0]
        max_score = selected["max_domain1_score"].values[0]
        essay_type = selected["type_of_essay"].values[0]
        grade_level = selected["grade_level"].values[0]
        
        st.info(f"📋 **Essay Type**: {essay_type} | **Grade Level**: {grade_level} | **Score Range**: {min_score}-{max_score}")
        
        input_type = st.radio("Input Method", ["📝 Text Input", "📷 Handwritten Image"], horizontal=True)
        
        essay_content = ""
        if input_type == "📝 Text Input":
            essay_content = st.text_area("Enter your essay:", height=250, placeholder="Start typing your essay here...")
            word_count = len(essay_content.split()) if essay_content else 0
            st.caption(f"Word count: {word_count}")
            
            grade_button = st.button("🚀 Grade Essay", use_container_width=True)
            
        else:
            st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Drop your handwritten essay image here", type=["png", "jpg", "jpeg"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file:
                col1a, col1b = st.columns([1, 1])
                with col1a:
                    img = Image.open(uploaded_file)
                    st.image(img, caption="Uploaded Essay", use_container_width=True)
                
                with col1b:
                    with st.spinner("🔍 Extracting text with OCR..."):
                        response = ocr.generate_content(["Extract full text from this image", img])
                        essay_content = response.text
                        st.text_area("Extracted Text", value=essay_content, height=150)
                        word_count = len(essay_content.split()) if essay_content else 0
                        st.caption(f"Word count: {word_count}")
                
                grade_button = st.button("🚀 Grade Essay", use_container_width=True)
            else:
                grade_button = False
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        
        st.markdown('<div class="card-header"><h3>Score & Analysis</h3></div>', unsafe_allow_html=True)
        
        if grade_button and essay_content:
            grade, word_count, ai_feedback = grade_essay(essay_content, question_number)
            
            if grade is not None:
                # Calculate score percentage
                score_percentage = (grade - min_score) / (max_score - min_score) * 100 if max_score > min_score else 0
                score_percentage = max(0, min(100, score_percentage))
                
                # Display score in a nice metric container
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Essay Score</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{grade}/{max_score}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-subtitle">{score_percentage:.1f}% of maximum score</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Score gauge chart
                fig = px.pie(values=[score_percentage, 100-score_percentage], 
                            names=['Score', 'Remaining'],
                            hole=0.7,
                            color_discrete_sequence=['#6C63FF', '#E0E0E0'])
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    annotations=[dict(text=f"{score_percentage:.0f}%", x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance indicators
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Words", word_count)
                with col2b:
                    st.metric("Reading Level", f"Grade {grade_level}")
                
                # Feedback section
                st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
                st.markdown("### AI Feedback")
                st.write(ai_feedback)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Performance assessment
                if score_percentage >= 80:
                    st.markdown('<div class="success-box">Excellent work! Your essay demonstrates strong understanding and execution.</div>', unsafe_allow_html=True)
                elif score_percentage >= 60:
                    st.markdown('<div class="warning-box">Good effort! There are areas for improvement but you have a solid foundation.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="danger-box">This essay needs further development. Review the feedback carefully.</div>', unsafe_allow_html=True)
            
            st.download_button("📥 Download Report", 
                             f"Essay Analysis Report\n\nScore: {grade}/{max_score}\nWord Count: {word_count}\n\nFeedback:\n{ai_feedback}",
                             file_name=f"essay_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                             mime="text/plain")
        
        else:
            st.markdown('<div style="text-align: center; padding: 2rem;">', unsafe_allow_html=True)
            st.markdown('📊 Enter your essay and click "Grade Essay" to receive analysis', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 2: Synoptic Grading ---
with tab2:
    st.markdown("## Model Essay Comparison")
    st.write("Compare student's essay against model Essay for precise evaluation")

    question_synoptics = load_question_synoptics()

    # Initialize session state variables if they don't exist
    if 'comparison_done' not in st.session_state:
        st.session_state.comparison_done = False
    if 'predicted_grade' not in st.session_state:
        st.session_state.predicted_grade = 0.0
    if 'similarity_score' not in st.session_state:
        st.session_state.similarity_score = 0.0
    if 'comparison_feedback' not in st.session_state:
        st.session_state.comparison_feedback = ""
    if 'teacher_comment' not in st.session_state:
        st.session_state.teacher_comment = ""
    if 'final_grade' not in st.session_state:
        st.session_state.final_grade = 0.0

    col1, col2 = st.columns([1, 1])

    with col1:

        st.markdown('<div class="card-header"><h3>Model Essay Management</h3></div>', unsafe_allow_html=True)

        with st.expander("➕ Add New Model Essay", expanded=False):
            question_text = st.text_input("Topic", placeholder="Enter the essay topic")
            synoptic_text = st.text_area("Model Essay", placeholder="Enter the ideal model Essay for this topic", height=150)

            if st.button("Save Model Essay"):
                if question_text and synoptic_text:
                    question_synoptics[question_text] = synoptic_text
                    save_question_synoptics(question_synoptics)
                    st.success("✅ Model essay saved successfully!")
                else:
                    st.warning("⚠️ Please fill both fields.")

        # Display existing questions
        if question_synoptics:
            st.markdown("### Saved Essays")
            for i, (question, answer) in enumerate(question_synoptics.items()):
                with st.expander(f"📌 {question[:50]}{'...' if len(question) > 50 else ''}"):
                    st.write(answer[:200] + "..." if len(answer) > 200 else answer)
                    if st.button("Delete", key=f"delete_{i}"):
                        del question_synoptics[question]
                        save_question_synoptics(question_synoptics)
                        st.rerun()

        else:
            st.info("No model answers saved yet. Add one using the form above.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        st.markdown('<div class="card-header"><h3>Student Essay Comparison</h3></div>', unsafe_allow_html=True)

        if question_synoptics:
            selected_question = st.selectbox("Select Essay", list(question_synoptics.keys()))
            selected_synoptic = question_synoptics[selected_question]

            max_marks = st.slider("Maximum Score", 1, 100, 10)

            input_method = st.radio("Input Method", ["✏️ Text Input", "📷 Image Upload"], horizontal=True)
            student_answer = ""

            if input_method == "✏️ Text Input":
                student_answer = st.text_area("Student Essay", placeholder="Enter the student's Essay here", height=150)
            else:
                uploaded_file = st.file_uploader("Upload Answer Image", type=["png", "jpg", "jpeg"], key="synoptic_upload")
                if uploaded_file:
                    img = Image.open(uploaded_file)
                    st.image(img, caption="Uploaded Answer", width=300)
                    with st.spinner("🔍 Extracting text..."):
                        response = ocr.generate_content(["Extract text", img])
                        student_answer = response.text
                        st.text_area("Extracted Text", value=student_answer, height=100)

            # Handler for the compare button
            def handle_compare():
                from sentence_transformers import SentenceTransformer
                if student_answer:
                    with st.spinner("Analyzing similarity..."):
                        model = SentenceTransformer('bert-base-nli-mean-tokens')
                        syn_vec = model.encode([selected_synoptic])[0]
                        stud_vec = model.encode([student_answer])[0]
                        sim_score = 1 - distance.cosine(syn_vec, stud_vec)

                        # Apply length normalization
                        length_norm = min(len(selected_synoptic), len(student_answer)) / max(len(selected_synoptic), len(student_answer))
                        sim_score *= length_norm

                        # Logistic function parameters
                        k = 8  # Steepness of the curve
                        c = 0.5  # Point of inflection (where the curve changes rapidly)

                        # If the similarity score is very low (below 0.3), grade should be 0
                        if sim_score < 0.3:
                            grade = 0
                        else:
                            # Apply logistic (sigmoidal) scaling to the similarity score
                            grade = max_marks / (1 + np.exp(-k * (sim_score - c)))

                        # Store in session state
                        st.session_state.similarity_score = sim_score
                        st.session_state.predicted_grade = round(grade, 1)
                        st.session_state.final_grade = round(grade, 1)  # Initialize final grade with predicted grade
                        
                        # Generate feedback
                        try:
                            compare_prompt = f"""
                            Compare these two answers:

                            MODEL ANSWER: {selected_synoptic[:500]}...
                            STUDENT ANSWER: {student_answer[:500]}...

                            Provide brief feedback on:
                            1. Key points included/missed
                            2. Areas of strength
                            3. Suggestions for improvement

                            Keep it under 150 words.
                            """

                            response = gen_model.generate_content(compare_prompt)
                            st.session_state.comparison_feedback = response.text
                        except:
                            st.session_state.comparison_feedback = "Could not generate detailed feedback."
                        
                        st.session_state.comparison_done = True
                        
                        # Store current student answer and question for later use
                        st.session_state.current_student_answer = student_answer
                        st.session_state.current_selected_question = selected_question
                        st.session_state.current_selected_synoptic = selected_synoptic
                        st.session_state.current_max_marks = max_marks

            # Compare button
            if st.button("Compare Essays", use_container_width=True):
                handle_compare()

            # Display results only if comparison has been done
            if st.session_state.comparison_done:
                # Similarity score display
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Similarity Score</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{st.session_state.similarity_score:.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Predicted Grade", f"{st.session_state.predicted_grade}/{st.session_state.current_max_marks}")
                
                # Add teacher grade adjustment section
                st.markdown("### Teacher Grade Adjustment")
                st.write("Adjust the final grade if needed:")
                
                # Function to update final grade in session state
                def update_final_grade():
                    st.session_state.final_grade = st.session_state.temp_final_grade
                
                # Create a number input for teacher to modify the grade
                st.number_input(
                    "Final Grade", 
                    min_value=0.0, 
                    max_value=float(st.session_state.current_max_marks), 
                    value=float(st.session_state.final_grade),
                    step=0.5,
                    format="%.1f",
                    key="temp_final_grade",
                    on_change=update_final_grade
                )
                
                # Show if grade was modified
                if st.session_state.final_grade != st.session_state.predicted_grade:
                    st.info(f"Grade adjusted from AI prediction of {st.session_state.predicted_grade} to {st.session_state.final_grade}")
                
                # Function to update teacher comment in session state
                def update_teacher_comment():
                    st.session_state.teacher_comment = st.session_state.temp_teacher_comment
                
                # Add a comment field for teacher's feedback
                st.text_area(
                    "Teacher's Comment (Optional)", 
                    value=st.session_state.teacher_comment,
                    placeholder="Add additional feedback or justification for grade adjustment",
                    key="temp_teacher_comment",
                    on_change=update_teacher_comment
                )

                # Visual indicators based on final grade
                if st.session_state.final_grade == 0:
                    st.markdown('<div class="danger-box">❌ Content is irrelevant or minimal. Very low grade assigned.</div>', unsafe_allow_html=True)
                elif st.session_state.final_grade > 0.85 * st.session_state.current_max_marks:
                    st.markdown('<div class="success-box">✅ Excellent match with model answer.</div>', unsafe_allow_html=True)
                elif st.session_state.final_grade > 0.7 * st.session_state.current_max_marks:
                    st.markdown('<div class="warning-box">⚠️ Partial similarity. Some key points matched.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="danger-box">❌ Significant differences from model answer.</div>', unsafe_allow_html=True)

                # Display AI feedback
                st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
                st.write(st.session_state.comparison_feedback)
                st.markdown('</div>', unsafe_allow_html=True)

                # Save button for finalizing the grade
                if st.button("Save Final Assessment", use_container_width=True):
                    # ✅ Save to model answer history with the adjusted grade
                    if 'model_answer_history' not in st.session_state:
                        st.session_state.model_answer_history = []

                    st.session_state.model_answer_history.append({
                        "date": datetime.now().strftime("%m/%d/%Y %H:%M"),
                        "student_answer": st.session_state.current_student_answer,
                        "question": st.session_state.current_selected_question,
                        "model_answer": st.session_state.current_selected_synoptic,
                        "similarity_score": st.session_state.similarity_score,
                        "predicted_grade": st.session_state.predicted_grade,
                        "final_grade": st.session_state.final_grade,
                        "grade": st.session_state.final_grade,  # For backward compatibility
                        "max_marks": st.session_state.current_max_marks,
                        "ai_feedback": st.session_state.comparison_feedback,
                        "teacher_comment": st.session_state.teacher_comment
                    })
                    
                    st.success("✅ Assessment saved successfully!")
                    
                    # Reset the comparison state to allow for a new comparison
                    st.session_state.comparison_done = False
                    st.session_state.teacher_comment = ""

        else:
            st.info("Please add model answers in the left panel first.")

        st.markdown('</div>', unsafe_allow_html=True)
# Modified Tab 3 to include Model Essay Comparison email feature
with tab3:
    st.markdown("## Email Results to Students")
    st.write("Send graded essay results and feedback directly to students via email")
    
    # Create tabs within the Email tab to differentiate between essay types
    email_tab1, email_tab2 = st.tabs(["📝 Essay Grading Results", "🧠 Model Essay Comparison Results"])
    
    # Email tab for regular essay grading results
    with email_tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card-header"><h3>Compose Email</h3></div>', unsafe_allow_html=True)
            
            # Email configuration
            email_sender = st.text_input("Sender Email", placeholder="your-email@school.edu", key="sender_email1")
            email_password = st.text_input("Email Password", type="password", placeholder="Enter email password", key="password1")
            
            # Student information
            student_name = st.text_input("Student Name", placeholder="John Doe", key="student_name1")
            student_email = st.text_input("Student Email", placeholder="student@example.com", key="student_email1")
            
            # Email content
            st.subheader("Email Content")
            email_subject = st.text_input("Subject", placeholder="Your Essay Results", value="Your Essay Results", key="subject1")
            
            # Select from grading history
            if st.session_state.grading_history:
                selected_essay_index = st.selectbox(
                    "Select Essay from History",
                    range(len(st.session_state.grading_history)),
                    format_func=lambda i: f"{st.session_state.grading_history[i]['date']} - Score: {st.session_state.grading_history[i]['score']}"
                )
                selected_essay = st.session_state.grading_history[selected_essay_index]
                
                # Display selected essay info
                st.info(f"Selected: Essay Set {selected_essay['essay_set']} - Score {selected_essay['score']}")
            else:
                st.warning("No graded essays in history. Grade an essay first.")
                selected_essay = None
            
            # Custom message
            email_message = st.text_area(
                "Custom Message (Optional)",
                placeholder="Add any additional comments for the student...",
                height=150,
                value="Here are your essay results and feedback. Please review and let me know if you have any questions.",
                key="message1"
            )
            
            # Include feedback options
            include_score = st.checkbox("Include Score", value=True, key="include_score1")
            include_feedback = st.checkbox("Include AI Feedback", value=True, key="include_feedback1")
            include_report = st.checkbox("Attach Full Report", value=True, key="include_report1")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card-header"><h3>Email Preview & Send</h3></div>', unsafe_allow_html=True)
            
            if selected_essay:
                st.markdown("### Preview")
                
                email_preview = f"""
                <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px;
                background-color: {'#2D2D44' if st.session_state.dark_mode else '#f9f9f9'};
                color: {'#FFFFFF' if st.session_state.dark_mode else '#000000'};">

                    <h3>To: {student_email}</h3>
                    <h3>Subject: {email_subject}</h3>
                    <hr>
                    <p>Dear {student_name},</p>
                    <p>{email_message}</p>
                    
                    {"<h4>Essay Results:</h4>" if include_score else ""}
                    {"<ul>" if include_score else ""}
                    {"<li><strong>Score:</strong> " + str(selected_essay['score']) + "</li>" if include_score else ""}
                    {"<li><strong>Word Count:</strong> " + str(selected_essay['word_count']) + "</li>" if include_score else ""}
                    {"<li><strong>Date Graded:</strong> " + selected_essay['date'] + "</li>" if include_score else ""}
                    {"</ul>" if include_score else ""}
                    
                    {"<h4>AI Feedback:</h4>" if include_feedback else ""}
                    {"<p>" + (selected_essay.get('ai_feedback', 'Feedback not available') if 'ai_feedback' in selected_essay else 'Feedback not available') + "</p>" if include_feedback else ""}
                    
                    {"<p><em>Full report attached</em></p>" if include_report else ""}
                    
                    <p>Best regards,<br>
                    Your Instructor</p>
                </div>
                """
                
                st.markdown(email_preview, unsafe_allow_html=True)
                
                send_button = st.button("📧 Send Email", use_container_width=True, key="send_button1")
                
                if send_button:
                    if email_sender and email_password and student_email:
                        try:
                            with st.spinner("Sending email..."):
                                # Create email
                                msg = MIMEMultipart()
                                msg['From'] = email_sender
                                msg['To'] = student_email
                                msg['Subject'] = email_subject
                                
                                # Email body
                                body = f"""Dear {student_name},

{email_message}

"""
                                if include_score:
                                    body += f"""Essay Results:
- Score: {selected_essay['score']}
- Word Count: {selected_essay['word_count']}
- Date Graded: {selected_essay['date']}

"""
                                
                                if include_feedback and 'ai_feedback' in selected_essay:
                                    body += f"""AI Feedback:
{selected_essay.get('ai_feedback', 'Feedback not available')}

"""
                                
                                body += """Best regards,
Your Instructor"""
                                
                                msg.attach(MIMEText(body, 'plain'))
                                
                                # Attach report if selected
                                if include_report:
                                    report_content = f"""Essay Analysis Report
                                    
Date: {selected_essay['date']}
Score: {selected_essay['score']}
Word Count: {selected_essay['word_count']}

Essay Set: {selected_essay['essay_set']}
Essay Snippet: {selected_essay['essay_snippet']}

Feedback:
{selected_essay.get('ai_feedback', 'Feedback not available')}
"""
                                    attachment = MIMEText(report_content)
                                    attachment.add_header('Content-Disposition', 'attachment', 
                                                         filename=f"essay_report_{student_name.replace(' ', '_')}.txt")
                                    msg.attach(attachment)
                                
                                # Setup SMTP server
                                server = smtplib.SMTP('smtp.gmail.com', 587)
                                server.starttls()
                                server.login(email_sender, email_password)
                                
                                # Send email
                                server.send_message(msg)
                                server.quit()
                                
                                st.success("✅ Email sent successfully!")
                                
                                # Log the email
                                if 'email_history' not in st.session_state:
                                    st.session_state.email_history = []
                                
                                st.session_state.email_history.append({
                                    "date": datetime.now().strftime("%m/%d/%Y %H:%M"),
                                    "student": student_name,
                                    "email": student_email,
                                    "essay_set": selected_essay['essay_set'],
                                    "score": selected_essay['score'],
                                    "type": "Essay Grading"
                                })
                                
                        except Exception as e:
                            st.error(f"Failed to send email: {str(e)}")
                            st.info("For Gmail, make sure to enable 'Allow less secure apps' in your account settings or use an App Password.")
                    else:
                        st.warning("Please fill in all required email fields.")
            else:
                st.info("Please select a graded essay from history to send.")
    
    # NEW EMAIL TAB FOR MODEL ESSAY COMPARISON
    with email_tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card-header"><h3>Compose Comparison Email</h3></div>', unsafe_allow_html=True)
            
            # Email configuration
            comp_email_sender = st.text_input("Sender Email", placeholder="your-email@school.edu", key="sender_email2")
            comp_email_password = st.text_input("Email Password", type="password", placeholder="Enter email password", key="password2")
            
            # Student information
            comp_student_name = st.text_input("Student Name", placeholder="John Doe", key="student_name2")
            comp_student_email = st.text_input("Student Email", placeholder="student@example.com", key="student_email2")
            
            # Email content
            st.subheader("Email Content")
            comp_email_subject = st.text_input("Subject", placeholder="Your Essay Comparison Results", value="Your Essay Comparison Results", key="subject2")
            
            # Select from model comparison history
            if 'model_answer_history' in st.session_state and st.session_state.model_answer_history:
                selected_comp_index = st.selectbox(
                    "Select Comparison from History",
                    range(len(st.session_state.model_answer_history)),
                    format_func=lambda i: f"{st.session_state.model_answer_history[i]['date']} - Score: {st.session_state.model_answer_history[i]['final_grade']}/{st.session_state.model_answer_history[i]['max_marks']}"
                )
                selected_comp = st.session_state.model_answer_history[selected_comp_index]
                
                # Display selected comparison info
                st.info(f"Selected: {selected_comp['question'][:30]}... - Score {selected_comp['final_grade']}/{selected_comp['max_marks']}")
            else:
                st.warning("No model comparisons in history. Compare an essay first.")
                selected_comp = None
            
            # Custom message
            comp_email_message = st.text_area(
                "Custom Message (Optional)",
                placeholder="Add any additional comments for the student...",
                height=150,
                value="Here are your essay comparison results and feedback. Please review and let me know if you have any questions.",
                key="message2"
            )
            
            # Include feedback options
            include_comp_score = st.checkbox("Include Score", value=True, key="include_score2")
            include_similarity = st.checkbox("Include Similarity Score", value=True, key="include_similarity")
            include_comp_feedback = st.checkbox("Include AI Feedback", value=True, key="include_feedback2")
            include_teacher_comments = st.checkbox("Include Teacher Comments", value=True, key="include_teacher")
            include_comp_report = st.checkbox("Attach Full Report", value=True, key="include_report2")
            include_model_answer = st.checkbox("Include Model Answer", value=False, key="include_model")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card-header"><h3>Email Preview & Send</h3></div>', unsafe_allow_html=True)
            
            if selected_comp:
                st.markdown("### Preview")
                
                comp_email_preview = f"""
                <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px;
                background-color: {'#2D2D44' if st.session_state.dark_mode else '#f9f9f9'};
                color: {'#FFFFFF' if st.session_state.dark_mode else '#000000'};">

                    <h3>To: {comp_student_email}</h3>
                    <h3>Subject: {comp_email_subject}</h3>
                    <hr>
                    <p>Dear {comp_student_name},</p>
                    <p>{comp_email_message}</p>
                    
                    <h4>Essay Topic:</h4>
                    <p>{selected_comp['question']}</p>
                    
                    {"<h4>Comparison Results:</h4>" if include_comp_score or include_similarity else ""}
                    {"<ul>" if include_comp_score or include_similarity else ""}
                    {"<li><strong>Final Score:</strong> " + str(selected_comp['final_grade']) + "/" + str(selected_comp['max_marks']) + "</li>" if include_comp_score else ""}
                    {"<li><strong>Similarity to Model Essay:</strong> " + f"{selected_comp['similarity_score']:.2f}" + "</li>" if include_similarity else ""}
                    {"<li><strong>Date Assessed:</strong> " + selected_comp['date'] + "</li>" if include_comp_score or include_similarity else ""}
                    {"</ul>" if include_comp_score or include_similarity else ""}
                    
                    {"<h4>AI Feedback:</h4>" if include_comp_feedback else ""}
                    {"<p>" + selected_comp.get('ai_feedback', 'Feedback not available') + "</p>" if include_comp_feedback else ""}
                    
                    {"<h4>Teacher Comments:</h4>" if include_teacher_comments and selected_comp.get('teacher_comment') else ""}
                    {"<p>" + selected_comp.get('teacher_comment', '') + "</p>" if include_teacher_comments and selected_comp.get('teacher_comment') else ""}
                    
                    {"<h4>Model Answer:</h4>" if include_model_answer else ""}
                    {"<p><em>The model answer is included in the attached report.</em></p>" if include_model_answer and include_comp_report else ""}
                    {"<p>" + selected_comp.get('model_answer', '')[:300] + "...</p>" if include_model_answer and not include_comp_report else ""}
                    
                    {"<p><em>Full report attached</em></p>" if include_comp_report else ""}
                    
                    <p>Best regards,<br>
                    Your Instructor</p>
                </div>
                """
                
                st.markdown(comp_email_preview, unsafe_allow_html=True)
                
                send_comp_button = st.button("📧 Send Email", use_container_width=True, key="send_button2")
                
                if send_comp_button:
                    if comp_email_sender and comp_email_password and comp_student_email:
                        try:
                            with st.spinner("Sending email..."):
                                # Create email
                                comp_msg = MIMEMultipart()
                                comp_msg['From'] = comp_email_sender
                                comp_msg['To'] = comp_student_email
                                comp_msg['Subject'] = comp_email_subject
                                
                                # Email body
                                comp_body = f"""Dear {comp_student_name},

{comp_email_message}

Essay Topic: {selected_comp['question']}

"""
                                if include_comp_score or include_similarity:
                                    comp_body += f"""Comparison Results:
"""
                                    if include_comp_score:
                                        comp_body += f"""- Final Score: {selected_comp['final_grade']}/{selected_comp['max_marks']}
"""
                                    if include_similarity:
                                        comp_body += f"""- Similarity to Model Essay: {selected_comp['similarity_score']:.2f}
"""
                                    comp_body += f"""- Date Assessed: {selected_comp['date']}

"""
                                
                                if include_comp_feedback:
                                    comp_body += f"""AI Feedback:
{selected_comp.get('ai_feedback', 'Feedback not available')}

"""
                                
                                if include_teacher_comments and selected_comp.get('teacher_comment'):
                                    comp_body += f"""Teacher Comments:
{selected_comp.get('teacher_comment', '')}

"""
                                
                                if include_model_answer and not include_comp_report:
                                    comp_body += f"""Model Answer (excerpt):
{selected_comp.get('model_answer', '')[:300]}...

"""
                                
                                comp_body += """Best regards,
Your Instructor"""
                                
                                comp_msg.attach(MIMEText(comp_body, 'plain'))
                                
                                # Attach report if selected
                                if include_comp_report:
                                    comp_report_content = f"""Essay Comparison Report

Date: {selected_comp['date']}
Topic: {selected_comp['question']}
Final Score: {selected_comp['final_grade']}/{selected_comp['max_marks']}
Similarity Score: {selected_comp['similarity_score']:.2f}

Student Essay:
{selected_comp.get('student_answer', 'Not available')}

"""
                                    if include_model_answer:
                                        comp_report_content += f"""
Model Answer:
{selected_comp.get('model_answer', 'Not available')}

"""
                                    
                                    comp_report_content += f"""
AI Feedback:
{selected_comp.get('ai_feedback', 'Not available')}

"""
                                    
                                    if selected_comp.get('teacher_comment'):
                                        comp_report_content += f"""
Teacher Comments:
{selected_comp.get('teacher_comment', '')}
"""
                                    
                                    comp_attachment = MIMEText(comp_report_content)
                                    comp_attachment.add_header('Content-Disposition', 'attachment', 
                                                         filename=f"essay_comparison_{comp_student_name.replace(' ', '_')}.txt")
                                    comp_msg.attach(comp_attachment)
                                
                                # Setup SMTP server
                                server = smtplib.SMTP('smtp.gmail.com', 587)
                                server.starttls()
                                server.login(comp_email_sender, comp_email_password)
                                
                                # Send email
                                server.send_message(comp_msg)
                                server.quit()
                                
                                st.success("✅ Email sent successfully!")
                                
                                # Log the email
                                if 'email_history' not in st.session_state:
                                    st.session_state.email_history = []
                                
                                st.session_state.email_history.append({
                                    "date": datetime.now().strftime("%m/%d/%Y %H:%M"),
                                    "student": comp_student_name,
                                    "email": comp_student_email,
                                    "essay_topic": selected_comp['question'][:30] + "...",
                                    "score": f"{selected_comp['final_grade']}/{selected_comp['max_marks']}",
                                    "type": "Model Comparison"
                                })
                                
                        except Exception as e:
                            st.error(f"Failed to send email: {str(e)}")
                            st.info("For Gmail, make sure to enable 'Allow less secure apps' in your account settings or use an App Password.")
                    else:
                        st.warning("Please fill in all required email fields.")
            else:
                st.info("Please select a model comparison from history to send.")
    
    # Email history display section
    st.markdown("### Email History")
    
    # Custom CSS for the history section
    st.markdown("""
    <style>
    .history-item {
        background-color: black;
        color: white;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display email history if available
    if 'email_history' in st.session_state and st.session_state.email_history:
        for i, item in enumerate(st.session_state.email_history):
            email_type = item.get('type', 'Essay Grading')  # Default to Essay Grading for backward compatibility
            
            if email_type == "Essay Grading":
                st.markdown(f"""
                <div class="history-item">
                    <strong>{item['date']}</strong> - {email_type} - Sent to {item['student']} ({item['email']})<br>
                    Score: {item['score']} - Essay Set: {item.get('essay_set', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
            else:  # Model Comparison
                st.markdown(f"""
                <div class="history-item">
                    <strong>{item['date']}</strong> - {email_type} - Sent to {item['student']} ({item['email']})<br>
                    Score: {item['score']} - Topic: {item.get('essay_topic', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No emails have been sent yet.")