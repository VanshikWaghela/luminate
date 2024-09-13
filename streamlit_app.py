import streamlit as st

def app1():
    import pandas as pd
    import numpy as np
    from career_decision_model import load_model

    # Load the model, scaler, and feature names
    model, scaler, feature_names = load_model()

    # Streamlit app to take inputs and predict
    st.title("Career Path Predictor (ML-based)")

    # User input for the model
    gre_score = st.number_input("GRE Score", min_value=0, max_value=340, step=1)
    test_score_toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, step=1)
    undergraduation_score = st.number_input("Undergraduate GPA", min_value=0.0, max_value=10.0, step=0.1)
    work_ex = st.number_input("Work Experience (years)", min_value=0, max_value=10, step=1)
    papers_published = st.number_input("Number of Papers Published", min_value=0, max_value=10, step=1)
    ranking = st.number_input("University Ranking", min_value=1, max_value=100, step=1)
    technical_skills_score = st.slider("Technical Skills Score", 0.0, 10.0, step=0.1)
    logical_quotient_rating = st.slider("Logical Quotient Rating", 0.0, 10.0, step=0.1)
    hours_working_per_day = st.slider("Hours Working per Day", 0.0, 24.0, step=0.1)

    # Prepare the input data for the model
    input_data = pd.DataFrame({
        'gre_score': [gre_score],
        'test_score_toefl': [test_score_toefl],
        'undergraduation_score': [undergraduation_score],
        'work_ex': [work_ex],
        'papers_published': [papers_published],
        'ranking': [ranking],
        'technical_skills_score': [technical_skills_score],
        'Logical quotient rating': [logical_quotient_rating],
        'Hours working per day': [hours_working_per_day]
    })

    # Ensure that input data has the same features (columns) as the training data
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Predict the outcome
    if st.button("Predict"):
        # Scale the input data if scaler is available
        if scaler:
            input_data_scaled = scaler.transform(input_data)
        else:
            input_data_scaled = input_data

        # Get base prediction probabilities
        base_proba = model.predict_proba(input_data_scaled)[0]

        # Adjust probabilities based on university ranking
        ranking_factor = (ranking - 1) / 99  # 0 for rank 1, 1 for rank 100
        adjustment = 0.2 * ranking_factor  # Max adjustment of 20%
        adjusted_proba = [
            max(0, min(1, base_proba[0] - adjustment)),  # Decrease Job probability
            max(0, min(1, base_proba[1] + adjustment))  # Increase Master's probability
        ]

        # Normalize probabilities to ensure they sum to 1
        total = sum(adjusted_proba)
        adjusted_proba = [p / total for p in adjusted_proba]

        # Determine the prediction based on the highest adjusted probability
        prediction = np.argmax(adjusted_proba)
        if prediction == 1:
            st.write(f"The model suggests you should pursue a **Master's degree** (Confidence: {adjusted_proba[1]:.2f}).")
        else:
            st.write(f"The model suggests you should pursue a **Job** (Confidence: {adjusted_proba[0]:.2f}).")

        st.write(f"Probability of Job: {adjusted_proba[0]:.2f}")
        st.write(f"Probability of Master's: {adjusted_proba[1]:.2f}")

    st.info("Remember, these predictions are based on historical data and should be used as one of many factors in your decision-making process.")
    st.info("To understand how your profile stands out in detail, we recommend consulting professional counselors [here](https://yocket.com).")

def app2():
    import pandas as pd
    import numpy as np

    

    # Function to calculate Normalized University Ranking Score (NURS)
    def calculate_nurs(rank, total_universities):
        return 10 * (rank / total_universities)

    # Function to calculate Composite Score for Master's
    def calculate_composite_score(gpa, work_ex, research_papers, ranking, gre_score, weights):
        gpa_normalized = gpa / 10
        work_ex_normalized = work_ex / 4
        research_papers_normalized = min(research_papers / 10, 1.0) * 10
        nurs = calculate_nurs(ranking, 100)
        gre_normalized = min(gre_score / 340, 1) * 10
        
        composite_score = (weights['gpa'] * gpa_normalized + 
                           weights['work_ex'] * work_ex_normalized + 
                           weights['research_papers'] * research_papers_normalized +
                           weights['nurs'] * nurs + 
                           weights['gre'] * gre_normalized)
        
        return min(composite_score, 10)

    # Function to calculate Job Suitability Score (JSS)
    def calculate_jss(technical_skills, project_experience, soft_skills, learning_adaptability, weights):
        jss_score = (weights['technical_skills'] * technical_skills +
                     weights['project_experience'] * project_experience +
                     weights['soft_skills'] * soft_skills +
                     weights['learning_adaptability'] * learning_adaptability)
        
        return min(jss_score, 10)

    # Streamlit App
    st.title("Masters vs Job Predictor (Score-based)")

    # User input for Master's data
    st.header("Input your Master's data")

    # Input fields for Master's
    gpa_score = st.number_input("GPA (out of 100)", min_value=0.0, max_value=100.0, step=0.1)
    work_ex = st.number_input("Work Experience (Years)", min_value=0.0, max_value=4.0, step=0.1)
    research_papers = st.number_input("Research Papers Published", min_value=0, max_value=10, step=1)
    university_ranking = st.number_input("Target University Ranking", min_value=1, max_value=100, step=1)
    gre_score = st.number_input("GRE Score (out of 340)", min_value=0, max_value=340, step=1)

    # Weights for Composite Score (Masters)
    weights_master = {
        'gpa': 0.4,
        'work_ex': 0.2,
        'research_papers': 0.2,
        'nurs': 0.1,
        'gre': 0.3
    }

    # Calculate Composite Decision Score for Master's
    composite_score = calculate_composite_score(gpa_score, work_ex, research_papers, university_ranking, gre_score, weights_master)

    st.write(f"**Composite Decision Score (Masters):** {composite_score:.2f} / 10")

    # User input for Job data
    st.header("Input your Job data")

    # Input fields for Job
    technical_skills = st.slider("Technical Skills Proficiency", 0.0, 10.0, step=0.1)
    project_experience = st.slider("Project Experience", 0.0, 10.0, step=0.1)
    soft_skills = st.slider("Soft Skills", 0.0, 10.0, step=0.1)
    learning_adaptability = st.slider("Learning and Adaptability", 0.0, 10.0, step=0.1)

    # Weights for Job Suitability Score (JSS)
    weights_jss = {
        'technical_skills': 0.4,
        'project_experience': 0.3,
        'soft_skills': 0.2,
        'learning_adaptability': 0.1
    }

    # Calculate Job Suitability Score (JSS)
    jss_score = calculate_jss(technical_skills, project_experience, soft_skills, learning_adaptability, weights_jss)

    st.write(f"**Job Suitability Score (JSS):** {jss_score:.2f} / 10")

    # Decision suggestion
    st.header("Suggested Path")
    if composite_score == jss_score:
        st.write("You have equal scores for pursuing a **Master's degree** and a **Job**.")
    elif composite_score > jss_score:
        st.write("You have a higher score for pursuing a **Master's degree**.")
    else:
        st.write("You have a higher score for pursuing a **Job**.")

    # Additional context for students
    st.info("**Note:** These scores are just to help you get a brief idea. To understand how your profile stands out in detail, we recommend consulting professional counselors [here](https://yocket.com).")

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the predictor", ["ML-based Predictor", "Score-based Predictor"])
    
    if app_mode == "ML-based Predictor":
        app1()
    elif app_mode == "Score-based Predictor":
        app2()

if __name__ == "__main__":
    main()