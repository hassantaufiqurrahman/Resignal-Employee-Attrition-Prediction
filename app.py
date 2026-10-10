import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from src.config import INPUT_CONSTRAINTS, MODELS_DIR
import os
from pycaret.classification import load_model, predict_model

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Resignal | Attrition Prediction App",
    page_icon="🏢",
    layout="centered",
)

# Mengambil URL dari Environment Variable Docker
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"
API_URL = os.getenv("API_URL", DEFAULT_API_URL)

if not API_URL.endswith("/predict"):
    API_URL = API_URL.rstrip("/") + "/predict"

# Inisialisasi Session State
if "page" not in st.session_state:
  st.session_state.page = 1

if "form_data" not in st.session_state:
  st.session_state.form_data = {}

# 2. Header & Progress Bar
with st.container():
    st.markdown(
        """
        <div style="padding: 20px 0; text-align: left;">
            <h1 style="color: #2b6cb0; margin-bottom: 0px;"> Resignal: Employee Attrition Prediction</h1>
            <p style="font-size: 16px; color: #4a5568; margin-top: 8px;">
                <b>Attrition Prediction System</b> — Prevent potential resignations early and protect your company talent asset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Menampilkan informasi pembuat dengan badge yang rapi
    st.info(
        "💡 **Project Info:** Developed by **Hassan Taufiqurrahman** "
        "as the Final Project for **Dibimbing Data Science & AI Machine Learning Batch 42**."
    )

total_pages = 6
current_page = st.session_state.page

progress_val = current_page / total_pages
st.progress(progress_val)
st.write(f"**Page {current_page} from {total_pages} Pages**")

st.markdown("---")


# Helper Function untuk Menentukan Index Default Selectbox
def get_choice_index(key, default_val):
  val = st.session_state.form_data.get(key, default_val)
  choices = INPUT_CONSTRAINTS[key]["choices"]
  return choices.index(val) if val in choices else 0


# Helper function untuk Feature Engineering Dalam Permutation Importance
def create_features_frontend(df):
  df = df.copy()

  # Financial Ratios
  df["Income_Per_Year_At_Company"] = (
    df["Monthly_Income"] * 12 / (df["Years_At_Company"] + 1)
  )
  df["Income_Per_Project"] = df["Monthly_Income"] / (
    df["Number_Of_Projects"]
  )
  df["Income_Per_Age"] = df["Monthly_Income"] / df["Age"]
  
  # Career Ratios
  df["Experience_Ratio"] = df["Years_In_Current_Role"] / (
    df["Years_At_Company"] + 1
  )
  
  # Aggregasi Satisfaction
  sat_cols = [
    "Job_Satisfaction",
    "Work_Life_Balance",
    "Environment_Satisfaction",
    "Relationship_Satisfaction",
  ]
  df["Avg_Satisfaction"] = df[sat_cols].mean(axis=1)
  
  # Workload & Performance
  df["Workload_Index"] = df["Number_Of_Projects"] * df["Team_Size"]
  df["Projects_Per_Team_Member"] = df["Number_Of_Projects"] / (
  df["Team_Size"]
  )
  
  # Commute
  df["Commute_Speed_Index"] = df["Distance_From_Home"] / (
    df["Commute_Time_Minutes"]
  )
  
  return df


# PAGE 1: Profil dan Demografi Karyawan
if st.session_state.page == 1:
    st.subheader("🧑‍💼 Profile and Demographics")

    col1, col2 = st.columns(2)

    with col1:
        employee_name = st.text_input(
            "Employee Name",
            value=st.session_state.form_data.get("Employee_Name", ""),
            placeholder="Please insert employee name...",
        )

        c_age = INPUT_CONSTRAINTS["Age"]
        age = st.number_input(
            "Age",
            min_value=int(c_age["min_value"]),
            max_value=int(c_age["max_value"]),
            value=st.session_state.form_data.get("Age", None),
            placeholder="Please insert age...",
        )

        gender = st.selectbox(
            "Gender",
            options=["Select Gender"] + list(INPUT_CONSTRAINTS["Gender"]["choices"]),
            index=0 if not st.session_state.form_data.get("Gender") else get_choice_index("Gender", st.session_state.form_data.get("Gender")) + 1,
        )

        marital_status = st.selectbox(
            "Marital Status",
            options=["Select Marital Status"] + list(INPUT_CONSTRAINTS["Marital_Status"]["choices"]),
            index=0 if not st.session_state.form_data.get("Marital_Status") else get_choice_index("Marital_Status", st.session_state.form_data.get("Marital_Status")) + 1,
        )

    with col2:
        education_level = st.selectbox(
            "Education Level",
            options=["Select Education Level"] + list(INPUT_CONSTRAINTS["Education_Level"]["choices"]),
            index=0 if not st.session_state.form_data.get("Education_Level") else get_choice_index("Education_Level", st.session_state.form_data.get("Education_Level")) + 1,
        )

        c_dist = INPUT_CONSTRAINTS["Distance_From_Home"]
        distance_from_home = st.number_input(
            "Distance From Home (KM)",
            min_value=int(c_dist["min_value"]),
            max_value=int(c_dist["max_value"]),
            value=st.session_state.form_data.get("Distance_From_Home", None),
            placeholder="Please insert distance from home...",
        )

        c_commute = INPUT_CONSTRAINTS["Commute_Time_Minutes"]
        commute_time_minutes = st.number_input(
            "Commute Time To Office (Minute)",
            min_value=int(c_commute["min_value"]),
            max_value=int(c_commute["max_value"]),
            value=st.session_state.form_data.get("Commute_Time_Minutes", None),
            placeholder="Please insert commute time...",
        )

    col_space, col_next = st.columns(2)
    
    with col_next:
        if st.button("Next ➡️", use_container_width=True):
            if age is None or distance_from_home is None or commute_time_minutes is None or gender == "Select Gender" or marital_status == "Select Marital Status" or education_level == "Select Education Level":
                st.warning("⚠️ Please fill in all required fields before proceeding!")
            else:
                st.session_state.form_data.update({
                    "Employee_Name": employee_name if employee_name else "Anonymous",
                    "Age": age,
                    "Gender": gender,
                    "Marital_Status": marital_status,
                    "Education_Level": education_level,
                    "Distance_From_Home": distance_from_home,
                    "Commute_Time_Minutes": commute_time_minutes,
                })
                st.session_state.page = 2
                st.rerun()


# PAGE 2: Struktur Pekerjaan Karyawan
elif st.session_state.page == 2:
    st.subheader("🗃️ Job Structure")

    col1, col2 = st.columns(2)

    with col1:
        department = st.selectbox(
            "Department",
            options=["Select Department"] + list(INPUT_CONSTRAINTS["Department"]["choices"]),
            index=0 if not st.session_state.form_data.get("Department") else get_choice_index("Department", st.session_state.form_data.get("Department")) + 1,
        )

        job_role = st.selectbox(
            "Job Role",
            options=["Select Job Role"] + list(INPUT_CONSTRAINTS["Job_Role"]["choices"]),
            index=0 if not st.session_state.form_data.get("Job_Role") else get_choice_index("Job_Role", st.session_state.form_data.get("Job_Role")) + 1,
        )

        employment_type = st.selectbox(
            "Employment Type",
            options=["Select Employment Type"] + list(INPUT_CONSTRAINTS["Employment_Type"]["choices"]),
            index=0 if not st.session_state.form_data.get("Employment_Type") else get_choice_index("Employment_Type", st.session_state.form_data.get("Employment_Type")) + 1,
        )

    with col2:
        job_level = st.selectbox(
            "Job Level",
            options=["Select Job Level"] + list(INPUT_CONSTRAINTS["Job_Level"]["choices"]),
            index=0 if not st.session_state.form_data.get("Job_Level") else get_choice_index("Job_Level", st.session_state.form_data.get("Job_Level")) + 1,
        )

        c_team = INPUT_CONSTRAINTS["Team_Size"]
        team_size = st.number_input(
            "Team Size",
            min_value=int(c_team["min_value"]),
            max_value=int(c_team["max_value"]),
            value=st.session_state.form_data.get("Team_Size", None),
            placeholder="Please insert team size...",
        )

        c_proj = INPUT_CONSTRAINTS["Number_Of_Projects"]
        number_of_projects = st.number_input(
            "Total Project",
            min_value=int(c_proj["min_value"]),
            max_value=int(c_proj["max_value"]),
            value=st.session_state.form_data.get("Number_Of_Projects", None),
            placeholder="Please insert total projects...",
        )

    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.page = 1
            st.rerun()
    with col_next:
        if st.button("Next ➡️", use_container_width=True):
            if team_size is None or number_of_projects is None or department == "Select Department" or job_role == "Select Job Role" or employment_type == "Select Employment Type" or job_level == "Select Job Level":
                st.warning("⚠️ Please fill in all required fields before proceeding!")
            else:
                st.session_state.form_data.update({
                    "Department": department,
                    "Job_Role": job_role,
                    "Employment_Type": employment_type,
                    "Job_Level": job_level,
                    "Team_Size": team_size,
                    "Number_Of_Projects": number_of_projects,
                })
                st.session_state.page = 3
                st.rerun()

# PAGE 3: Finansial dan Karier Karyawan
elif st.session_state.page == 3:
  st.subheader("💵 Finances and Career")

  col1, col2 = st.columns(2)

  with col1:
      c_inc = INPUT_CONSTRAINTS["Monthly_Income"]
      monthly_income = st.number_input(
          "Monthly Salary (Dollar)",
          min_value=int(c_inc["min_value"]),
          max_value=int(c_inc["max_value"]),
          value=st.session_state.form_data.get("Monthly_Income", None),
          placeholder="Please insert monthly salary...",
      )

      c_hike = INPUT_CONSTRAINTS["Salary_Hike_Percent"]
      salary_hike_percent = st.number_input(
          "Salary Hike Percent",
          min_value=float(c_hike["min_value"]),
          max_value=float(c_hike["max_value"]),
          value=st.session_state.form_data.get("Salary_Hike_Percent", None),
          placeholder="Please insert salary hike percent...",
      )

      c_yac = INPUT_CONSTRAINTS["Years_At_Company"]
      years_at_company = st.number_input(
          "Years At Company",
          min_value=int(c_yac["min_value"]),
          max_value=int(c_yac["max_value"]),
          value=st.session_state.form_data.get("Years_At_Company", None),
          placeholder="Please insert years at company...",
      ) 

  with col2:
      c_ycr = INPUT_CONSTRAINTS["Years_In_Current_Role"]
      years_in_current_role = st.number_input(
          "Years In Current Role",
          min_value=int(c_ycr["min_value"]),
          max_value=int(c_ycr["max_value"]),
          value=st.session_state.form_data.get("Years_In_Current_Role", None),
          placeholder="Please insert years in role...",
      )

      promotion_last_5_years = st.selectbox(
          "Promotion In Last 5 Years",
          options=["Select Option"] + list(INPUT_CONSTRAINTS["Promotion_Last_5_Years"]["choices"]),
          index=0 if not st.session_state.form_data.get("Promotion_Last_5_Years") else get_choice_index("Promotion_Last_5_Years", st.session_state.form_data.get("Promotion_Last_5_Years")) + 1,
      )

      stock_option = st.selectbox(
          "Stock Option",
          options=["Select Option"] + list(INPUT_CONSTRAINTS["Stock_Option"]["choices"]),
          index=0 if not st.session_state.form_data.get("Stock_Option") else get_choice_index("Stock_Option", st.session_state.form_data.get("Stock_Option")) + 1,
      )

  col_prev, col_next = st.columns([1, 1])
  with col_prev:
    if st.button("⬅️ Back", use_container_width=True):
      st.session_state.page = 2
      st.rerun()
  with col_next:
    if st.button("Next ➡️", use_container_width=True):
      if monthly_income is None or salary_hike_percent is None or years_at_company is None or years_in_current_role is None or promotion_last_5_years == "Select Option" or stock_option == "Select Option":
          st.warning("⚠️ Please fill in all required fields before proceeding!")
      else:
          st.session_state.form_data.update({
              "Monthly_Income": monthly_income,
              "Salary_Hike_Percent": salary_hike_percent,
              "Years_At_Company": years_at_company,
              "Years_In_Current_Role": years_in_current_role,
              "Promotion_Last_5_Years": promotion_last_5_years,
              "Stock_Option": stock_option,
          })
          st.session_state.page = 4
          st.rerun()

# PAGE 4: Sistem Kerja dan Kehadiran Karyawan
elif st.session_state.page == 4:
  st.subheader("🕖 Work and Attendance System")

  col1, col2 = st.columns(2)
  
  with col1:
      overtime = st.selectbox(
          "Overtime",
          options=["Select Option"] + list(INPUT_CONSTRAINTS["Overtime"]["choices"]),
          index=0 if not st.session_state.form_data.get("Overtime") else get_choice_index("Overtime", st.session_state.form_data.get("Overtime")) + 1,
      )

      remote_work = st.selectbox(
          "Work From Anywhere",
          options=["Select Option"] + list(INPUT_CONSTRAINTS["Remote_Work"]["choices"]),
          index=0 if not st.session_state.form_data.get("Remote_Work") else get_choice_index("Remote_Work", st.session_state.form_data.get("Remote_Work")) + 1,
      )

      business_travel = st.selectbox(
          "Business Travel",
          options=["Select Option"] + list(INPUT_CONSTRAINTS["Business_Travel"]["choices"]),
          index=0 if not st.session_state.form_data.get("Business_Travel") else get_choice_index("Business_Travel", st.session_state.form_data.get("Business_Travel")) + 1,
      )

  with col2:
      c_abs = INPUT_CONSTRAINTS["Absence_Days"]
      absence_days = st.number_input(
          "Total Absence (Day)",
          min_value=int(c_abs["min_value"]),
          max_value=int(c_abs["max_value"]),
          value=st.session_state.form_data.get("Absence_Days", None),
          placeholder="Please insert absence days...",
      )

      c_train = INPUT_CONSTRAINTS["Training_Hours_Last_Year"]
      training_hours_last_year = st.number_input(
          "Total Training In Last Year (Hour)",
          max_value=int(c_train["max_value"]),
          min_value=int(c_train["min_value"]),
          value=st.session_state.form_data.get("Training_Hours_Last_Year", None),
          placeholder="Please insert training hours...",
      )

  col_prev, col_next = st.columns([1, 1])
  with col_prev:
    if st.button("⬅️ Back", use_container_width=True):
      st.session_state.page = 3
      st.rerun()
  with col_next:
    if st.button("Next ➡️", use_container_width=True):
      if absence_days is None or training_hours_last_year is None or overtime == "Select Option" or remote_work == "Select Option" or business_travel == "Select Option":
          st.warning("⚠️ Please fill in all required fields before proceeding!")
      else:
          st.session_state.form_data.update({
              "Overtime": overtime,
              "Remote_Work": remote_work,
              "Business_Travel": business_travel,
              "Absence_Days": absence_days,
              "Training_Hours_Last_Year": training_hours_last_year,
          })
          st.session_state.page = 5
          st.rerun()

# PAGE 5: Skor Kepuasan dan Rating Karyawan
elif st.session_state.page == 5:
    st.subheader("🌟 Satisfaction Scores and Ratings")

    def render_slider(label, key):
        choices = INPUT_CONSTRAINTS[key]["choices"]
        min_val, max_val = min(choices), max(choices)
        curr_val = st.session_state.form_data.get(key, min_val)
        return st.slider(
            label, min_value=min_val, max_value=max_val, value=int(curr_val)
        )

    col1, col2 = st.columns(2)

    with col1:
        job_satisfaction = render_slider("Job Satisfaction", "Job_Satisfaction")
        environment_satisfaction = render_slider("Environment Satisfaction", "Environment_Satisfaction")
        relationship_satisfaction = render_slider("Relationship Satisfaction", "Relationship_Satisfaction")

    with col2:
        work_life_balance = render_slider("Work-Life Balance", "Work_Life_Balance")
        performance_rating = render_slider("Employee Performance", "Performance_Rating")
        manager_rating = render_slider("Manager Assessment", "Manager_Rating")

    col_prev, col_submit = st.columns([1, 1])
    with col_prev:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.page = 4
            st.rerun()
    with col_submit:
        if st.button("📄 Predict Result", type="primary", use_container_width=True):
            st.session_state.form_data.update({
                "Job_Satisfaction": job_satisfaction,
                "Work_Life_Balance": work_life_balance,
                "Environment_Satisfaction": environment_satisfaction,
                "Relationship_Satisfaction": relationship_satisfaction,
                "Performance_Rating": performance_rating,
                "Manager_Rating": manager_rating,
            })

            api_payload = {
                k: v for k, v in st.session_state.form_data.items() if k != "Employee_Name"
            }
            emp_name = st.session_state.form_data.get("Employee_Name", "Karyawan")

            try:
                with st.spinner(f"Analyzing the resignation potential for {emp_name}..."):
                    response = requests.post(API_URL, json=api_payload, timeout=5)

                    if response.status_code == 200:
                        st.session_state.api_result = response.json()
                        st.session_state.page = 6
                        st.rerun()
                    else:
                        st.error(
                            f"Failed to process prediction (Code: {response.status_code})"
                        )
                        st.json(response.text)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Failed to connect to FastAPI. Make sure the Uvicorn server is running in the terminal!"
                )

# PAGE 6: Hasil Prediksi (Dedicated Result Page)
elif st.session_state.page == 6:
    st.subheader("🎯 Prediction Employee Attrition Result")

    emp_name = st.session_state.form_data.get("Employee_Name", "Employee")
    res_data = st.session_state.get("api_result", {})

    pred_result = res_data.get("prediction_result")
    pred_prob = res_data.get("resign_probability", 0.0)

    st.markdown(f"### 📊 Employee Attrition Prediction Results: **{emp_name}**")

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        if pred_result == "Resign":
            st.error(f"**Status: {pred_result}** ⚠️")
            st.warning(
                f"**{emp_name}** potentially resigning from the company."
            )
        else:
            st.success(f"**Status: {pred_result}** ✅")
            st.info(f"**{emp_name}** will not resign from the company.")

    with col_res2:
        st.metric(
            label="Potential Resignation Rate",
            value=f"{pred_prob * 100:.1f}%",
        )
        st.progress(float(pred_prob))

    st.markdown("---")
    st.subheader("🔍 Feature Importance")
    st.info(
        "The visualization below displays the key factors driving the model's decisions."
    )

    # Variabel untuk Menyimpan Path Gambar Grafik Sementara untuk PDF
    temp_img_path = "temp_importance_chart.png"
    chart_generated = False
    chart_title_pdf = "Key Driving Factors Analysis"

    # Proses Occulusion Sensitivity
    try:
        @st.cache_resource
        def load_cached_model():
            model_path = os.path.join(MODELS_DIR, "best_model")
            return load_model(model_path)

        loaded_model = load_cached_model()

        input_dict = {
            k: v for k, v in st.session_state.form_data.items() if k != "Employee_Name"
        }
        df_input = pd.DataFrame([input_dict])
        sample_data = create_features_frontend(df_input)

        # Menghitung Probabilitas Dasar
        base_res = predict_model(loaded_model, data=sample_data)
        score_cols = [
            c for c in base_res.columns 
            if "score" in c.lower() or "prob" in c.lower() or "prediction_score" in c.lower()
        ]
        
        if score_cols:
            base_score = base_res[score_cols[0]].iloc[0]
            raw_lbl = base_res['prediction_label'].iloc[0]
            if str(raw_lbl).lower() in ['yes', '1', 'resign']:
                base_prob = float(base_score)
            else:
                base_prob = float(1.0 - base_score)
        else:
            base_prob = pred_prob

        # Menghitung Dampak Feature
        importance_results = []
        for col in input_dict.keys():
            if col not in sample_data.columns:
                continue
                
            original_val = sample_data.loc[0, col]
            temp_df = sample_data.copy()

            if pd.api.types.is_numeric_dtype(temp_df[col]):
                temp_df.loc[0, col] = 0
            else:
                temp_df.loc[0, col] = "Unknown"

            temp_df = create_features_frontend(temp_df)

            try:
                pred_res = predict_model(loaded_model, data=temp_df)
                sc_cols = [
                    c for c in pred_res.columns 
                    if "score" in c.lower() or "prob" in c.lower() or "prediction_score" in c.lower()
                ]
                if sc_cols:
                    new_score = pred_res[sc_cols[0]].iloc[0]
                    new_lbl = pred_res['prediction_label'].iloc[0]
                    if str(new_lbl).lower() in ['yes', '1', 'resign']:
                        new_prob = float(new_score)
                    else:
                        new_prob = float(1.0 - new_score)
                else:
                    new_prob = base_prob

                # Dampak Feature berarah: Positif = mendorong ke Resign dan Negatif = menahan ke Stayed
                impact = base_prob - new_prob
                importance_results.append({
                    "Feature": col,
                    "Nilai_Asli": original_val,
                    "Dampak": impact,
                })
            except Exception:
                continue

        df_importance = pd.DataFrame(importance_results)

        # Membuat Visualisasi Occulusion Sensitivity
        plt.figure(figsize=(10, 5))

        if len(df_importance) > 0:
            if pred_result == "Resign":
                df_filtered = (
                    df_importance[df_importance["Dampak"] > 0]
                    .sort_values(by="Dampak", ascending=True)
                    .tail(5)
                )
                chart_title = "Key Factors Triggering Resignation"
                bar_color = "#e74c3c"
                x_values = df_filtered["Dampak"]
            else:
                df_filtered = df_importance[df_importance["Dampak"] < 0].copy()
                df_filtered["Abs_Dampak"] = df_filtered["Dampak"].abs()
                df_filtered = (
                    df_filtered.sort_values(by="Abs_Dampak", ascending=True)
                    .tail(5)
                )
                chart_title = "Key Factors Triggering Not to Resign"
                bar_color = "#2ecc71"
                x_values = df_filtered["Abs_Dampak"]

            chart_title_pdf = chart_title

            if len(df_filtered) > 0:
                y_labels = [
                    f"{row['Feature']} (Value: {row['Nilai_Asli']})"
                    for _, row in df_filtered.iterrows()
                ]

                plt.barh(
                    y_labels, x_values, color=bar_color, alpha=0.85, edgecolor="black"
                )
                plt.title(chart_title, fontsize=13, fontweight="bold", pad=12)
                plt.xlabel(
                    "Relative Impact Ranking of Factors",
                    fontsize=10,
                    fontweight="bold",
                )
                
                plt.xticks([])
                plt.grid(axis="x", linestyle="--", alpha=0.5)
                plt.tight_layout()

                plt.savefig(temp_img_path, bbox_inches='tight', dpi=300)
                chart_generated = True

                st.pyplot(plt)
                plt.close()
            else:
                st.info(
                "💡 The model assesses this employee profile as highly stable. "
                "No single factor significantly dominates the changes in risk."
                )
        else:
            st.info("No feature importance data is available.")

    except Exception as e:
        st.warning(f"Failed to load the feature importance visualization locally: {e}")

    st.markdown("---")

    # Function untuk Pembuatan PDF
    def generate_pdf_report(name, status, prob, fig_path, section_title):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2b6cb0'),
            spaceAfter=10
        )
        
        normal_bold = ParagraphStyle(
            'NormalBold',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6
        )

        # Header Laporan
        story.append(Paragraph("Resignal: Employee Attrition Prediction Report", title_style))
        story.append(Spacer(1, 5))
        story.append(Paragraph(f"<b>Employee Name:</b> {name}", normal_bold))
        story.append(Paragraph(f"<b>Prediction Status:</b> {status}", normal_bold))
        story.append(Paragraph(f"<b>Resignation Probability:</b> {prob * 100:.1f}%", normal_bold))
        story.append(Spacer(1, 10))
        
        # Menyisipkan Grafik Sesuai Kondisi Dinamis (Resign / Stayed)
        if chart_generated and os.path.exists(fig_path):
            story.append(Paragraph(f"<b>{section_title}</b>", styles['Heading2']))
            story.append(Spacer(1, 6))
            story.append(RLImage(fig_path, width=440, height=220))
            
        doc.build(story)
        buffer.seek(0)
        return buffer

    col_pdf, col_reset, col_back = st.columns([1.2, 1, 1])

    with col_pdf:
        try:
            pdf_data = generate_pdf_report(emp_name, pred_result, pred_prob, temp_img_path, chart_title_pdf)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_data,
                file_name=f"Resignal_Prediction_Report_{emp_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as pdf_err:
            st.error(f"PDF Error: {pdf_err}")

    with col_reset:
        if st.button("🔄 Predict Other Employees", use_container_width=True):
            st.session_state.form_data = {}
            st.session_state.page = 1
            st.rerun()
            
    with col_back:
        if st.button("✏️ Edit Data Input", use_container_width=True):
            st.session_state.page = 5
            st.rerun()