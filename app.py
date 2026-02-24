import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# --- ส่วนของการตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Gear Processing Time Predictor", layout="wide")

st.title("⚙️ PARADOX: Gear Processing Time Predictor")
st.markdown("""
Gear production time prediction application using a machine learning model trained on a 700-row dataset, designed to provide users with accurate and efficient production time estimates.
""")

# --- ส่วนจำลองการโหลดโมเดล ---
# หมายเหตุ: ในการใช้งานจริง พี่ต้องเทรนโมเดลแล้วใช้ joblib.dump(model, 'rf_model.pkl')
# ในตัวอย่างนี้ผมขอสมมติรายการ Operator และ Work Number ขึ้นมาตามที่พี่เคยให้ข้อมูลไว้ครับ

# รายการตัวเลือก (พี่สามารถแก้ไขให้ตรงกับข้อมูลจริงได้เลย)
work_numbers = ['100', '110', '80', '90', '70', '120', '30', '140', '130', '150', '10', '40']
operators = ['Operator_3', 'Operator_6', 'Operator_1', 'Operator_5', 'Operator_4', 'Operator_10', 'Operator_2', 'Operator_8', 'Operator_7', 'Operator_9']

# --- สร้าง Sidebar สำหรับ Input ---
st.sidebar.header("🛠️ Input Parameters")

def user_input_features():
    # 1. ข้อมูลตัวเลข
    date = st.sidebar.date_input("Start Production Date")
    time = st.sidebar.time_input("Start Production Time")
    out_qty = st.sidebar.number_input("Output Quantity", min_value=1, value=10)
    module = st.sidebar.number_input("Module (m)", min_value=0.1, value=2.0, step=0.1)
    teeth = st.sidebar.number_input("Number of Teeth", min_value=1, value=40)
    thickness = st.sidebar.number_input("Thickness (mm)", min_value=1.0, value=50.0)
    continuous = st.sidebar.selectbox("Continuous", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # 2. ข้อมูลกลุ่ม (Dropdown)
    work_no = st.sidebar.selectbox("Work Number", options=work_numbers)
    operator = st.sidebar.selectbox("Operator Name", options=operators)

    # 'work_no_100', 'work_no_110',
    # 'work_no_120', 'work_no_130', 'work_no_140', 'work_no_150',
    # 'work_no_30', 'work_no_40', 'work_no_70', 'work_no_80', 'work_no_90'
    # 'OperatorEng_Operator_10', 'OperatorEng_Operator_2',
    # 'OperatorEng_Operator_3', 'OperatorEng_Operator_4',
    # 'OperatorEng_Operator_5', 'OperatorEng_Operator_6',
    # 'OperatorEng_Operator_7', 'OperatorEng_Operator_8',
    # 'OperatorEng_Operator_9'

    data = {
        'work_no_100': 1 if work_no == '100' else 0,
        'work_no_110': 1 if work_no == '110' else 0,
        'work_no_120': 1 if work_no == '120' else 0,
        'work_no_130': 1 if work_no == '130' else 0,
        'work_no_140': 1 if work_no == '140' else 0,
        'work_no_150': 1 if work_no == '150' else 0,
        'work_no_30': 1 if work_no == '30' else 0,
        'work_no_40': 1 if work_no == '40' else 0,
        'work_no_70': 1 if work_no == '70' else 0,
        'work_no_80': 1 if work_no == '80' else 0,
        'work_no_90': 1 if work_no == '90' else 0,
        'OperatorEng_Operator_10': 1 if operator == 'Operator_10' else 0,
        'OperatorEng_Operator_2': 1 if operator == 'Operator_2' else 0,
        'OperatorEng_Operator_3': 1 if operator == 'Operator_3' else 0,
        'OperatorEng_Operator_4': 1 if operator == 'Operator_4' else 0,
        'OperatorEng_Operator_5': 1 if operator == 'Operator_5' else 0,
        'OperatorEng_Operator_6': 1 if operator == 'Operator_6' else 0,
        'OperatorEng_Operator_7': 1 if operator == 'Operator_7' else 0,
        'OperatorEng_Operator_8': 1 if operator == 'Operator_8' else 0,
        'OperatorEng_Operator_9': 1 if operator == 'Operator_9' else 0,
        'output quantity': out_qty,
        'module': module,
        'number of teeth': teeth,
        'thickness': thickness,
        'Continuous': continuous,
    }

    # 'output quantity','module', 'number of teeth', 'thickness', 'Continuous'

    return pd.DataFrame(data, index=['data']), work_no, operator, date, time

input_df, work_no, operator, date, time = user_input_features()

# --- ส่วนแสดงผล ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Input Summary")
    st.write(f"📅 **Date:** {date}")
    st.write(f"🕰️ **Time:** {time}")
    st.info(f"**Output Quantity:** {input_df['output quantity'][0]}")
    st.info(f"**Module:** {input_df['module'][0]}")
    st.info(f"**Number of Teeth:** {input_df['number of teeth'][0]}")
    st.info(f"**Thickness:** {input_df['thickness'][0]}")
    st.info(f"**Continuous:** {'Yes' if input_df['Continuous'][0] == 1 else 'No'}")
    st.info(f"**Work Number:** {work_no}")
    st.info(f"**Operator Engineer:** {operator}")

with col2:
    st.subheader("🔮 Prediction Result")
    
    # พยายามโหลดโมเดล
    try:
        model = joblib.load('xgb_model.pkl') # ต้องมีไฟล์นี้ในโฟลเดอร์เดียวกัน
        prediction = model.predict(input_df)
        
        # แสดงผลเวลาที่ทำนายได้
        st.success(f"### Predicted Processing Time: {prediction[0]:.2f} Minutes")
        
        # คำนวณเวลาเฉลี่ยต่อชิ้น
        time_per_unit = prediction[0] / input_df['output quantity'][0]
        st.write(f"⏱️ Estimated time per unit: **{time_per_unit:.2f} minutes/piece**")
        finish_time = datetime.combine(date, time) + timedelta(minutes=float(prediction[0]))
        st.write(f"✅ Production will finish at: **{finish_time.strftime('%Y-%m-%d %H:%M:%S')}**")

    except:
        st.warning("⚠️ Fix the error")

# --- ส่วนการแสดงความสัมพันธ์ (Option) ---
st.divider()
st.caption("PARADOX is a project by Tanaphon Loesphuwiwat that currently in development")
