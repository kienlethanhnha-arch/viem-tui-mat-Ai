import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. CẤU HÌNH TRANG WEB
st.set_page_config(
    page_title="Dự báo Hoại tử Túi mật",
    page_icon="🏥",
    layout="centered"
)

# 2. TẢI MÔ HÌNH ĐÃ LƯU
@st.cache_resource
def load_model():
    return joblib.load('mo_hinh_du_doan_hoai_tu.pkl')

try:
    model = load_model()
except:
    st.error("⚠️ Không tìm thấy file mô hình. Hãy đảm bảo bạn đã upload file .pkl cùng thư mục!")
    st.stop()

# 3. GIAO DIỆN NHẬP LIỆU
st.title("🏥 Dự báo Hoại tử Túi mật (AI)")
st.markdown("---")
st.info("Công cụ hỗ trợ bác sĩ lâm sàng dự đoán nguy cơ hoại tử túi mật trước mổ.")

# Chia cột để giao diện đẹp hơn
col1, col2 = st.columns(2)

with col1:
    st.header("1. Thông tin chung")
    age = st.number_input("Tuổi", min_value=1, max_value=100, value=50)
    sex = st.selectbox("Giới tính", options=[1, 0], format_func=lambda x: "Nam" if x == 1 else "Nữ")
    bmi = st.number_input("BMI", value=22.0)
    dm = st.selectbox("Đái tháo đường", options=[0, 1], format_func=lambda x: "Không" if x == 0 else "Có")
    hta = st.selectbox("Tăng huyết áp", options=[0, 1], format_func=lambda x: "Không" if x == 0 else "Có")

with col2:
    st.header("2. Lâm sàng & XN")
    fever = st.selectbox("Sốt (>38 độ)", options=[0, 1], format_func=lambda x: "Không" if x == 0 else "Có")
    wbc = st.number_input("Bạch cầu (WBC - G/L)", value=10.0)
    crp = st.number_input("CRP (mg/L)", value=5.0)
    onset_hours = st.number_input("Thời gian đau (giờ)", value=24)

st.markdown("---")
st.header("3. Chẩn đoán hình ảnh (SA/CT)")

col3, col4 = st.columns(2)
with col3:
    wall_thickened = st.selectbox("Dày thành túi mật (SA)", options=[0, 1], format_func=lambda x: "Không" if x == 0 else "Có")
    pericholecystic_fluid = st.selectbox("Dịch quanh túi mật (SA)", options=[0, 1], format_func=lambda x: "Không" if x == 0 else "Có")
    impacted_stone = st.selectbox("Sỏi kẹt cổ (SA)", options=[0, 1], format_func=lambda x: "Không" if x == 0 else "Có")

with col4:
    ct_wall_thickened = st.selectbox("Dày thành (CT Scan)", options=[-1, 0, 1],
                                     format_func=lambda x: "Không chụp CT" if x == -1 else ("Có" if x == 1 else "Không"))
    # Nếu không chụp CT thì gán giá trị NaN hoặc Missing tùy theo cách bạn train model
    # Ở đây tôi để -1 và giả định pipeline của bạn có bước xử lý (như code mẫu trước tôi gửi đã có SimpleImputer fill -1)

# 4. XỬ LÝ DỮ LIỆU ĐẦU VÀO
# Lưu ý: Tên cột phải KHỚP CHÍNH XÁC với lúc huấn luyện
input_data = pd.DataFrame({
    'age': [age], 'sex': [sex], 'bmi': [bmi],
    'dm': [dm], 'hta': [hta], 'heart_disease': [0], 'chronic_kidney': [0], # Gán mặc định nếu không nhập
    'fever': [fever], 'murphy_clinical': [0], # Gán mặc định
    'onset_hours': [onset_hours],
    'heart_rate': [90], 'systolic_bp': [120], 'diastolic_bp': [70], # Gán trung bình
    'wbc': [wbc], 'neutrophil_pct': [70], 'lymphocyte_pct': [20], 'nlr': [3.5],
    'crp': [crp], 'ast': [30], 'alt': [30], 'bilirubin_total': [10], 'creatinine': [80],
    'wall_thickened': [wall_thickened], 'pericholecystic_fluid': [pericholecystic_fluid],
    'impacted_stone': [impacted_stone],
    'gallbladder_distended': [0], 'gas_in_wall': [0], 'murphy_ultrasound': [1],
    'ct_wall_thickened': [np.nan if ct_wall_thickened == -1 else ct_wall_thickened],
    'ct_pericholecystic_fluid': [np.nan] # Giả sử thiếu
})

# 5. DỰ BÁO
if st.button("🔍 PHÂN TÍCH NGAY", use_container_width=True):
    try:
        # Dự báo xác suất
        prob = model.predict_proba(input_data)[0][1]
        percent = prob * 100

        st.markdown("### KẾT QUẢ PHÂN TÍCH:")

        # Thanh hiển thị mức độ nguy cơ
        st.progress(int(percent))

        if percent < 30:
            st.success(f"✅ NGUY CƠ THẤP: {percent:.1f}% - Có thể mổ nội soi chương trình/trì hoãn.")
        elif percent < 70:
            st.warning(f"⚠️ NGUY CƠ TRUNG BÌNH: {percent:.1f}% - Cẩn trọng, chuẩn bị khả năng mổ khó.")
        else:
            st.error(f"🚨 NGUY CƠ CAO HOẠI TỬ: {percent:.1f}% - Cần mổ sớm, chuẩn bị chuyển mổ mở.")

    except Exception as e:
        st.error(f"Có lỗi xảy ra: {e}")
        st.info("Hãy kiểm tra lại số lượng biến đầu vào có khớp với mô hình không.")

# Disclaimer
st.markdown("-----------")
st.caption("Lưu ý: Kết quả chỉ mang tính chất tham khảo hỗ trợ nghiên cứu. Quyết định cuối cùng thuộc về bác sĩ lâm sàng.")
