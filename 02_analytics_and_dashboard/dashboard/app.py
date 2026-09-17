import streamlit as st

st.set_page_config(
    page_title="He Thong Dinh Gia Va Phan Tich Thi Truong Xe Cu",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def render_overview():
    st.header("Tong quan thi truong xe cu")
    st.write("Noi dung dang duoc phat trien")

def render_analytics():
    st.header("Phan tich chuyen sau va ban do nhiet")
    st.write("Noi dung dang duoc phat trien")

def render_valuation():
    st.header("Tham dinh gia va phat hien Deal")
    st.write("Noi dung dang duoc phat trien")

def main():
    pages = {
        "Tong quan": render_overview,
        "Phan tich & Ban do": render_analytics,
        "Dinh gia xe": render_valuation
    }
    selected_page = st.sidebar.radio("Chuc nang", list(pages.keys()))
    pages[selected_page]()

if __name__ == "__main__":
    main()
