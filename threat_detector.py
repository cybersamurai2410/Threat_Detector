from yolor import *
import tempfile
import cv2
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import sqlite3


def main():
    st.title('Threat Detector')
    st.sidebar.title('Settings')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{ width: 400px;}
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{ width: 400px; margin-left: -400px}
        </style>
        """,
        unsafe_allow_html=True,
    )

    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Database
    connection = sqlite3.connect("threats.db")
    cursor = connection.cursor()
    database = (connection, cursor)

    try:
        cursor.execute("create table threats (Object_ID integer, Threat text, Location text, Date text, Time text)")
    except:
        print("Database exists")

    # Menu
    st.sidebar.markdown('---')
    app_mode = st.sidebar.selectbox('Select mode',
                                    ['Video Analysis', 'Image Analysis', 'Real Time Analysis', 'Threat History']
                                    )

    if app_mode == 'Real Time Analysis':

        st.markdown('---')
        st.header("Open camera to perform real-time analysis")
        st.markdown('---')

        cam = st.sidebar.button('Open Camera')
        record = st.sidebar.checkbox('Record Video')

        st.sidebar.markdown('---')
        confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
        st.sidebar.markdown('---')

        enable_gpu = st.sidebar.checkbox('Enable GPU')
        custom_classes = st.sidebar.checkbox('Custom Objects')
        assigned_class_id = []
        st.sidebar.markdown('---')

        st_frame = st.empty()
        kpi1, kpi2, kpi3 = st.columns(3)

        if cam:
            # vid = cv2.VideoCapture(0)
            # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # fps = input(vid.get(cv2.CAP_PROP_FPS))
            stop = st.button('Close Camera')

            with kpi1:
                st.markdown("**Frame Rate**")
                kpi1_text = st.markdown("0")
            with kpi2:
                st.markdown("**Tracked Objects**")
                kpi2_text = st.markdown("0")
            with kpi3:
                st.markdown("**Status**")
                kpi3_text = st.markdown("Monitoring footage...")

            # if record:
            #     st.checkbox("Recording", value=True)
            #     codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            #     out = cv2.VideoWriter('record.mp4', codec, fps, (width, height))

            # st.markdown('---')
            # st.header("Location")
            # g = geocoder.ip('me')
            # df = pd.DataFrame(np.array([g.latlng]), columns=['lat', 'lon'])
            # st.map(df)
            # st.markdown('---')

            kpi = (kpi1_text, kpi2_text, kpi3_text)
            load_yolor_camera(enable_gpu, confidence, assigned_class_id, kpi, st_frame, database, stop)
            print("Camera Closed")

    elif app_mode == 'Image Analysis':

        st.markdown('---')
        st.header("Upload image files to perform analysis")
        st.markdown('---')
        save_img = st.sidebar.checkbox('Save Image')
        custom_classes = st.sidebar.checkbox('Custom Objects')
        assigned_class_id = []
        st.sidebar.markdown('---')

        if custom_classes:
            assigned_class = st.sidebar.multiselect('Select Custom Objects', list(names), default='person')
            for each in assigned_class:
                assigned_class_id.append(names.index(each))

        uploaded_files = st.sidebar.file_uploader("Upload multiple files", type=["jpg", "png"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                load_yolor_image(uploaded_file.name, assigned_class_id)
                st.markdown('---')

    elif app_mode == 'Video Analysis':

        st.sidebar.markdown('---')
        confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
        st.sidebar.markdown('---')

        save_img = st.sidebar.checkbox('Save Video')
        enable_gpu = st.sidebar.checkbox('Enable GPU')
        custom_classes = st.sidebar.checkbox('Custom Objects')
        assigned_class_id = []
        st.sidebar.markdown('---')

        if custom_classes:
            assigned_class = st.sidebar.multiselect('Select Custom Objects', list(names), default='person')
            for each in assigned_class:
                assigned_class_id.append(names.index(each))

        video_file_buffer = st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "avi", "asf", "m4v"])
        footage = 'test.mp4'
        tf_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

        # Load input video
        if not video_file_buffer:
            vid = cv2.VideoCapture(footage)
            tf_file.name = footage
            current_vid = open(tf_file.name, 'rb')
            current_bytes = current_vid.read()

            st.sidebar.text('Surveillance Footage')
            st.sidebar.video(current_bytes)
        else:
            tf_file.write(video_file_buffer.read())
            current_vid = open(tf_file.name, 'rb')
            current_bytes = current_vid.read()

            st.sidebar.text('Surveillance Footage')
            st.sidebar.video(current_bytes)

        print(tf_file.name)
        st_frame = st.empty()
        st.sidebar.markdown('---')

        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("**Tracked Objects**")
            kpi2_text = st.markdown("0")
        with kpi3:
            st.markdown("**Status**")
            kpi3_text = st.markdown("Monitoring footage...")

        # st.markdown('---')
        # st.header("Location")
        # g = geocoder.ip('me')
        # df = pd.DataFrame(np.array([g.latlng]), columns=['lat', 'lon'])
        # st.map(df)

        st.markdown('---')
        st.header("Analytics")

        kpi = (kpi1_text, kpi2_text, kpi3_text)
        load_yolor_video(tf_file.name, enable_gpu, confidence, assigned_class_id, kpi, st_frame, database)

        st.text('Footage Processed')
        vid.release()

    if app_mode == 'Threat History':
        st.markdown('---')
        st.header("Threat History")
        data = list()
        for row in cursor.execute("select * from threats"):
            data.append(row)
            print(row)

        df = pd.DataFrame(data, columns=['Object ID', 'Threat Type', 'Location', 'Date', 'Time'])
        st.table(df)

    connection.close()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
