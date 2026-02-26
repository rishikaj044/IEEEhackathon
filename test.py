import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

st.set_page_config(page_title="SamajhAI", layout="wide")

st.title("SamajhAI")
st.markdown("SamajhAI helps you understand your data clearly and effortlessly.")

st.sidebar.title("Control Panel")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
sheet_url = st.sidebar.text_input("Or enter a Google Sheet link")

df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)

elif sheet_url:
    csv_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    df = pd.read_csv(csv_url)

if df is None:
    st.info("Please upload a dataset using the sidebar to continue.")
    st.stop()

section = st.sidebar.radio(
    "Select Section",
    ["Overview", "AI Insights", "Visualizations", "Chat"]
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #0E1117;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #262730;
}
</style>
""", unsafe_allow_html=True)

st.subheader("Dataset Preview")
st.dataframe(df)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "SamajhAI"
    }
)

if section == "Overview":

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric(
        "Numeric Columns",
        len(df.select_dtypes(include=['int64', 'float64']).columns)
    )

elif section == "AI Insights":

    colA, colB = st.columns(2)

    if colA.button("Generate Summary"):

        with st.spinner("Analyzing data..."):

            sample = df.head(20).to_string()

            prompt = f"""
            This is a dataset sample:

            {sample}

            Explain what this dataset represents and identify important patterns.
            """

            response = client.chat.completions.create(
                model="meta-llama/llama-3-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                
            )

            st.subheader("Summary")
            st.write(response.choices[0].message.content)

    if colB.button("Find Insights"):

        with st.spinner("Generating insights..."):

            stats = df.describe().to_string()

            prompt = f"""
            Based on this statistical summary:

            {stats}

            Provide five meaningful and actionable insights.
            """

            response = client.chat.completions.create(
                model="meta-llama/llama-3-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                
            )

            st.subheader("Insights")
            st.write(response.choices[0].message.content)

elif section == "Visualizations":

    st.subheader("Data Visualizations")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart"]
    )

    if chart_type == "Histogram":

        col = st.selectbox("Select Column", numeric_cols)
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":

        col = st.selectbox("Select Column", numeric_cols)
        fig = px.box(df, y=col)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":

        x = st.selectbox("X Axis", numeric_cols)
        y = st.selectbox("Y Axis", numeric_cols)
        fig = px.scatter(df, x=x, y=y)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar Chart":

        cat = st.selectbox("Category", categorical_cols)
        num = st.selectbox("Value", numeric_cols)
        fig = px.bar(df, x=cat, y=num)
        st.plotly_chart(fig, use_container_width=True)

elif section == "Chat":

    st.subheader("Chat with your data")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask anything about your dataset")

    if user_input:

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                data_context = df.head(50).to_string()

                messages = [
                    {
                        "role": "system",
                        "content": f"""
                        You are a professional data analyst.

                        Dataset sample:
                        {data_context}

                        Answer questions clearly.
                        Explain trends, patterns, and meanings.
                        Use simple language when needed.
                        """
                    }
                ]

                messages.extend(st.session_state.chat_history)

                response = client.chat.completions.create(
                    model="meta-llama/llama-3-8b-instruct",
                    messages=messages,
                )

                reply = response.choices[0].message.content

                st.write(reply)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": reply
                })