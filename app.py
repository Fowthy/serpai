import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import advertools as adv
import time
from pathlib import Path
from textblob import TextBlob

st.title("SERP Analysis")
st.markdown("""
This tool allows you to track and analyze search engine results pages (SERPs) over time for specified queries. 
""")
st.sidebar.header("Settings")

# Aquire API Key and CSE ID
def get_default_credentials():
    api_key_default = st.secrets.get("API_KEY")
    cse_id_default = st.secrets.get("CSE_ID")
    return api_key_default, cse_id_default

# User inputs for API Key, Custom Search Engine ID, and folder path
api_key = st.sidebar.text_input("Enter your API Key", type="password")
cse_id = st.sidebar.text_input("Enter your Custom Search Engine ID")
folder_path = Path(
    st.sidebar.text_input(
        "Enter folder path for saving CSV files (default: ./serpdata/)",
        "./serpdata/",
    )
)
folder_path.mkdir(parents=True, exist_ok=True)

# Use default credentials if inputs are empty
if not api_key or not cse_id:
    api_key_default, cse_id_default = get_default_credentials()
    if api_key_default and cse_id_default:
        api_key = api_key or api_key_default
        cse_id = cse_id or cse_id_default
        st.sidebar.info("Using default API credentials from secrets.")
    else:
        st.sidebar.error("Default API credentials are not set in secrets.")
        st.stop() 


queries_input = st.sidebar.text_area(
    "Enter Queries (separated by commas)", "SEO rank tracking, SERP tracking tools"
)
queries = [q.strip() for q in queries_input.split(",") if q.strip()]
interval = st.sidebar.number_input(
    "Interval between requests (seconds)", min_value=10, value=60
)
iterations = st.sidebar.number_input("Number of iterations", min_value=1, value=15)

# Initialize session state for SERP data (for storing data after app refresh)
if "serp_csv" not in st.session_state:
    st.session_state["serp_csv"] = None

# Function to record SERP data
def record_serp():
    date = datetime.now().strftime("%d%m%Y%H_%M_%S")
    df = adv.serp_goog(q=queries, key=api_key, cx=cse_id)
    df.to_csv(folder_path / f"serp_{date}_scheduled_serp.csv")
    return df

# Start SERP Tracking
if st.sidebar.button("Start SERP Tracking"):
    with st.spinner("Recording SERP data..."):
        collected_dfs = []
        for i in range(iterations):
            st.info(f"Recording iteration {i+1}/{iterations}")
            df = record_serp()
            collected_dfs.append(df)
            if i < iterations - 1:
                time.sleep(interval)
        st.success("SERP tracking completed and saved!")

    # Concatenate collected dfs
    serp_csv = pd.concat(collected_dfs, ignore_index=True)
    serp_csv.drop(columns="Unnamed: 0", inplace=True, errors="ignore")
    serp_csv.set_index("queryTime", inplace=True)
    serp_csv.index = (
        pd.to_datetime(serp_csv.index, errors="coerce").strftime("%Y-%m-%d %H:%M:%S")
    )

    st.session_state["serp_csv"] = serp_csv

uploaded_files = st.file_uploader(
    "Upload your SERP CSV files (multi-file supported)",
    type="csv",
    accept_multiple_files=True,
)
if uploaded_files:
    serp_csvs = [pd.read_csv(file) for file in uploaded_files]
    serp_csv = pd.concat(serp_csvs, ignore_index=True)
    serp_csv.drop(columns="Unnamed: 0", inplace=True, errors="ignore")
    serp_csv.set_index("queryTime", inplace=True)
    serp_csv.index = (
        pd.to_datetime(serp_csv.index, errors="coerce").strftime("%Y-%m-%d %H:%M:%S")
    )
    st.session_state["serp_csv"] = serp_csv

serp_csv = st.session_state.get("serp_csv")

if serp_csv is not None:
    st.dataframe(serp_csv)

    # Download option
    csv = serp_csv.to_csv().encode("utf-8")
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="serp_data.csv",
        mime="text/csv",
    )

    # Sentiment analysis model
    st.subheader("Sentiment Analysis")
    st.write("""
    The sentiment score represents the emotional tone of the titles in the search results, calculated using TextBlob.
    - **Positive Sentiment (0 to 1)**: The title conveys a positive tone.
    - **Negative Sentiment (-1 to 0)**: The title conveys a negative tone.
    - **Neutral Sentiment (Around 0)**: The title is neutral.
    Understanding sentiment can help tailor content strategies and gain insights into market perceptions.
    """)
    serp_csv["sentiment"] = serp_csv["title"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    # Adjust sentiment scores to be positive for plotting
    # Range from 0 to 2
    serp_csv["sentiment_positive"] = serp_csv["sentiment"] + 1

    st.write("#### Sentiment Scores")
    st.dataframe(serp_csv[["title", "sentiment"]])

    # Histogram for sentiment scores
    fig = px.histogram(
        serp_csv, x="sentiment", nbins=20, title="Distribution of Sentiment Scores"
    )
    st.plotly_chart(fig)

    # Scatter Plot
    st.subheader("SERP Tracking Scatter Plot")
    st.write("""
    This scatter plot visualizes the ranking positions of different domains over time. The x-axis represents the domains, and the y-axis shows their ranking positions (lower values mean higher ranks). Use the animation controls to observe changes over time.
    """)
    st.write("##### Filter Results")
    st.write("Use the input below to filter results based on keywords found in the search queries or titles.")
    keyword = st.text_input("Enter a keyword to filter results", "serp")
    filter_field = st.selectbox("Filter results by:", ["Search Terms", "Title", "Both"])

    # Filter
    if filter_field == "Search Terms":
        serp_results = serp_csv[
            serp_csv["searchTerms"].str.contains(keyword, regex=True, case=False)
        ].copy()
    elif filter_field == "Title":
        serp_results = serp_csv[
            serp_csv["title"].str.contains(keyword, regex=True, case=False)
        ].copy()
    else:
        serp_results = serp_csv[
            serp_csv["searchTerms"].str.contains(keyword, regex=True, case=False) |
            serp_csv["title"].str.contains(keyword, regex=True, case=False)
        ].copy()

    serp_results["bubble_size"] = 35
    serp_results['bubble_size'] = serp_results['sentiment_positive'] * 20 + 10

    fig = px.scatter(
        serp_results,
        x="displayLink",
        y="rank",
        animation_frame=serp_results.index,
        animation_group="displayLink",
        color="displayLink",
        hover_name="link",
        hover_data=["searchTerms", "title", "rank", "sentiment"],
        size="bubble_size",
        text="displayLink",
        template="plotly_white",
        height=700,
    )

    fig.layout.title = "SERP Tracking"
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 500
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Rank (Lower is Better)",
        xaxis_title="Domain",
    )
    fig.update_traces(
        textfont=dict(color='black'),
        marker=dict(line=dict(color='black', width=0.5))
    )

    st.plotly_chart(fig)

    # Scatter Plot title length vs rank
    st.subheader("Title Length vs. Rank")
    st.write("""
    This scatter plot shows the relationship between the length of the titles and their ranking positions. Bubble sizes indicate the sentiment scores of the titles.
    """)

    serp_results["title_length"] = serp_results["title"].str.len()
    serp_results["sentiment_size"] = serp_results["sentiment_positive"] * 20 + 10

    fig = px.scatter(
        serp_results,
        x="title_length",
        y="rank",
        color="displayLink",
        size="sentiment_size",
        title="Title Length vs. Rank (Bubble size indicates sentiment)",
        template="plotly_white",
        height=600,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        yaxis_title="Rank (Lower is Better)",
        xaxis_title="Title Length (Number of Characters)",
    )

    st.plotly_chart(fig)

    # Keyword frequency in titles
    st.subheader("Keyword frequency in titles")
    st.write("""
    The word cloud below displays the most frequent words found in the titles of the filtered search results.
    """)

    text = " ".join(serp_results["title"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    st.pyplot(plt)

else:
    st.info("Please start SERP tracking or upload CSV files to proceed.")
