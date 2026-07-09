import streamlit as st
import pandas as pd
from momentum_analyzer import MomentumAnalyzer


DEFAULT_TICKERS = [
    'META','GOOGL','MU', 'NVDA', 'MSFT', 'AAPL', 'AMD', 'AMZN', 'RXT', 'RBRK', 'NOK', 'PATH', 'CBRS', 'RKLB', 'SOFI','MRAM','GSIT',
    'SNDK', 'CRWV', 'IREN', 'VNET', 'AMBA', 'FLEX', 'BEP', 'TSLA','RDDT', 'ONDS', 'SPCX', 'NBIS', 'ARM', 'MRVL',
    'SPY', 'QQQ', 'IWM', 'COHR', 'LITE', 'SNPS', 'CLS', 'ALAB','UMAC', 'LUNR', 'CDNS', 'STX',
]


st.set_page_config(page_title="Momentum Analyzer", layout="wide")


def parse_tickers(text: str):
    if not text:
        return []
    # split by comma, newline or whitespace
    parts = [p.strip().upper() for p in text.replace('\n', ',').split(',')]
    return [p for p in parts if p]


def main():
    st.title("Sustained Momentum Analyzer — Streamlit")

    with st.sidebar:
        st.header("Scan Settings")
        tickers_input = st.text_area("Tickers (comma or newline separated)", value=", ".join(DEFAULT_TICKERS), height=200)
        min_confidence = st.slider("Minimum confidence (%)", 0, 100, 60)
        save_output = st.checkbox("Save results to output directory", value=False)
        run_button = st.button("Run Scan")

    tickers = parse_tickers(tickers_input)

    if run_button:
        if not tickers:
            st.warning("Please provide at least one ticker.")
            return

        analyzer = MomentumAnalyzer(tickers, min_confidence=min_confidence)

        with st.spinner(f"Scanning {len(tickers)} tickers. This may take a while..."):
            results = analyzer.run()

        if results.empty:
            st.info("No sustained momentum signals detected for the provided tickers.")
        else:
            st.subheader("Signals Detected")
            st.dataframe(results.reset_index(drop=True))

            # Confidence bar chart
            try:
                conf_chart = results.set_index('Ticker')['Confidence'].sort_values(ascending=False)
                st.bar_chart(conf_chart)
            except Exception:
                pass

            # Download buttons
            csv = results.to_csv(index=False).encode('utf-8')
            json = results.to_json(orient='records', indent=2).encode('utf-8')

            st.download_button("Download CSV", csv, file_name="momentum_results.csv", mime='text/csv')
            st.download_button("Download JSON", json, file_name="momentum_results.json", mime='application/json')

            if save_output:
                analyzer.save_results(results)

    else:
        st.write("Enter tickers and press 'Run Scan' to start analysis.")


if __name__ == '__main__':
    main()
