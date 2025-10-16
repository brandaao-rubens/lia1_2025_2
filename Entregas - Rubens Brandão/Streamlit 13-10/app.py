import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date, timedelta
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Análise e Previsão de Ações", layout="wide")

st.title("📈 Análise e Previsão de Ações da Bolsa de Valores")

# --- BARRA LATERAL (INPUTS DO USUÁRIO) ---
with st.sidebar:
    st.header("Parâmetros")

    ticker_symbol = st.text_input("Digite o Ticker da Ação", "PETR4.SA")
    st.info("Exemplos: PETR4.SA, MGLU3.SA, WEGE3.SA, AAPL, GOOGL")

    today = date.today()
    start_date = st.date_input("Data Inicial", today - timedelta(days=365 * 2))
    end_date = st.date_input("Data Final", today)

    forecast_period = st.number_input("Dias para previsão", min_value=1, max_value=90, value=40)

    process_button = st.button("Analisar e Prever")

# --- CACHE DE DOWNLOAD ---
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data

# --- LÓGICA PRINCIPAL ---
if process_button:
    try:
        st.header(f"📊 Análise de {ticker_symbol}")
        data = load_data(ticker_symbol, start_date, end_date)

        if data is None or data.empty:
            st.error("Nenhum dado encontrado para o ticker ou período especificado.")
        else:
            ts_data = data['Close'].asfreq('B').fillna(method='ffill')

            # KPIs principais
            col1, col2, col3 = st.columns(3)
            preco_atual = float(ts_data.iloc[-1])
            variacao_30d = float((ts_data.iloc[-1] / ts_data.iloc[-21] - 1) * 100) if len(ts_data) > 21 else 0.0
            volatilidade = float(ts_data.pct_change().std() * 100)

            col1.metric("💰 Preço Atual", f"R$ {preco_atual:.2f}")
            col2.metric("📉 Variação 30d", f"{variacao_30d:.2f}%")
            col3.metric("⚡ Volatilidade", f"{volatilidade:.2f}%")

            tab1, tab2, tab3 = st.tabs(["📊 Análise Histórica", "🔮 Previsão", "📑 Dados Brutos"])

            # --- ABA 1: ANÁLISE HISTÓRICA COM MATPLOTLIB ---
            with tab1:
                st.subheader("Variação do Preço e Médias Móveis")
                
                plot_data = data.copy()
                plot_data['MM21'] = plot_data['Close'].rolling(window=21).mean()
                plot_data['MM50'] = plot_data['Close'].rolling(window=50).mean()
                plot_data.dropna(inplace=True)

                if plot_data.empty:
                    st.warning("Não há dados suficientes no período para calcular as médias móveis. Por favor, selecione um intervalo maior.")
                else:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    ax.plot(plot_data.index, plot_data['Close'], label='Fechamento', color='cyan')
                    ax.plot(plot_data.index, plot_data['MM21'], label='Média Móvel 21d', linestyle='--', color='orange')
                    ax.plot(plot_data.index, plot_data['MM50'], label='Média Móvel 50d', linestyle=':', color='lime')
                    
                    ax.set_title(f"Preço de Fechamento e Médias Móveis - {ticker_symbol}", color='white')
                    ax.set_xlabel("Data", color='white')
                    ax.set_ylabel("Preço (R$)", color='white')
                    ax.tick_params(axis='x', colors='white', rotation=25)
                    ax.tick_params(axis='y', colors='white')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.3)
                    fig.patch.set_facecolor('#0E1117')
                    ax.set_facecolor('#0E1117')
                    
                    st.pyplot(fig)

            # --- ABA 2: PREVISÃO COM MATPLOTLIB ---
            with tab2:
                st.subheader(f"Previsão para os próximos {forecast_period} dias")

                with st.spinner('Treinando o modelo de previsão...'):
                    model = SARIMAX(ts_data, order=(4, 1, 0), seasonal_order=(1, 1, 1, 5))
                    model_fit = model.fit(disp=False)
                    forecast = model_fit.get_forecast(steps=forecast_period)

                    forecast_index = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='B')
                    forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index, name="Previsão")
                    confidence_int = forecast.conf_int()
                    
                    historical_tail = ts_data.tail(120)

                    fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))

                    ax_forecast.plot(historical_tail.index, historical_tail.values, label='Histórico de Fechamento', color='royalblue')
                    ax_forecast.plot(forecast_series.index, forecast_series.values, label='Previsão', linestyle='--', color='tomato')
                    
                    if not confidence_int.isnull().values.any():
                        ax_forecast.fill_between(forecast_index,
                                                 confidence_int.iloc[:, 0],
                                                 confidence_int.iloc[:, 1], 
                                                 color='pink', 
                                                 alpha=0.3, 
                                                 label='Intervalo de Confiança')

                    ax_forecast.set_title(f"Previsão de Preço de Fechamento - {ticker_symbol}", color='white')
                    ax_forecast.set_xlabel("Data", color='white')
                    ax_forecast.set_ylabel("Preço (R$)", color='white')
                    ax_forecast.tick_params(axis='x', colors='white', rotation=25)
                    ax_forecast.tick_params(axis='y', colors='white')
                    ax_forecast.legend()
                    ax_forecast.grid(True, linestyle='--', alpha=0.3)
                    fig_forecast.patch.set_facecolor('#0E1117')
                    ax_forecast.set_facecolor('#0E1117')
                    
                col_chart, col_data = st.columns([3, 1])
                with col_chart:
                    st.pyplot(fig_forecast)
                    if confidence_int.isnull().values.any():
                        st.warning("Aviso: O intervalo de confiança não pôde ser calculado para este ativo/período.")
                with col_data:
                    st.write("Dados Previstos:")
                    st.dataframe(forecast_series)
                
                st.download_button(
                    label="📥 Baixar Previsão em CSV",
                    data=forecast_series.to_csv(header=True).encode('utf-8'),
                    file_name=f"previsao_{ticker_symbol}.csv",
                    mime="text/csv"
                )

            # --- ABA 3: DADOS BRUTOS ---
            with tab3:
                st.subheader("Dados Originais")
                st.dataframe(data, use_container_width=True)

    except Exception as ex:
        st.error(f"Ocorreu um erro durante a execução: {ex}")
        st.exception(ex)
