import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt

# ---------------------------
# CONFIGURAÇÕES INICIAIS
# ---------------------------
st.set_page_config(page_title="Análise de Imóveis", layout="wide")
st.title("🏠 Análise e Avaliação de Imóveis")

# ---------------------------
# CARREGAR E PREPARAR OS DADOS
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("houses.csv", sep=';')
    cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
            'YearBuilt', 'Neighborhood', 'SalePrice']
    df = df[cols].dropna()
    return df

df = load_data()

@st.cache_resource
def get_regression_model_and_encoder(data):
    num_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
    cat_cols = ['Neighborhood']
    
    X = data[num_cols + cat_cols]
    y = data['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    X_train.loc[:, cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test.loc[:, cat_cols] = encoder.transform(X_test[cat_cols])

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    return model, encoder, X_test, y_test

@st.cache_resource
def get_classification_model_and_encoder(data):
    """Cria faixas de preço, treina o modelo de classificação e o encoder, e os armazena em cache."""
    df_class = data.copy()
    df_class["FaixaPreco"] = pd.cut(df_class["SalePrice"],
                                     bins=[0, 150000, 250000, 400000, df_class["SalePrice"].max()],
                                     labels=["baixo", "médio", "alto", "luxo"])
    
    num_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
    cat_cols = ['Neighborhood']
    
    X = df_class[num_cols + cat_cols]
    y = df_class['FaixaPreco']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    X_train.loc[:, cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test.loc[:, cat_cols] = encoder.transform(X_test[cat_cols])

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, encoder, X_test, y_test

def get_user_input():
    st.subheader("Insira as características do imóvel:")
    col1, col2 = st.columns(2)
    
    with col1:
        overall = st.slider("Qualidade geral (1-10):", 1, 10, 7)
        area = st.number_input("Área útil (GrLivArea):", 300, 6000, 1500)
        garagem = st.slider("Capacidade da garagem (carros):", 0, 4, 2)
    with col2:
        basement = st.number_input("Área total do porão (TotalBsmtSF):", 0, 3000, 800)
        ano = st.number_input("Ano de construção:", 1872, 2010, 2005)
        bairro = st.selectbox("Bairro:", sorted(df['Neighborhood'].unique()))

    entrada = pd.DataFrame(
        [[overall, area, garagem, basement, ano, bairro]],
        columns=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'Neighborhood']
    )
    return entrada

# ---------------------------
# NAVEGAÇÃO E PÁGINAS
# ---------------------------
page = st.sidebar.radio("Escolha a página:", ["Previsão de Preço", "Classificação de Faixa de Preço"])

# --- PÁGINA 1 – REGRESSÃO ---
if page == "Previsão de Preço":
    st.header("💰 Previsão do Preço de Venda")

    # Carrega modelo, encoder e dados de teste do cache
    model, encoder, X_test, y_test = get_regression_model_and_encoder(df)

    # Avaliação
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**R² do modelo (dados de teste):** {r2:.2f}")

    # Entradas do usuário
    user_input = get_user_input()

    if st.button("Prever preço"):
        cat_cols = ['Neighborhood']
        user_input.loc[:, cat_cols] = encoder.transform(user_input[cat_cols])
        pred = model.predict(user_input)[0]
        st.success(f"🏡 Preço estimado: **US$ {pred:,.2f}**")

    # Gráfico com reta de referência
    st.subheader("Comparação Real vs. Previsto (dados de teste)")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    # Adicionando a reta de referência (y=x)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
    ax.set_xlabel("Preço Real")
    ax.set_ylabel("Preço Previsto")
    ax.legend()
    st.pyplot(fig)

# --- PÁGINA 2 – CLASSIFICAÇÃO ---
else:
    st.header("🏷️ Classificação de Faixa de Preço")

    # Carrega modelo, encoder e dados de teste do cache
    model_c, encoder_c, X_test_c, y_test_c = get_classification_model_and_encoder(df)

    # Avaliação
    acc = accuracy_score(y_test_c, model_c.predict(X_test_c))
    st.write(f"**Acurácia do modelo (dados de teste):** {acc:.2f}")

    # Entradas do usuário
    user_input = get_user_input()

    if st.button("Classificar imóvel"):
        cat_cols = ['Neighborhood']
        user_input.loc[:, cat_cols] = encoder_c.transform(user_input[cat_cols])
        pred_class = model_c.predict(user_input)[0]
        st.success(f"🏷️ Faixa de preço estimada: **{pred_class.upper()}**")
