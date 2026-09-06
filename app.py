import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import t

st.set_page_config(page_title="Крипто-вероятности (улучшенная)", layout="wide")

st.title("📊 Калькулятор вероятностей движения криптомонеты")
st.markdown("Введите тикер монеты, и сервис рассчитает вероятности роста/падения на основе исторических данных.")

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    ticker = st.text_input("Введите тикер монеты", value="BTC-USD").upper()
    period = st.selectbox("Период для анализа", ["7d", "30d", "90d", "6mo", "1y"], index=2)
    forecast_days = st.slider("Прогнозируемый период (дней)", 1, 30, 7)
    
    method = st.selectbox(
        "Метод моделирования",
        ["t-распределение (рекомендуется)", "Нормальное распределение", "Исторический бутстрап"],
        index=0
    )
    
    half_life = st.slider("Период полураспада для весов (дней)", 5, 60, 20)
    
    calculate = st.button("🚀 Рассчитать вероятности")

# Функция загрузки данных с автоматическим увеличением периода
def load_data_with_retry(ticker, period, min_days=30):
    """Загружает данные, при необходимости увеличивая период до 90d или 1y."""
    periods_to_try = [period, "90d", "1y"]
    for p in periods_to_try:
        data = yf.download(ticker, period=p, progress=False)
        if not data.empty and 'Close' in data.columns:
            close_series = data['Close']
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            if len(close_series) >= min_days:
                return data, close_series, p
    return None, None, None

# Основная логика
if calculate:
    try:
        with st.spinner("Загружаем данные..."):
            data, close_series, used_period = load_data_with_retry(ticker, period, min_days=30)
            
            if data is None:
                st.error(f"❌ Не удалось загрузить достаточно данных для {ticker}. Попробуйте другой тикер.")
                st.info("💡 Попробуйте: BTC-USD, ETH-USD, SOL-USD, ADA-USD")
                st.stop()
            
            if used_period != period:
                st.info(f"ℹ️ Данных за {period} недостаточно. Используем период {used_period} (дней: {len(close_series)}).")
            
            current_price = float(close_series.iloc[-1])
        
        # Расчёт логарифмических доходностей
        returns = np.log(close_series / close_series.shift(1)).dropna()
        
        if len(returns) < 10:
            st.error(f"❌ После расчёта доходностей осталось всего {len(returns)} точек. Нужно минимум 10.")
            st.stop()
        
        # Экспоненциально взвешенные параметры (EWMA)
        lambda_ = np.exp(-np.log(2) / half_life)
        weights = (1 - lambda_) * (lambda_ ** np.arange(len(returns)-1, -1, -1))
        weights = weights / weights.sum()
        
        mean_return = np.average(returns, weights=weights)
        var_w = np.average((returns - mean_return)**2, weights=weights)
        std_return = np.sqrt(var_w)
        
        # Для t-распределения подгоняем параметры
        df, loc, scale = t.fit(returns)
        
        # Моделирование
        n_simulations = 10000
        np.random.seed(42)
        
        if method == "Нормальное распределение":
            random_returns = np.random.normal(mean_return, std_return, (forecast_days, n_simulations))
        elif method == "t-распределение (рекомендуется)":
            random_returns = t.rvs(df, loc=loc, scale=scale, size=(forecast_days, n_simulations))
        else:  # Исторический бутстрап
            indices = np.random.choice(len(returns), size=(forecast_days, n_simulations), p=weights)
            random_returns = returns.iloc[indices].values
        
        cumulative_returns = np.cumsum(random_returns, axis=0)
        final_prices = current_price * np.exp(cumulative_returns[-1, :])
        
        prob_up = np.mean(final_prices > current_price) * 100
        prob_down = 100 - prob_up
        expected_price = float(np.mean(final_prices))
        percentile_5 = float(np.percentile(final_prices, 5))
        percentile_95 = float(np.percentile(final_prices, 95))
        var_95 = np.percentile(final_prices - current_price, 5)
        prob_gain_10 = np.mean(final_prices > current_price * 1.1) * 100
        prob_loss_10 = np.mean(final_prices < current_price * 0.9) * 100
        
        st.session_state.results = {
            'current_price': current_price,
            'expected_price': expected_price,
            'prob_up': prob_up,
            'prob_down': prob_down,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'final_prices': final_prices,
            'data': data,
            'ticker': ticker,
            'forecast_days': forecast_days,
            'mean_return': mean_return,
            'std_return': std_return,
            'var_95': var_95,
            'prob_gain_10': prob_gain_10,
            'prob_loss_10': prob_loss_10,
            'method': method,
            'half_life': half_life,
            'used_period': used_period
        }
        
        st.success("✅ Расчёт завершён!")
        
    except Exception as e:
        st.error(f"⚠️ Ошибка: {str(e)}")
        st.info("💡 Попробуйте другой тикер или проверьте интернет-соединение.")
        import traceback
        st.code(traceback.format_exc())

# Отображение результатов (без изменений)
if st.session_state.get("results"):
    results = st.session_state.results
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Текущая цена", f"${results['current_price']:.2f}")
    with col2:
        delta_percent = ((results['expected_price'] / results['current_price'] - 1) * 100)
        st.metric("📈 Ожидаемая цена", f"${results['expected_price']:.2f}", 
                 delta=f"{delta_percent:.2f}%")
    with col3:
        st.metric("📊 Вероятность роста", f"{results['prob_up']:.1f}%")
    with col4:
        st.metric("📉 Вероятность падения", f"{results['prob_down']:.1f}%")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📉 VaR (95%)", f"${results['var_95']:.2f}", 
                 delta=f"{(results['var_95']/results['current_price']*100):.1f}%")
    with col2:
        st.metric("🚀 Рост >10%", f"{results['prob_gain_10']:.1f}%")
    with col3:
        st.metric("📉 Падение >10%", f"{results['prob_loss_10']:.1f}%")
    
    st.subheader("📈 Распределение вероятных цен")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=results['final_prices'],
        nbinsx=50,
        name="Распределение",
        marker_color='lightblue',
        opacity=0.7
    ))
    fig.add_vline(x=results['current_price'], line_color="red", line_dash="dash", 
                  annotation_text=f"Текущая: ${results['current_price']:.2f}")
    fig.add_vline(x=results['expected_price'], line_color="green", line_dash="dash",
                  annotation_text=f"Ожидаемая: ${results['expected_price']:.2f}")
    fig.add_vline(x=results['percentile_5'], line_color="orange", line_dash="dot",
                  annotation_text="5%")
    fig.add_vline(x=results['percentile_95'], line_color="orange", line_dash="dot",
                  annotation_text="95%")
    fig.update_layout(
        xaxis_title="Цена ($)",
        yaxis_title="Частота",
        height=400,
        showlegend=True,
        bargap=0.05
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📋 Детальная статистика и параметры модели"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**📊 Параметры доходности (взвешенные):**")
            st.write(f"Средняя дневная доходность (EWMA): {results['mean_return']*100:.3f}%")
            st.write(f"Волатильность (EWMA): {results['std_return']*100:.3f}%")
            st.write(f"Количество дней в выборке: {len(results['data'])}")
            st.write(f"Период: {results['data'].index[0].strftime('%Y-%m-%d')} - {results['data'].index[-1].strftime('%Y-%m-%d')}")
            st.write(f"Использованный период: {results.get('used_period', 'не указан')}")
        with col2:
            st.write("**🎯 Доверительные интервалы (5-95%):**")
            st.write(f"Нижняя граница: ${results['percentile_5']:.2f}")
            st.write(f"Верхняя граница: ${results['percentile_95']:.2f}")
            st.write(f"Диапазон: ${results['percentile_95'] - results['percentile_5']:.2f}")
            st.write(f"VaR (95%): ${results['var_95']:.2f}")
            st.write(f"Вероятность роста >10%: {results['prob_gain_10']:.1f}%")
            st.write(f"Вероятность падения >10%: {results['prob_loss_10']:.1f}%")
        st.write(f"Метод моделирования: {results['method']}")
        st.write(f"Период полураспада: {results['half_life']} дней")
    
    st.subheader("📉 Историческая динамика")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=results['data'].index,
        y=results['data']['Close'],
        mode='lines',
        name='Цена закрытия',
        line=dict(color='blue', width=2)
    ))
    if len(results['data']) > 20:
        ma20 = results['data']['Close'].rolling(window=20).mean()
        fig2.add_trace(go.Scatter(
            x=results['data'].index,
            y=ma20,
            mode='lines',
            name='MA 20 дней',
            line=dict(color='orange', width=1, dash='dash')
        ))
    fig2.update_layout(
        xaxis_title="Дата",
        yaxis_title=f"Цена ({results['ticker']})",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    if st.button("🔄 Новый расчёт"):
        st.session_state.results = None
        st.rerun()

else:
    st.info("👈 Введите тикер и нажмите 'Рассчитать вероятности'")
    with st.expander("📚 Примеры тикеров"):
        st.write("""
        - **BTC-USD** - Bitcoin
        - **ETH-USD** - Ethereum
        - **SOL-USD** - Solana
        - **ADA-USD** - Cardano
        - **DOT-USD** - Polkadot
        - **AVAX-USD** - Avalanche
        - **MATIC-USD** - Polygon
        """)
    
    with st.expander("🧠 О методе моделирования"):
        st.write("""
        - **t-распределение (рекомендуется)**: учитывает «толстые хвосты» – экстремальные движения случаются чаще, чем в нормальном распределении. 
        - **Нормальное распределение**: классический подход, но недооценивает риски экстремальных движений.
        - **Исторический бутстрап**: использует случайную выборку из реальных исторических доходностей (с весами по времени). Не делает предположений о распределении.
        """)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Улучшения в этой версии:**\n\n"
    "✅ t-распределение для учёта толстых хвостов\n"
    "✅ Экспоненциально взвешенные параметры (EWMA)\n"
    "✅ Возможность выбора метода моделирования\n"
    "✅ Расчёт VaR (95%) и вероятностей сильных движений\n"
    "✅ Автоматическое увеличение периода при недостатке данных\n\n"
    "⚠️ **Важно:** Результаты основаны на исторических данных и не гарантируют будущие движения."
)
st.sidebar.markdown("---")
st.sidebar.caption("Сделано с ❤️ для криптоэнтузиастов")