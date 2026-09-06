import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Крипто-вероятности", layout="wide")

st.title("📊 Калькулятор вероятностей движения криптомонеты")
st.markdown("Введите тикер монеты, и сервис рассчитает вероятности роста/падения на основе исторических данных.")

# Боковая панель с вводом
with st.sidebar:
    st.header("⚙️ Настройки")
    ticker = st.text_input("Введите тикер монеты", value="BTC-USD").upper()
    period = st.selectbox("Период для анализа", ["7d", "30d", "90d", "6mo", "1y"], index=2)
    forecast_days = st.slider("Прогнозируемый период (дней)", 1, 30, 7)
    
    if st.button("🚀 Рассчитать вероятности"):
        st.session_state.calculate = True
    else:
        st.session_state.calculate = False

# Основная логика
if st.session_state.get("calculate", False):
    try:
        # Загрузка данных
        with st.spinner("Загружаем данные..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90 if period in ["7d", "30d"] else 180 if period == "90d" else 365)
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Монета не найдена. Проверьте тикер.")
                st.stop()
        
        # Расчёт логарифмических доходностей
        data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data.dropna(inplace=True)
        
        # Текущая цена
        current_price = data['Close'].iloc[-1]
        
        # Статистика
        mean_return = data['returns'].mean()
        std_return = data['returns'].std()
        
        # Моделирование Монте-Карло
        n_simulations = 10000
        n_days = forecast_days
        
        # Генерация случайных доходностей
        random_returns = np.random.normal(mean_return, std_return, (n_days, n_simulations))
        
        # Кумулятивная доходность
        cumulative_returns = np.cumsum(random_returns, axis=0)
        
        # Итоговые цены
        final_prices = current_price * np.exp(cumulative_returns[-1, :])
        
        # Расчёт вероятностей
        prob_up = np.mean(final_prices > current_price) * 100
        prob_down = 100 - prob_up
        
        # Средняя ожидаемая цена
        expected_price = np.mean(final_prices)
        price_std = np.std(final_prices)
        
        # Процентили
        percentile_5 = np.percentile(final_prices, 5)
        percentile_95 = np.percentile(final_prices, 95)
        
        # Сохраняем в сессию
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
            'forecast_days': forecast_days
        }
        
        st.success("✅ Расчёт завершён!")
        
    except Exception as e:
        st.error(f"⚠️ Ошибка: {e}")
        st.stop()

# Отображение результатов
if st.session_state.get("results"):
    results = st.session_state.results
    
    # Метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Текущая цена", f"${results['current_price']:.2f}")
    with col2:
        st.metric("📈 Ожидаемая цена", f"${results['expected_price']:.2f}", 
                 delta=f"{((results['expected_price']/results['current_price']-1)*100):.2f}%")
    with col3:
        st.metric("📊 Вероятность роста", f"{results['prob_up']:.1f}%")
    with col4:
        st.metric("📉 Вероятность падения", f"{results['prob_down']:.1f}%")
    
    # График распределения
    st.subheader("📈 Распределение вероятных цен")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=results['final_prices'],
        nbinsx=50,
        name="Распределение",
        marker_color='lightblue'
    ))
    fig.add_vline(x=results['current_price'], line_color="red", line_dash="dash", 
                  annotation_text="Текущая цена")
    fig.add_vline(x=results['expected_price'], line_color="green", line_dash="dash",
                  annotation_text="Ожидаемая цена")
    fig.update_layout(
        xaxis_title="Цена",
        yaxis_title="Частота",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Дополнительная статистика
    with st.expander("📋 Детальная статистика"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Параметры доходности:**")
            st.write(f"Средняя дневная доходность: {results['data']['returns'].mean()*100:.3f}%")
            st.write(f"Волатильность: {results['data']['returns'].std()*100:.3f}%")
            st.write(f"Количество дней в выборке: {len(results['data'])}")
        with col2:
            st.write("**Доверительные интервалы:**")
            st.write(f"5-й процентиль: ${results['percentile_5']:.2f}")
            st.write(f"95-й процентиль: ${results['percentile_95']:.2f}")
            st.write(f"Диапазон: ${results['percentile_95'] - results['percentile_5']:.2f}")
    
    # График исторической динамики
    st.subheader("📉 Историческая динамика")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=results['data'].index,
        y=results['data']['Close'],
        mode='lines',
        name='Цена закрытия',
        line=dict(color='blue')
    ))
    fig2.update_layout(
        xaxis_title="Дата",
        yaxis_title="Цена ($)",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)
    
else:
    st.info("👈 Введите тикер и нажмите 'Рассчитать вероятности'")

# Информация о сервисе
st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Как это работает:**\n"
    "- Анализируется историческая волатильность монеты\n"
    "- Проводится 10 000 симуляций Монте-Карло\n"
    "- Рассчитываются вероятности роста/падения\n"
    "- Учитывается нормальное распределение доходностей"
)