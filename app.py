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
    
    # Добавляем кнопку для ручного расчёта
    calculate = st.button("🚀 Рассчитать вероятности")

# Основная логика
if calculate:
    try:
        # Загрузка данных с правильными параметрами
        with st.spinner("Загружаем данные..."):
            # Используем период напрямую для yfinance
            data = yf.download(
                ticker, 
                period=period,
                progress=False,
                auto_adjust=True  # Автоматическая корректировка цен
            )
            
            if data.empty:
                st.error(f"❌ Монета {ticker} не найдена. Проверьте тикер.")
                st.info("💡 Попробуйте: BTC-USD, ETH-USD, SOL-USD, ADA-USD")
                st.stop()
            
            # Проверяем, что данных достаточно
            if len(data) < 5:
                st.error("❌ Недостаточно данных для анализа (нужно минимум 5 дней)")
                st.stop()
            
            # Переименовываем колонки для единообразия
            if 'Close' not in data.columns:
                st.error("❌ Данные не содержат цену закрытия")
                st.stop()
        
        # Расчёт логарифмических доходностей
        data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()
        
        # Проверяем, что есть данные для расчёта
        if len(data) < 2:
            st.error("❌ Недостаточно данных после расчёта доходностей")
            st.stop()
        
        # Текущая цена (последнее значение)
        current_price = float(data['Close'].iloc[-1])
        
        # Статистика доходностей
        mean_return = float(data['returns'].mean())
        std_return = float(data['returns'].std())
        
        # Проверка на нулевую волатильность
        if std_return == 0:
            st.warning("⚠️ Волатильность равна 0, вероятности не могут быть рассчитаны")
            st.stop()
        
        # Моделирование Монте-Карло
        n_simulations = 10000
        n_days = forecast_days
        
        # Генерация случайных доходностей
        np.random.seed(42)  # Для воспроизводимости
        random_returns = np.random.normal(mean_return, std_return, (n_days, n_simulations))
        
        # Кумулятивная доходность
        cumulative_returns = np.cumsum(random_returns, axis=0)
        
        # Итоговые цены
        final_prices = current_price * np.exp(cumulative_returns[-1, :])
        
        # Расчёт вероятностей
        prob_up = np.mean(final_prices > current_price) * 100
        prob_down = 100 - prob_up
        
        # Средняя ожидаемая цена
        expected_price = float(np.mean(final_prices))
        price_std = float(np.std(final_prices))
        
        # Процентили
        percentile_5 = float(np.percentile(final_prices, 5))
        percentile_95 = float(np.percentile(final_prices, 95))
        
        # Сохраняем результаты в сессию
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
            'std_return': std_return
        }
        
        st.success("✅ Расчёт завершён!")
        
    except Exception as e:
        st.error(f"⚠️ Ошибка: {str(e)}")
        st.info("💡 Попробуйте другой тикер или период")
        import traceback
        st.code(traceback.format_exc())

# Отображение результатов
if st.session_state.get("results"):
    results = st.session_state.results
    
    # Метрики
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
    
    # График распределения
    st.subheader("📈 Распределение вероятных цен")
    
    # Создаём гистограмму
    fig = go.Figure()
    
    # Добавляем гистограмму
    fig.add_trace(go.Histogram(
        x=results['final_prices'],
        nbinsx=50,
        name="Распределение",
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Добавляем вертикальные линии
    fig.add_vline(x=results['current_price'], line_color="red", line_dash="dash", 
                  annotation_text=f"Текущая: ${results['current_price']:.2f}")
    fig.add_vline(x=results['expected_price'], line_color="green", line_dash="dash",
                  annotation_text=f"Ожидаемая: ${results['expected_price']:.2f}")
    
    # Добавляем доверительные интервалы
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
    
    # Дополнительная статистика
    with st.expander("📋 Детальная статистика"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**📊 Параметры доходности:**")
            st.write(f"Средняя дневная доходность: {results['mean_return']*100:.3f}%")
            st.write(f"Волатильность (σ): {results['std_return']*100:.3f}%")
            st.write(f"Количество дней в выборке: {len(results['data'])}")
            st.write(f"Период: {results['data'].index[0].strftime('%Y-%m-%d')} - {results['data'].index[-1].strftime('%Y-%m-%d')}")
        
        with col2:
            st.write("**🎯 Доверительные интервалы (5-95%):**")
            st.write(f"Нижняя граница: ${results['percentile_5']:.2f}")
            st.write(f"Верхняя граница: ${results['percentile_95']:.2f}")
            st.write(f"Диапазон: ${results['percentile_95'] - results['percentile_5']:.2f}")
            
            # Добавляем вероятность пробоя
            prob_breakout_up = np.mean(results['final_prices'] > results['current_price'] * 1.1) * 100
            prob_breakout_down = np.mean(results['final_prices'] < results['current_price'] * 0.9) * 100
            st.write(f"Вероятность роста >10%: {prob_breakout_up:.1f}%")
            st.write(f"Вероятность падения >10%: {prob_breakout_down:.1f}%")
    
    # График исторической динамики
    st.subheader("📉 Историческая динамика")
    fig2 = go.Figure()
    
    # Добавляем свечной график или линию
    fig2.add_trace(go.Scatter(
        x=results['data'].index,
        y=results['data']['Close'],
        mode='lines',
        name='Цена закрытия',
        line=dict(color='blue', width=2)
    ))
    
    # Добавляем скользящую среднюю
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
        yaxis_title=f"Цена ({ticker})",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Кнопка для сброса
    if st.button("🔄 Новый расчёт"):
        st.session_state.results = None
        st.rerun()

else:
    st.info("👈 Введите тикер и нажмите 'Рассчитать вероятности'")
    
    # Примеры тикеров
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

# Информация о сервисе
st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Как это работает:**\n\n"
    "1. Анализируется историческая волатильность\n"
    "2. Проводится 10 000 симуляций Монте-Карло\n"
    "3. Рассчитываются вероятности роста/падения\n"
    "4. Учитывается нормальное распределение доходностей\n\n"
    "⚠️ **Важно:** Результаты основаны на исторических данных и не гарантируют будущие движения."
)

st.sidebar.markdown("---")
st.sidebar.caption("Сделано с ❤️ для криптоэнтузиастов")