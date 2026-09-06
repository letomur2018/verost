import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import t

st.set_page_config(page_title="Крипто-вероятности + Паттерны", layout="wide")

st.title("📊 Калькулятор вероятностей + Паттерны криптомонет")
st.markdown("Введите тикер монеты, и сервис рассчитает вероятности, найдёт свечные паттерны и выдаст сигналы с подтверждениями.")

# ---------- Функции для паттернов (без TA-Lib) ----------
def detect_hammer(o, h, l, c, idx):
    """Молот (Hammer) - бычий разворот"""
    body = abs(c[idx] - o[idx])
    lower_shadow = min(o[idx], c[idx]) - l[idx]
    upper_shadow = h[idx] - max(o[idx], c[idx])
    # нижняя тень >= 2*body, верхняя <= 0.3*body, тело не слишком маленькое
    if body > 0 and lower_shadow >= 2*body and upper_shadow <= 0.3*body:
        return True
    return False

def detect_shooting_star(o, h, l, c, idx):
    """Падающая звезда (Shooting Star) - медвежий разворот"""
    body = abs(c[idx] - o[idx])
    upper_shadow = h[idx] - max(o[idx], c[idx])
    lower_shadow = min(o[idx], c[idx]) - l[idx]
    if body > 0 and upper_shadow >= 2*body and lower_shadow <= 0.3*body:
        return True
    return False

def detect_doji(o, h, l, c, idx):
    """Доджи - неопределённость (тело очень маленькое)"""
    body = abs(c[idx] - o[idx])
    high_low = h[idx] - l[idx]
    if high_low > 0 and body / high_low < 0.1:
        return True
    return False

def detect_engulfing(o, h, l, c, idx):
    """Поглощение (бычье или медвежье)"""
    if idx < 1:
        return None
    # Бычье: текущая свеча зелёная, предыдущая красная, тело текущей перекрывает предыдущую
    if c[idx] > o[idx] and c[idx-1] < o[idx-1]:
        if c[idx] >= o[idx-1] and o[idx] <= c[idx-1]:
            return 'bullish'
    # Медвежье: текущая красная, предыдущая зелёная, тело перекрывает
    if c[idx] < o[idx] and c[idx-1] > o[idx-1]:
        if o[idx] >= c[idx-1] and c[idx] <= o[idx-1]:
            return 'bearish'
    return None

def detect_morning_star(o, h, l, c, idx):
    """Утренняя звезда (бычий разворот) - три свечи"""
    if idx < 2:
        return False
    # 1-я свеча: медвежья (длинная)
    # 2-я: доджи или маленькая свеча (с тенью)
    # 3-я: бычья (длинная), закрывается выше середины 1-й
    if c[idx-2] < o[idx-2] and c[idx] > o[idx]:
        body1 = abs(c[idx-2] - o[idx-2])
        body2 = abs(c[idx-1] - o[idx-1])
        body3 = abs(c[idx] - o[idx])
        if body2 < body1 * 0.5 and body3 > body1 * 0.5:
            if c[idx] > (o[idx-2] + c[idx-2]) / 2:
                return True
    return False

def detect_evening_star(o, h, l, c, idx):
    """Вечерняя звезда (медвежий разворот) - три свечи"""
    if idx < 2:
        return False
    # 1-я бычья, 2-я доджи/маленькая, 3-я медвежья, закрывается ниже середины 1-й
    if c[idx-2] > o[idx-2] and c[idx] < o[idx]:
        body1 = abs(c[idx-2] - o[idx-2])
        body2 = abs(c[idx-1] - o[idx-1])
        body3 = abs(c[idx] - o[idx])
        if body2 < body1 * 0.5 and body3 > body1 * 0.5:
            if c[idx] < (o[idx-2] + c[idx-2]) / 2:
                return True
    return False

# ---------- Индикаторы ----------
def rsi(series, period=14):
    """Индекс относительной силы"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def moving_average(series, window):
    return series.rolling(window=window).mean()

# ---------- Анализ паттернов и подтверждений ----------
def analyze_patterns_and_signals(data):
    """
    Возвращает словарь с найденными паттернами и подтверждениями.
    """
    o = data['Open'].values
    h = data['High'].values
    l = data['Low'].values
    c = data['Close'].values
    vol = data['Volume'].values
    
    # Последние 10 свечей для поиска
    last_idx = len(c) - 1
    signals = []
    
    # Проверяем каждую из последних 5 свечей на наличие паттернов
    for i in range(max(0, last_idx-5), last_idx+1):
        # Молот
        if detect_hammer(o, h, l, c, i):
            signals.append({
                'date': data.index[i].strftime('%Y-%m-%d'),
                'pattern': 'Молот (Hammer)',
                'type': 'бычий разворот',
                'price': c[i],
                'strength': 'средний'
            })
        # Падающая звезда
        if detect_shooting_star(o, h, l, c, i):
            signals.append({
                'date': data.index[i].strftime('%Y-%m-%d'),
                'pattern': 'Падающая звезда (Shooting Star)',
                'type': 'медвежий разворот',
                'price': c[i],
                'strength': 'средний'
            })
        # Доджи
        if detect_doji(o, h, l, c, i):
            signals.append({
                'date': data.index[i].strftime('%Y-%m-%d'),
                'pattern': 'Доджи (Doji)',
                'type': 'неопределённость',
                'price': c[i],
                'strength': 'слабый'
            })
        # Поглощение
        engulf = detect_engulfing(o, h, l, c, i)
        if engulf:
            signals.append({
                'date': data.index[i].strftime('%Y-%m-%d'),
                'pattern': 'Поглощение (Engulfing)',
                'type': f'{engulf} разворот',
                'price': c[i],
                'strength': 'сильный'
            })
        # Утренняя звезда
        if detect_morning_star(o, h, l, c, i):
            signals.append({
                'date': data.index[i].strftime('%Y-%m-%d'),
                'pattern': 'Утренняя звезда (Morning Star)',
                'type': 'бычий разворот',
                'price': c[i],
                'strength': 'сильный'
            })
        # Вечерняя звезда
        if detect_evening_star(o, h, l, c, i):
            signals.append({
                'date': data.index[i].strftime('%Y-%m-%d'),
                'pattern': 'Вечерняя звезда (Evening Star)',
                'type': 'медвежий разворот',
                'price': c[i],
                'strength': 'сильный'
            })
    
    # Добавляем подтверждения на основе последних данных
    if len(data) > 20:
        # 1. Текущий объём относительно среднего (за 20 дней)
        avg_vol = data['Volume'].rolling(20).mean().iloc[-1]
        current_vol = data['Volume'].iloc[-1]
        vol_confirmation = 'высокий' if current_vol > avg_vol * 1.5 else 'нормальный' if current_vol > avg_vol * 0.8 else 'низкий'
        
        # 2. Пересечение цены с MA20 и MA50
        ma20 = moving_average(data['Close'], 20).iloc[-1]
        ma50 = moving_average(data['Close'], 50).iloc[-1] if len(data) >= 50 else None
        price = data['Close'].iloc[-1]
        price_above_ma20 = price > ma20
        price_above_ma50 = price > ma50 if ma50 is not None else None
        
        # 3. RSI
        rsi_vals = rsi(data['Close'], 14)
        last_rsi = rsi_vals.iloc[-1]
        rsi_signal = 'перекупленность' if last_rsi > 70 else 'перепроданность' if last_rsi < 30 else 'нейтрально'
        
        # 4. Дивергенция (цена растёт, RSI падает или наоборот) - упрощённо
        # Берём последние 5 значений
        price_slope = data['Close'].iloc[-5:].values
        rsi_slope = rsi_vals.iloc[-5:].values
        # Если цена идёт вверх, а RSI вниз -> медвежья дивергенция
        if len(price_slope) >= 2 and len(rsi_slope) >= 2:
            price_change = price_slope[-1] - price_slope[0]
            rsi_change = rsi_slope[-1] - rsi_slope[0]
            if price_change > 0 and rsi_change < 0:
                div_signal = 'медвежья дивергенция'
            elif price_change < 0 and rsi_change > 0:
                div_signal = 'бычья дивергенция'
            else:
                div_signal = 'дивергенции нет'
        else:
            div_signal = 'недостаточно данных'
        
        # Добавляем подтверждения в первый сигнал (если есть)
        if signals:
            # Добавим поля к последнему сигналу
            signals[0]['volume_confirmation'] = vol_confirmation
            signals[0]['price_above_ma20'] = price_above_ma20
            signals[0]['price_above_ma50'] = price_above_ma50 if ma50 is not None else None
            signals[0]['rsi'] = last_rsi
            signals[0]['rsi_signal'] = rsi_signal
            signals[0]['divergence'] = div_signal
        else:
            # Если паттернов нет, создаём "сигнал" из индикаторов
            signals.append({
                'date': data.index[-1].strftime('%Y-%m-%d'),
                'pattern': 'Нет свечного паттерна',
                'type': 'только индикаторы',
                'price': data['Close'].iloc[-1],
                'strength': 'нет',
                'volume_confirmation': vol_confirmation,
                'price_above_ma20': price_above_ma20,
                'price_above_ma50': price_above_ma50 if ma50 is not None else None,
                'rsi': last_rsi,
                'rsi_signal': rsi_signal,
                'divergence': div_signal
            })
    
    return signals

# ---------- Основной интерфейс ----------
with st.sidebar:
    st.header("⚙️ Настройки")
    ticker = st.text_input("Введите тикер монеты", value="BTC-USD").upper()
    period = st.selectbox("Период для анализа", ["7d", "30d", "90d", "6mo", "1y"], index=2)
    forecast_days = st.slider("Прогнозируемый период (дней)", 1, 30, 7)
    
    method = st.selectbox(
        "Метод моделирования вероятностей",
        ["t-распределение (рекомендуется)", "Нормальное распределение", "Исторический бутстрап"],
        index=0
    )
    
    half_life = st.slider("Период полураспада для весов (дней)", 5, 60, 20)
    
    calculate = st.button("🚀 Рассчитать вероятности и паттерны")

# Функция загрузки с автоувеличением периода
def load_data_with_retry(ticker, period, min_days=30):
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

if calculate:
    try:
        with st.spinner("Загружаем данные и анализируем..."):
            data, close_series, used_period = load_data_with_retry(ticker, period, min_days=30)
            if data is None:
                st.error(f"❌ Не удалось загрузить достаточно данных для {ticker}.")
                st.stop()
            
            if used_period != period:
                st.info(f"ℹ️ Используем период {used_period} (дней: {len(close_series)})")
            
            current_price = float(close_series.iloc[-1])
        
        # Расчёт доходностей (как ранее)
        returns = np.log(close_series / close_series.shift(1)).dropna()
        if len(returns) < 10:
            st.error(f"❌ Недостаточно доходностей ({len(returns)} точек). Нужно минимум 10.")
            st.stop()
        
        # EWMA
        lambda_ = np.exp(-np.log(2) / half_life)
        weights = (1 - lambda_) * (lambda_ ** np.arange(len(returns)-1, -1, -1))
        weights = weights / weights.sum()
        mean_return = np.average(returns, weights=weights)
        var_w = np.average((returns - mean_return)**2, weights=weights)
        std_return = np.sqrt(var_w)
        
        # t-распределение
        df, loc, scale = t.fit(returns)
        
        # Моделирование
        n_simulations = 10000
        np.random.seed(42)
        if method == "Нормальное распределение":
            random_returns = np.random.normal(mean_return, std_return, (forecast_days, n_simulations))
        elif method == "t-распределение (рекомендуется)":
            random_returns = t.rvs(df, loc=loc, scale=scale, size=(forecast_days, n_simulations))
        else:
            indices = np.random.choice(len(returns), size=(forecast_days, n_simulations), p=weights)
            random_returns = returns.iloc[indices].values
        
        cumulative_returns = np.cumsum(random_returns, axis=0)
        final_prices = current_price * np.exp(cumulative_returns[-1, :])
        
        # Вероятности и метрики
        prob_up = np.mean(final_prices > current_price) * 100
        prob_down = 100 - prob_up
        expected_price = float(np.mean(final_prices))
        percentile_5 = float(np.percentile(final_prices, 5))
        percentile_95 = float(np.percentile(final_prices, 95))
        var_95 = np.percentile(final_prices - current_price, 5)
        prob_gain_10 = np.mean(final_prices > current_price * 1.1) * 100
        prob_loss_10 = np.mean(final_prices < current_price * 0.9) * 100
        
        # Анализ паттернов и подтверждений
        signals = analyze_patterns_and_signals(data)
        
        # Сохраняем всё в сессию
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
            'used_period': used_period,
            'signals': signals
        }
        
        st.success("✅ Анализ завершён!")
        
    except Exception as e:
        st.error(f"⚠️ Ошибка: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# ---------- Отображение ----------
if st.session_state.get("results"):
    results = st.session_state.results
    
    # --- Верхние метрики ---
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
    
    # --- Блок паттернов и сигналов ---
    st.subheader("🕯️ Обнаруженные свечные паттерны и подтверждения")
    signals = results['signals']
    if signals:
        # Создаём DataFrame для красивого вывода
        df_signals = pd.DataFrame(signals)
        # Выводим таблицу
        st.dataframe(
            df_signals[['date', 'pattern', 'type', 'price', 'strength']],
            use_container_width=True,
            hide_index=True
        )
        
        # Дополнительные подтверждения (выводим из первого сигнала, если есть)
        first = signals[0]
        if 'volume_confirmation' in first:
            st.write("**📊 Подтверждения (на основе последней свечи):**")
            col1, col2, col3 = st.columns(3)
            with col1:
                vol_color = "🟢" if first['volume_confirmation'] == 'высокий' else "🟡" if first['volume_confirmation'] == 'нормальный' else "🔴"
                st.write(f"{vol_color} Объём: {first['volume_confirmation']}")
            with col2:
                ma20_color = "🟢" if first['price_above_ma20'] else "🔴"
                st.write(f"{ma20_color} Цена выше MA20: {'Да' if first['price_above_ma20'] else 'Нет'}")
                if first['price_above_ma50'] is not None:
                    ma50_color = "🟢" if first['price_above_ma50'] else "🔴"
                    st.write(f"{ma50_color} Цена выше MA50: {'Да' if first['price_above_ma50'] else 'Нет'}")
            with col3:
                rsi_val = first['rsi']
                rsi_color = "🟢" if rsi_val < 30 else "🔴" if rsi_val > 70 else "🟡"
                st.write(f"{rsi_color} RSI: {rsi_val:.1f} ({first['rsi_signal']})")
                div_color = "🟢" if 'бычья' in first['divergence'] else "🔴" if 'медвежья' in first['divergence'] else "⚪"
                st.write(f"{div_color} Дивергенция: {first['divergence']}")
    else:
        st.info("ℹ️ Паттернов не обнаружено. Проверьте другие периоды.")
    
    # --- График распределения ---
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
    
    # --- Детальная статистика ---
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
    
    # --- Историческая динамика ---
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
        ma20 = results['data']['Close'].rolling(20).mean()
        fig2.add_trace(go.Scatter(
            x=results['data'].index,
            y=ma20,
            mode='lines',
            name='MA20',
            line=dict(color='orange', width=1, dash='dash')
        ))
        if len(results['data']) >= 50:
            ma50 = results['data']['Close'].rolling(50).mean()
            fig2.add_trace(go.Scatter(
                x=results['data'].index,
                y=ma50,
                mode='lines',
                name='MA50',
                line=dict(color='purple', width=1, dash='dot')
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
    st.info("👈 Введите тикер и нажмите 'Рассчитать вероятности и паттерны'")
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
    with st.expander("🧠 Как работают паттерны и подтверждения"):
        st.write("""
        **Свечные паттерны** (без TA-Lib, собственная реализация):
        - Молот / Падающая звезда – разворотные сигналы.
        - Доджи – неопределённость.
        - Поглощение – сильный разворот.
        - Утренняя/Вечерняя звезда – разворот из трёх свечей.
        
        **Подтверждения:**
        - Объём торгов (высокий = подтверждение движения).
        - Положение цены относительно скользящих средних (MA20/MA50).
        - RSI (перекупленность/перепроданность).
        - Дивергенция цены и RSI.
        """)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Что нового:**\n\n"
    "✅ Детекция свечных паттернов (без TA-Lib)\n"
    "✅ Подтверждения (объём, MA, RSI, дивергенция)\n"
    "✅ Вывод сигналов в таблице\n"
    "✅ t-распределение + EWMA для вероятностей\n\n"
    "⚠️ Результаты не являются инвестиционной рекомендацией."
)
st.sidebar.caption("Сделано с ❤️ для криптоэнтузиастов")