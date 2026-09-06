import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import t

st.set_page_config(page_title="Крипто-вероятности + Паттерны", layout="wide")

st.title("📊 Калькулятор вероятностей + Паттерны криптомонет")
st.markdown("Введите тикер монеты, и сервис рассчитает вероятности движения на выбранном таймфрейме.")

# ---------- Функции для паттернов (без TA-Lib) ----------
def detect_hammer(o, h, l, c, idx):
    body = abs(c[idx] - o[idx])
    lower_shadow = min(o[idx], c[idx]) - l[idx]
    upper_shadow = h[idx] - max(o[idx], c[idx])
    if body > 0 and lower_shadow >= 2*body and upper_shadow <= 0.3*body:
        return True
    return False

def detect_shooting_star(o, h, l, c, idx):
    body = abs(c[idx] - o[idx])
    upper_shadow = h[idx] - max(o[idx], c[idx])
    lower_shadow = min(o[idx], c[idx]) - l[idx]
    if body > 0 and upper_shadow >= 2*body and lower_shadow <= 0.3*body:
        return True
    return False

def detect_doji(o, h, l, c, idx):
    body = abs(c[idx] - o[idx])
    high_low = h[idx] - l[idx]
    if high_low > 0 and body / high_low < 0.1:
        return True
    return False

def detect_engulfing(o, h, l, c, idx):
    if idx < 1:
        return None
    if c[idx] > o[idx] and c[idx-1] < o[idx-1]:
        if c[idx] >= o[idx-1] and o[idx] <= c[idx-1]:
            return 'bullish'
    if c[idx] < o[idx] and c[idx-1] > o[idx-1]:
        if o[idx] >= c[idx-1] and c[idx] <= o[idx-1]:
            return 'bearish'
    return None

def detect_morning_star(o, h, l, c, idx):
    if idx < 2:
        return False
    if c[idx-2] < o[idx-2] and c[idx] > o[idx]:
        body1 = abs(c[idx-2] - o[idx-2])
        body2 = abs(c[idx-1] - o[idx-1])
        body3 = abs(c[idx] - o[idx])
        if body2 < body1 * 0.5 and body3 > body1 * 0.5:
            if c[idx] > (o[idx-2] + c[idx-2]) / 2:
                return True
    return False

def detect_evening_star(o, h, l, c, idx):
    if idx < 2:
        return False
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
    close_series = data['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    open_series = data['Open']
    if isinstance(open_series, pd.DataFrame):
        open_series = open_series.iloc[:, 0]
    high_series = data['High']
    if isinstance(high_series, pd.DataFrame):
        high_series = high_series.iloc[:, 0]
    low_series = data['Low']
    if isinstance(low_series, pd.DataFrame):
        low_series = low_series.iloc[:, 0]
    vol_series = data['Volume']
    if isinstance(vol_series, pd.DataFrame):
        vol_series = vol_series.iloc[:, 0]

    o = open_series.values
    h = high_series.values
    l = low_series.values
    c = close_series.values
    vol = vol_series.values

    last_idx = len(c) - 1
    signals = []

    for i in range(max(0, last_idx - 5), last_idx + 1):
        if detect_hammer(o, h, l, c, i):
            signals.append({
                'date': data.index[i].strftime('%Y-%m-%d %H:%M'),
                'pattern': 'Молот (Hammer)',
                'type': 'бычий разворот',
                'price': float(c[i]),
                'strength': 'средний'
            })
        if detect_shooting_star(o, h, l, c, i):
            signals.append({
                'date': data.index[i].strftime('%Y-%m-%d %H:%M'),
                'pattern': 'Падающая звезда (Shooting Star)',
                'type': 'медвежий разворот',
                'price': float(c[i]),
                'strength': 'средний'
            })
        if detect_doji(o, h, l, c, i):
            signals.append({
                'date': data.index[i].strftime('%Y-%m-%d %H:%M'),
                'pattern': 'Доджи (Doji)',
                'type': 'неопределённость',
                'price': float(c[i]),
                'strength': 'слабый'
            })
        engulf = detect_engulfing(o, h, l, c, i)
        if engulf:
            signals.append({
                'date': data.index[i].strftime('%Y-%m-%d %H:%M'),
                'pattern': 'Поглощение (Engulfing)',
                'type': f'{engulf} разворот',
                'price': float(c[i]),
                'strength': 'сильный'
            })
        if detect_morning_star(o, h, l, c, i):
            signals.append({
                'date': data.index[i].strftime('%Y-%m-%d %H:%M'),
                'pattern': 'Утренняя звезда (Morning Star)',
                'type': 'бычий разворот',
                'price': float(c[i]),
                'strength': 'сильный'
            })
        if detect_evening_star(o, h, l, c, i):
            signals.append({
                'date': data.index[i].strftime('%Y-%m-%d %H:%M'),
                'pattern': 'Вечерняя звезда (Evening Star)',
                'type': 'медвежий разворот',
                'price': float(c[i]),
                'strength': 'сильный'
            })

    if len(data) > 20:
        current_vol = float(vol[-1])
        avg_vol = float(np.mean(vol[-20:])) if len(vol) >= 20 else current_vol
        vol_confirmation = 'высокий' if current_vol > avg_vol * 1.5 else 'нормальный' if current_vol > avg_vol * 0.8 else 'низкий'

        price = float(close_series.iloc[-1])
        ma20 = moving_average(close_series, 20).iloc[-1]
        ma50 = moving_average(close_series, 50).iloc[-1] if len(data) >= 50 else None
        price_above_ma20 = price > ma20
        price_above_ma50 = price > ma50 if ma50 is not None else None

        rsi_vals = rsi(close_series, 14)
        last_rsi = float(rsi_vals.iloc[-1])
        rsi_signal = 'перекупленность' if last_rsi > 70 else 'перепроданность' if last_rsi < 30 else 'нейтрально'

        price_slope = close_series.iloc[-5:].values
        rsi_slope = rsi_vals.iloc[-5:].values
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

        if signals:
            signals[0]['volume_confirmation'] = vol_confirmation
            signals[0]['price_above_ma20'] = price_above_ma20
            signals[0]['price_above_ma50'] = price_above_ma50 if ma50 is not None else None
            signals[0]['rsi'] = last_rsi
            signals[0]['rsi_signal'] = rsi_signal
            signals[0]['divergence'] = div_signal
        else:
            signals.append({
                'date': data.index[-1].strftime('%Y-%m-%d %H:%M'),
                'pattern': 'Нет свечного паттерна',
                'type': 'только индикаторы',
                'price': price,
                'strength': 'нет',
                'volume_confirmation': vol_confirmation,
                'price_above_ma20': price_above_ma20,
                'price_above_ma50': price_above_ma50 if ma50 is not None else None,
                'rsi': last_rsi,
                'rsi_signal': rsi_signal,
                'divergence': div_signal
            })

    return signals

# ---------- Функция текстового анализа ----------
def analyze_probabilities(prob_up, prob_down, expected_price, current_price, 
                          var_95, prob_gain_10, prob_loss_10, std_return, 
                          forecast_steps, signals, timeframe):
    analysis = []
    
    # 1. Направление
    if prob_up > 60:
        direction = "📈 **Бычий настрой** — вероятность роста значительно выше падения."
    elif prob_down > 60:
        direction = "📉 **Медвежий настрой** — вероятность падения значительно выше роста."
    else:
        direction = "⚖️ **Неопределённость** — вероятности близки к 50/50."
    analysis.append(direction)
    
    # 2. Ожидаемое изменение
    expected_change = (expected_price / current_price - 1) * 100
    if expected_change > 2:
        change_desc = f"Ожидаемый рост на {expected_change:.1f}% за {forecast_steps} {timeframe}-свечей."
    elif expected_change < -2:
        change_desc = f"Ожидаемое падение на {-expected_change:.1f}% за {forecast_steps} {timeframe}-свечей."
    else:
        change_desc = f"Ожидаемое изменение незначительно ({expected_change:.1f}%)."
    analysis.append(f"📊 {change_desc}")
    
    # 3. Волатильность
    if std_return > 0.03:
        vol_desc = "Высокая волатильность — возможны сильные колебания."
    elif std_return > 0.015:
        vol_desc = "Умеренная волатильность."
    else:
        vol_desc = "Низкая волатильность — рынок спокойный."
    analysis.append(f"🌊 {vol_desc}")
    
    # 4. VaR
    var_percent = (var_95 / current_price) * 100
    analysis.append(f"📉 **VaR (95%)**: потенциальные потери не превысят {var_percent:.1f}% с вероятностью 95%.")
    
    # 5. Сильные движения
    if prob_gain_10 > 20:
        analysis.append(f"🚀 Высокая вероятность ({prob_gain_10:.0f}%) роста >10%.")
    elif prob_loss_10 > 20:
        analysis.append(f"📉 Высокая вероятность ({prob_loss_10:.0f}%) падения >10%.")
    else:
        analysis.append("🔒 Вероятность сильных движений (>10%) невысока.")
    
    # 6. Паттерны и подтверждения
    if signals:
        latest = signals[0]
        if 'pattern' in latest and latest['pattern'] != 'Нет свечного паттерна':
            analysis.append(f"🕯️ Найден паттерн **{latest['pattern']}** ({latest['type']}).")
            confirms = []
            if latest.get('volume_confirmation') == 'высокий':
                confirms.append("высокий объём")
            if latest.get('price_above_ma20', False):
                confirms.append("цена выше MA20")
            if latest.get('rsi_signal') == 'перепроданность':
                confirms.append("RSI перепроданность (бычий сигнал)")
            elif latest.get('rsi_signal') == 'перекупленность':
                confirms.append("RSI перекупленность (медвежий сигнал)")
            if confirms:
                analysis.append(f"✅ Подтверждения: {', '.join(confirms)}.")
            else:
                analysis.append("⚠️ Подтверждений недостаточно — сигнал слабый.")
        else:
            analysis.append("ℹ️ Свечных паттернов не обнаружено. Ориентируйтесь на индикаторы.")
    
    # 7. Рекомендация
    if prob_up > 55 and prob_down < 45 and expected_change > 0:
        analysis.append("💡 **Рекомендация**: рассмотреть возможность покупки (бычий сценарий).")
    elif prob_down > 55 and prob_up < 45 and expected_change < 0:
        analysis.append("💡 **Рекомендация**: рассмотреть возможность продажи или удержания (медвежий сценарий).")
    else:
        analysis.append("💡 **Рекомендация**: рынок неопределён — лучше дождаться более чёткого сигнала.")
    
    return analysis

# ---------- Основной интерфейс ----------
with st.sidebar:
    st.header("⚙️ Настройки")
    ticker = st.text_input("Введите тикер монеты", value="BTC-USD").upper()
    
    timeframe = st.selectbox(
        "Таймфрейм (интервал свечи)",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        index=5,
        help="Для коротких интервалов может потребоваться больше исторических данных."
    )
    
    forecast_steps = st.number_input(
        "Количество шагов прогноза (свечей вперёд)",
        min_value=1,
        max_value=50,
        value=5,
        help="Сколько свечей вперёд моделировать."
    )
    
    method = st.selectbox(
        "Метод моделирования вероятностей",
        ["t-распределение (рекомендуется)", "Нормальное распределение", "Исторический бутстрап"],
        index=0
    )
    
    half_life = st.slider("Период полураспада для весов (дней)", 5, 60, 20,
                          help="Меньшее значение — больше вес последних данных.")
    
    calculate = st.button("🚀 Рассчитать вероятности и паттерны")

# ---------- Загрузка данных с интервалом ----------
def load_data_with_retry(ticker, timeframe, min_candles=100):
    # Подбираем период в зависимости от таймфрейма
    if timeframe in ['1m', '5m']:
        period = '7d'       # yfinance даёт максимум 7 дней для минутных данных
    elif timeframe in ['15m', '30m']:
        period = '30d'
    elif timeframe in ['1h', '4h']:
        period = '90d'
    else:  # '1d'
        period = '1y'
    
    # Пробуем разные периоды, если данных мало
    periods_to_try = [period, '90d', '1y']
    for p in periods_to_try:
        try:
            data = yf.download(ticker, period=p, interval=timeframe, progress=False)
            if not data.empty and 'Close' in data.columns:
                close_series = data['Close']
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
                if len(close_series) >= min_candles:
                    return data, close_series, p
        except Exception:
            continue
    return None, None, None

# ---------- Основная логика ----------
if calculate:
    try:
        with st.spinner(f"Загружаем данные ({timeframe})..."):
            data, close_series, used_period = load_data_with_retry(ticker, timeframe, min_candles=100)
            if data is None:
                st.error(f"❌ Не удалось загрузить достаточно данных для {ticker} с таймфреймом {timeframe}.")
                st.info("Попробуйте выбрать более крупный таймфрейм или другой тикер.")
                st.stop()
            
            st.info(f"ℹ️ Загружено {len(close_series)} свечей за период {used_period}.")
            current_price = float(close_series.iloc[-1])
        
        # Расчёт доходностей на интервале
        returns = np.log(close_series / close_series.shift(1)).dropna()
        if len(returns) < 10:
            st.error(f"❌ Недостаточно доходностей ({len(returns)} точек). Нужно минимум 10.")
            st.stop()
        
        # EWMA
        lambda_ = np.exp(-np.log(2) / (half_life * 24))  # адаптация для любого таймфрейма (в часах)
        # Для упрощения используем half_life как количество свечей (не дней)
        lambda_ = np.exp(-np.log(2) / half_life)  # half_life в свечах
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
            random_returns = np.random.normal(mean_return, std_return, (forecast_steps, n_simulations))
        elif method == "t-распределение (рекомендуется)":
            random_returns = t.rvs(df, loc=loc, scale=scale, size=(forecast_steps, n_simulations))
        else:
            indices = np.random.choice(len(returns), size=(forecast_steps, n_simulations), p=weights)
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
        
        # Анализ паттернов
        signals = analyze_patterns_and_signals(data)
        
        st.session_state.results = {
            'current_price': current_price,
            'expected_price': expected_price,
            'prob_up': prob_up,
            'prob_down': prob_down,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'final_prices': final_prices,
            'random_returns': random_returns,
            'data': data,
            'ticker': ticker,
            'forecast_steps': forecast_steps,
            'timeframe': timeframe,
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
    
    # Анализ вероятностей
    st.subheader("🧠 Анализ вероятностей и рекомендации")
    analysis_text = analyze_probabilities(
        results['prob_up'], results['prob_down'],
        results['expected_price'], results['current_price'],
        results['var_95'], results['prob_gain_10'],
        results['prob_loss_10'], results['std_return'],
        results['forecast_steps'],
        results['signals'],
        results['timeframe']
    )
    for line in analysis_text:
        st.write(line)
    
    st.subheader("🕯️ Обнаруженные свечные паттерны и подтверждения")
    signals = results['signals']
    if signals:
        df_signals = pd.DataFrame(signals)
        st.dataframe(
            df_signals[['date', 'pattern', 'type', 'price', 'strength']],
            use_container_width=True,
            hide_index=True
        )
        
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
    
    st.subheader("📉 Примеры возможных траекторий (20 случайных сценариев)")
    random_returns = results['random_returns']
    current_price = results['current_price']
    n_paths = 20
    indices = np.random.choice(range(random_returns.shape[1]), min(n_paths, random_returns.shape[1]), replace=False)
    fig_paths = go.Figure()
    for idx in indices:
        path_prices = current_price * np.exp(np.cumsum(random_returns[:, idx], axis=0))
        path_prices = np.insert(path_prices, 0, current_price)
        fig_paths.add_trace(go.Scatter(
            x=list(range(len(path_prices))),
            y=path_prices,
            mode='lines',
            line=dict(width=0.8, color='lightgray'),
            showlegend=False
        ))
    avg_path = current_price * np.exp(np.mean(np.cumsum(random_returns, axis=0), axis=1))
    avg_path = np.insert(avg_path, 0, current_price)
    fig_paths.add_trace(go.Scatter(
        x=list(range(len(avg_path))),
        y=avg_path,
        mode='lines',
        line=dict(color='red', width=2),
        name='Средняя траектория'
    ))
    fig_paths.update_layout(
        xaxis_title="Шаг (количество свечей)",
        yaxis_title="Цена ($)",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_paths, use_container_width=True)
    
    with st.expander("📋 Детальная статистика и параметры модели"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**📊 Параметры доходности (взвешенные):**")
            st.write(f"Средняя доходность за свечу (EWMA): {results['mean_return']*100:.3f}%")
            st.write(f"Волатильность за свечу (EWMA): {results['std_return']*100:.3f}%")
            st.write(f"Количество свечей в выборке: {len(results['data'])}")
            st.write(f"Период: {results['data'].index[0].strftime('%Y-%m-%d %H:%M')} - {results['data'].index[-1].strftime('%Y-%m-%d %H:%M')}")
            st.write(f"Использованный период загрузки: {results.get('used_period', 'не указан')}")
        with col2:
            st.write("**🎯 Доверительные интервалы (5-95%):**")
            st.write(f"Нижняя граница: ${results['percentile_5']:.2f}")
            st.write(f"Верхняя граница: ${results['percentile_95']:.2f}")
            st.write(f"Диапазон: ${results['percentile_95'] - results['percentile_5']:.2f}")
            st.write(f"VaR (95%): ${results['var_95']:.2f}")
            st.write(f"Вероятность роста >10%: {results['prob_gain_10']:.1f}%")
            st.write(f"Вероятность падения >10%: {results['prob_loss_10']:.1f}%")
        st.write(f"Метод моделирования: {results['method']}")
        st.write(f"Период полураспада (в свечах): {results['half_life']}")
    
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
        xaxis_title="Дата/время",
        yaxis_title=f"Цена ({results['ticker']})",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    if st.button("🔄 Новый расчёт"):
        st.session_state.results = None
        st.rerun()

else:
    st.info("👈 Выберите настройки и нажмите 'Рассчитать вероятности и паттерны'")
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
    with st.expander("🧠 Как работают таймфреймы и прогноз"):
        st.write("""
        - **Таймфрейм** определяет длину одной свечи (1 минута, 5 минут, 1 час, день и т.д.).
        - **Количество шагов** — сколько таких свечей вперёд моделируется.
        - Прогноз на 15 минут = таймфрейм '15m', шаг=1.
        - Прогноз на 1 час = '1h', шаг=1 (или '15m', шаг=4).
        - Для коротких таймфреймов требуется больше исторических данных — сервис автоматически подбирает период загрузки.
        """)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Возможности:**\n\n"
    "✅ Прогноз на любом таймфрейме (от 1 минуты до дня)\n"
    "✅ Детекция свечных паттернов\n"
    "✅ Подтверждения (объём, MA, RSI, дивергенция)\n"
    "✅ Текстовый анализ и рекомендации\n"
    "✅ Графики распределения и сценариев\n"
    "⚠️ Результаты не являются инвестиционной рекомендацией."
)
st.sidebar.caption("Сделано с ❤️ для криптоэнтузиастов")