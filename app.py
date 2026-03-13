from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from functools import wraps


def get_ticker_symbol(symbol):
    clean_symbol = (symbol or "").strip().upper()
    if clean_symbol.endswith('.NS'):
        return clean_symbol
    return f"{clean_symbol}.NS"


def get_stock_metadata(symbol):
    ticker_symbol = get_ticker_symbol(symbol)
    ticker = yf.Ticker(ticker_symbol)

    info = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    price = info.get("regularMarketPrice")
    display_name = info.get("longName") or info.get("shortName") or symbol.strip().upper()
    website = info.get("website") or ""
    has_market_data = price is not None

    if not has_market_data:
        try:
            history = ticker.history(period="5d", interval="1d", auto_adjust=False)
        except Exception:
            history = pd.DataFrame()
        has_market_data = history is not None and not history.empty
        if has_market_data and (not price or pd.isna(price)) and "Close" in history.columns:
            close = pd.to_numeric(history["Close"], errors="coerce").dropna()
            if not close.empty:
                price = float(close.iloc[-1])

    is_valid = has_market_data or bool(info.get("symbol")) or bool(info.get("longName")) or bool(info.get("shortName"))

    return {
        "symbol": ticker_symbol.replace(".NS", ""),
        "ticker_symbol": ticker_symbol,
        "display_name": display_name,
        "website": website,
        "price": float(price) if price not in (None, "") and not pd.isna(price) else 0.0,
        "valid": is_valid,
    }


def search_symbols(query, max_results=8):
    query = (query or "").strip().upper()
    if not query:
        return []

    try:
        results = yf.Search(query, max_results=max_results, enable_fuzzy_query=True).quotes
    except Exception:
        return []

    preferred_quotes = []
    fallback_quotes = []
    for quote in results or []:
        quote_symbol = (quote.get("symbol") or "").upper()
        exchange = (quote.get("exchange") or "").upper()
        exchange_display = (quote.get("exchangeDisplay") or "").upper()
        name = quote.get("longname") or quote.get("shortname") or quote_symbol
        item = {
            "symbol": quote_symbol.replace(".NS", ""),
            "name": name,
        }

        if quote_symbol.endswith(".NS") or "NSE" in exchange or "NATIONAL STOCK EXCHANGE" in exchange_display:
            preferred_quotes.append(item)
        elif quote_symbol:
            fallback_quotes.append(item)

    matches = preferred_quotes or fallback_quotes
    deduped_matches = []
    seen_symbols = set()
    for match in matches:
        if match["symbol"] in seen_symbols:
            continue
        deduped_matches.append(match)
        seen_symbols.add(match["symbol"])
    return deduped_matches


def suggest_symbol(symbol):
    matches = search_symbols(symbol, max_results=8)
    if not matches:
        return None
    return matches[0]

def predict_stock(symbol):
    ticker_symbol = get_ticker_symbol(symbol)

    try:
        data = yf.download(
            ticker_symbol,
            period="3mo",
            interval="1d",
            auto_adjust=False,
            progress=False
        )
        if data is None or data.empty:
            data = yf.Ticker(ticker_symbol).history(
                period="3mo",
                interval="1d",
                auto_adjust=False
            )

        if data is None or data.empty or "Close" not in data.columns:
            return 0, "Unknown"

        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna()

        if close.empty:
            return 0, "Unknown"
        if len(close) < 2:
            return round(float(close.iloc[-1]), 2), "Unknown"

        X = np.arange(len(close)).reshape(-1, 1)
        y = close.values.astype(float)

        model = LinearRegression()
        model.fit(X, y)

        prediction = float(model.predict([[len(close)]])[0])

        returns = close.pct_change().dropna()
        if returns.empty:
            risk = "Unknown"
        else:
            volatility = float(np.std(returns))
            if volatility < 0.01:
                risk = "Low"
            elif volatility < 0.02:
                risk = "Medium"
            else:
                risk = "High"

        return round(max(prediction, 0), 2), risk

    except Exception:
        return 0, "Unknown"


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    shares = db.Column(db.Integer, nullable=False)

@app.route('/')
@login_required
def home():
    user_id = session['user_id']
    user = User.query.get(user_id)
    username = user.username if user else "User"
    stocks = Portfolio.query.filter_by(user_id=user_id).all()
    data = []
    total_portfolio_value = 0
    risk_score_map = {"Low": 1, "Medium": 2, "High": 3}
    weighted_risk_sum = 0.0
    weighted_risk_total = 0.0
    for stock in stocks:
        try:
            stock_meta = get_stock_metadata(stock.symbol)
            price = stock_meta["price"]
        except Exception:
            stock_meta = {
                "symbol": stock.symbol.upper(),
                "display_name": stock.symbol.upper(),
                "website": "",
            }
            price = 0
        prediction, risk = predict_stock(stock.symbol)
        total = round(price * stock.shares, 2)
        total_portfolio_value += total
        if risk in risk_score_map and total > 0:
            weighted_risk_sum += risk_score_map[risk] * total
            weighted_risk_total += total
        data.append({
            "id": stock.id,
            "symbol": stock_meta["symbol"],
            "display_name": stock_meta["display_name"],
            "website": stock_meta["website"],
            "shares": stock.shares,
            "price": price,
            "total": total,
            "prediction": prediction,
            "risk": risk,
        })

    if weighted_risk_total > 0:
        avg_risk_score = weighted_risk_sum / weighted_risk_total
        if avg_risk_score < 1.5:
            total_portfolio_risk = "Low"
        elif avg_risk_score < 2.5:
            total_portfolio_risk = "Medium"
        else:
            total_portfolio_risk = "High"
    else:
        total_portfolio_risk = "Unknown"

    return render_template(
        'dashboard.html',
        username=username,
        data=data,
        total_portfolio_value=total_portfolio_value,
        total_portfolio_risk=total_portfolio_risk
    )

# --- NEW ROUTE TO GET LIVE PRICES ---
@app.route('/get_live_prices')
@login_required
def get_live_prices():
    user_id = session['user_id']
    stocks = Portfolio.query.filter_by(user_id=user_id).all()
    data = []
    total_portfolio_value = 0
    
    for stock in stocks:
        try:
            price = get_stock_metadata(stock.symbol)["price"]
        except Exception:
            price = 0
            
        total = round(price * stock.shares, 2)
        total_portfolio_value += total
        data.append({
            'id': stock.id,
            'price': price,
            'total': total,
        })
        
    return jsonify({
        'stocks': data,
        'total_portfolio_value': total_portfolio_value
    })


@app.route('/search_symbols')
@login_required
def search_symbols_route():
    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return jsonify({'matches': []})
    return jsonify({'matches': search_symbols(query, max_results=6)})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        if User.query.filter_by(username=username).first():
            flash("Username already exists", "danger")
            return redirect(url_for('register'))
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        flash("Registered successfully", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            session['user_id'] = user.id
            return redirect(url_for('home'))
        flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/add', methods=['POST'])
@login_required
def add():
    selected_symbol = request.form.get('selected_symbol', '').strip().upper()

    try:
        shares = int(request.form['shares'])
        if shares <= 0:
            raise ValueError
    except (ValueError, TypeError):
        flash("Please enter a valid positive number of shares.", "danger")
        return redirect(url_for('home'))

    if not selected_symbol:
        flash("Please choose a stock from the suggestions before adding it.", "danger")
        return redirect(url_for('home'))

    try:
        stock_meta = get_stock_metadata(selected_symbol)
    except Exception:
        stock_meta = {"valid": False}

    if not stock_meta.get("valid"):
        suggestion = suggest_symbol(selected_symbol)
        if suggestion:
            flash(
                f"'{selected_symbol}' was not found. Did you mean {suggestion['symbol']} ({suggestion['name']})?",
                "danger"
            )
        else:
            flash(f"'{selected_symbol}' is not a valid stock symbol. Please check and try again.", "danger")
        return redirect(url_for('home'))

    portfolio = Portfolio(user_id=session['user_id'], symbol=stock_meta["symbol"], shares=shares)
    db.session.add(portfolio)
    db.session.commit()
    flash(f"Added {stock_meta['display_name']} ({stock_meta['symbol']}).", "success")
    return redirect(url_for('home'))

@app.route('/delete/<int:stock_id>', methods=['POST'])
@login_required
def delete(stock_id):
    stock_to_delete = Portfolio.query.get_or_404(stock_id)
    if stock_to_delete.user_id != session['user_id']:
        flash("You do not have permission to delete this stock.", "danger")
        return redirect(url_for('home'))

    db.session.delete(stock_to_delete)
    db.session.commit()
    flash("Stock has been deleted successfully.", "success")
    return redirect(url_for('home'))

if __name__ == '__main__':
    # Safe to call repeatedly; creates tables only if they do not already exist.
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000,debug=True)
