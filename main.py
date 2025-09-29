import os
import pandas as pd
from datetime import datetime, timedelta
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib, ssl

# Nếu dùng OpenAI để phân tích
try:
    import openai
except ImportError:
    openai = None

# --- Cấu hình ---
TOPN = 30
HISTORY_FILE = "history.csv"
REPORT_FILE = "latest_report.txt"

# Email (lấy từ secret/env)
EMAIL_USER = os.getenv("nguyenhung52020@gmail.com")
EMAIL_PASS = os.getenv("aesv aoby rxxx aaxg")
EMAIL_TO = os.getenv("nguyenhung52020@gmail.com")

# GPT / OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Hàm lấy dữ liệu từ CafeF ---
def fetch_from_cafef():
    url = "https://liveboard.cafef.vn/Hose/Market/Total.php"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    html = resp.text

    # đọc các bảng HTML
    tables = pd.read_html(html)
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("mã" in c or "ticker" in c for c in cols) and any("giá trị" in c or "value" in c or "gtgd" in c for c in cols):
            df = t.copy()
            # tìm cột phù hợp
            ticker_col = next((c for c in df.columns if "mã" in str(c).lower() or "ticker" in str(c).lower()), None)
            value_col = next((c for c in df.columns if "giá trị" in str(c).lower() or "value" in str(c).lower() or "gtgd" in str(c).lower()), None)
            if ticker_col is None or value_col is None:
                continue
            df2 = df[[ticker_col, value_col]].rename(columns={ticker_col: "Ticker", value_col: "RawValue"})
            # chuyển RawValue sang số
            def parse_raw(x):
                if pd.isna(x):
                    return 0.0
                s = str(x)
                # loại bỏ dấu phẩy, dấu chấm nếu là phần nghìn
                s = s.replace(",", "").replace(".", "")
                s = "".join(ch for ch in s if ch.isdigit())
                try:
                    return float(s)
                except:
                    return 0.0
            df2["Value"] = df2["RawValue"].apply(parse_raw)
            return df2[["Ticker", "Value"]]
    raise RuntimeError("Không tìm được bảng GTGD từ CafeF")

# --- Cập nhật lịch sử ---
def update_history(df_today):
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
    else:
        hist = pd.DataFrame(columns=["Date","Ticker","Value"])
    # thêm ngày hôm nay
    today_str = datetime.today().strftime("%Y-%m-%d")
    recs = []
    for _, row in df_today.iterrows():
        recs.append({"Date": today_str, "Ticker": row["Ticker"], "Value": row["Value"]})
    df_new = pd.DataFrame(recs)
    hist = pd.concat([hist, df_new], ignore_index=True)
    hist.to_csv(HISTORY_FILE, index=False)
    return hist

# --- So sánh với quá khứ ---
def get_value(hist, date_str, ticker):
    df = hist[(hist["Date"] == date_str) & (hist["Ticker"] == ticker)]
    if not df.empty:
        return float(df.iloc[0]["Value"])
    return None

def make_comparison(df_top, hist):
    today = datetime.today().strftime("%Y-%m-%d")
    date_y = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    date_7 = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    date_30 = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")

    recs = []
    for _, r in df_top.iterrows():
        tkr = r["Ticker"]
        v0 = r["Value"]
        vy = get_value(hist, date_y, tkr)
        v7 = get_value(hist, date_7, tkr)
        v30 = get_value(hist, date_30, tkr)
        def pct(a, b):
            if b is None or b == 0:
                return None
            return (a - b) / b * 100.0
        recs.append({
            "Ticker": tkr,
            "ValueToday": v0,
            "ValueYesterday": vy,
            "Pct_vs_Yesterday": pct(v0, vy),
            "Value7d": v7,
            "Pct_vs_7d": pct(v0, v7),
            "Value30d": v30,
            "Pct_vs_30d": pct(v0, v30),
        })
    df_cmp = pd.DataFrame(recs)
    df_cmp = df_cmp.sort_values("ValueToday", ascending=False)
    return df_cmp

# --- Gọi GPT để phân tích ---
def analyze_by_gpt(summary_text):
    if openai is None or not OPENAI_API_KEY:
        return "GPT không được cấu hình."
    openai.api_key = OPENAI_API_KEY
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user", "content": summary_text}],
            max_tokens=500,
            temperature=0.7
        )
        return resp.choices[0].message["content"]
    except Exception as e:
        return f"Lỗi khi gọi GPT: {e}"

# --- Gửi email ---
def send_email(subject, body):
    if not (EMAIL_USER and EMAIL_PASS and EMAIL_TO):
        print("Email chưa được cấu hình.")
        return
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(body, "plain", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
    print("Email đã gửi.")

# --- Tạo báo cáo ---
def build_and_send_report():
    # 1. Lấy dữ liệu toàn HOSE
    df_all = fetch_from_cafef()
    # 2. Lấy top 30
    df_top = df_all.sort_values("Value", ascending=False).head(TOPN)
    # 3. Tính tỉ trọng
    total_market = df_all["Value"].sum()
    top_sum = df_top["Value"].sum()
    ratio = (top_sum / total_market * 100.0) if total_market > 0 else None
    # 4. Cập nhật lịch sử
    hist = update_history(df_top)
    # 5. So sánh
    cmp_df = make_comparison(df_top, hist)
    # 6. Chuẩn bị summary + prompt
    summary = f"Ngày: {datetime.today().strftime('%Y-%m-%d')}\n"
    summary += f"Top {TOPN} chiếm: {ratio:.2f}% của tổng GTGD: {total_market}\n"
    summary += "\nTop 5 hôm nay:\n"
    for i, row in df_top.head(5).iterrows():
        summary += f"{row['Ticker']}: {row['Value']}\n"
    summary += "\n-- Bảng so sánh --\n"
    summary += cmp_df[["Ticker","Pct_vs_Yesterday","Pct_vs_7d","Pct_vs_30d"]].to_string(index=False)

    analysis = analyze_by_gpt(summary)

    # 7. Kết hợp nội dung email
    subject = f"[TỰ ĐỘNG] Báo cáo dòng tiền HOSE Top{TOPN} {datetime.today().strftime('%Y-%m-%d')}"
    body = summary + "\n\nPhân tích:\n" + analysis

    # 8. Gửi email
    send_email(subject, body)

    # 9. Ghi file report ra local
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(body)
    print("Báo cáo đã lưu.")

if __name__ == "__main__":
    build_and_send_report()
