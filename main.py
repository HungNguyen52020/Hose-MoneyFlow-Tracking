import os
import io
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import smtplib
from email.mime.text import MIMEText

# --- Google Drive API setup ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Lấy credentials từ secret
with open("credentials.json", "w") as f:
    f.write(os.getenv("GDRIVE_CREDENTIALS"))

creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

# --- Hàm tải file từ Google Drive ---
def download_file_from_gdrive(file_id, filename):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(filename, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    print(f"Downloaded {filename}")

# --- Lấy toàn bộ file trong folder ---
def get_all_files_from_folder(folder_id):
    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        orderBy="createdTime",
        fields="files(id, name, createdTime)"
    ).execute()
    items = results.get('files', [])
    if not items:
        raise Exception("No files found in folder")
    return items  # list file

# --- Load toàn bộ file Excel ---
def load_all_excels(folder_id):
    files = get_all_files_from_folder(folder_id)
    dfs = []
    for f in files:
        file_id, file_name = f['id'], f['name']
        download_file_from_gdrive(file_id, file_name)

        try:
            df = pd.read_excel(file_name, skiprows=15, usecols="B:S")
            dfs.append(df)
            print(f"Loaded {file_name}, shape={df.shape}")
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc {file_name}: {e}")

    if not dfs:
        raise Exception("Không load được file Excel nào!")
    return pd.concat(dfs, ignore_index=True)

# --- Email ---
def send_email_report(report_text):
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")
    EMAIL_TO = os.getenv("EMAIL_TO")

    msg = MIMEText(report_text, "plain", "utf-8")
    msg['Subject'] = "Daily Report"
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_TO

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
    print("Email sent successfully!")

# --- Main ---
if __name__ == "__main__":
    FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")

    # Load toàn bộ file Excel
    df = load_all_excels(FOLDER_ID)

    # 🚧 Hùng sẽ tự viết logic tính toán ở đây
    report = f"Tổng số dòng dữ liệu sau khi gộp: {len(df)}"

    # Gửi mail
    send_email_report(report)
