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

import io
from googleapiclient.http import MediaIoBaseDownload

def download_file(file_id, filename):
    """Tải file từ Google Drive về và lưu local"""
    request = service.files().get_media(fileId=file_id)
    file_path = f"./{filename}"
    fh = io.FileIO(file_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        if status:
            print(f"Downloading {filename}: {int(status.progress() * 100)}%")
    return file_path


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

def read_excel_file(file_path):
    if file_path.endswith(".xls"):
        return pd.read_excel(file_path, engine="xlrd", skiprows=15, usecols="B:S")
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path, engine="openpyxl", skiprows=15, usecols="B:S")
    else:
        raise Exception(f"Unsupported file format: {file_path}")

# --- Load toàn bộ file Excel ---
def load_all_excels(folder_id):
    files = get_all_files_from_folder(folder_id)
    df_list = []
    for f in files:
        file_id = f["id"]
        name = f["name"]
        file_path = download_file(file_id, name)   # << bây giờ hàm này đã có
        try:
            df = read_excel_file(file_path)
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc {name}: {e}")
    if not df_list:
        raise Exception("Không load được file Excel nào!")
    return pd.concat(df_list, ignore_index=True)

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
