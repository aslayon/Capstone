import requests
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv

# .env 로드 (.env는 프로젝트 루트: c:\GIT\Capstone-1\.env)
load_dotenv(find_dotenv(), override=False)
url = "https://openapi.its.go.kr:9443/cctvInfo"
params = {
    "apiKey": os.getenv("ITS_API_KEY"),  # 여기에 본인의 키 입력
    "type": "all",
    "cctvType": "4",
    "minX": "124.0",      # 서쪽 (인천·백령도 근처)
    "maxX": "132.0",      # 동쪽 (울릉도 근처)
    "minY": "33.0",       # 남쪽 (제주도 포함)
    "maxY": "39.5",       # 북쪽 (강원도 북부까지)
    "getType": "json",
}

response = requests.get(url, params=params)
data = response.json()

if "response" in data and "data" in data["response"]:
    cctv_list = data["response"]["data"]

    # DataFrame으로 변환
    df = pd.DataFrame(cctv_list)

    # 필요한 컬럼만 선택 (원하면 전체 저장도 가능)
    df = df[["cctvname", "cctvurl", "coordx", "coordy", "cctvtype", "cctvformat"]]

    # 엑셀로 저장
    df.to_excel("data\cctv_list_4.xlsx", index=False)
    df.to_json(
        "data\cctv_list_4.json", orient="records", force_ascii=False
    )  # JSON도 함께 저장
    print("✅ CCTV 목록이 'cctv_list.xlsx' 파일로 저장되었습니다.")
else:
    print("Status Code:", response.status_code)
    print("Final URL:", response.url)
    print("Response JSON:", data)

    print("⚠️ CCTV 데이터가 없습니다. 응답 메시지:", data.get("resultMsg", "알 수 없음"))
