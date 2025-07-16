import requests
import pandas as pd

url = "https://openapi.its.go.kr:9443/cctvInfo"
params = {
    "apiKey": "83b405ae143b49da85d1d998563492b9",  # 여기에 본인의 키 입력
    "type": "all",
    "cctvType": "1",
    "minX": "126.8",
    "maxX": "127.89",
    "minY": "34.9",
    "maxY": "35.1",
    "getType": "json"
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
    df.to_excel("cctv_list.xlsx", index=False)
    df.to_json("cctv_list.json", orient="records", force_ascii=False)  # JSON도 함께 저장
    print("✅ CCTV 목록이 'cctv_list.xlsx' 파일로 저장되었습니다.")
else:
    print("⚠️ CCTV 데이터가 없습니다. 응답 메시지:", data.get("resultMsg", "알 수 없음"))





#83b405ae143b49da85d1d998563492b9