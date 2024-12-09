
from collections import defaultdict
import numpy as np
import pandas as pd
from filter_graph_scale_live.data_processing import scale_list_dict, scale_dict_improved, clean_dict
import streamlit as st
from filter_graph_scale_live.filters import apply_filters, search_pl_filter

def update_pler_data(record, pler_data):
    pler_id = record["PLER ID"]
    tnsw = record.get("tnsw")
    tnusw = record.get("tnusw")
    pler_gu = record["성인 구분"]
    pler_language = record["언어"]
    pler_name = record["이름"]
    round_num = record["회차"]
    class_type = record["수업 구분"]

    composite_key = (pler_id, pler_gu, pler_language)
    
    if composite_key not in pler_data:
        pler_data[composite_key] = {
            "tnsw": [],
            "tnusw": [],
            "rounds_tnsw": {},
            "rounds_tnusw": {},
            "pler_gu": pler_gu,
            "pler_language": pler_language,
            "pler_name": pler_name,
            "class_type": class_type
        }
    
    if tnsw is not None:
        pler_data[composite_key]["tnsw"].append(tnsw)
        pler_data[composite_key]["rounds_tnsw"][round_num] = tnsw
    if tnusw is not None:
        pler_data[composite_key]["tnusw"].append(tnusw)
        pler_data[composite_key]["rounds_tnusw"][round_num] = tnusw

def calculate_percentiles(pler_averages, top=True, percent=10):
    pler_averages.sort(key=lambda x: x[4], reverse=top)  # TNSW 평균 기준 정렬
    count = max(1, len(pler_averages) * percent // 100)
    return pler_averages[:count]

def safe_none(data):
    """
    데이터를 JSON으로 변환 가능하도록 NaN을 None으로 변환.
    """
    if isinstance(data, float) and np.isnan(data):
        return None
    if isinstance(data, dict):
        return {k: safe_none(v) for k, v in data.items()}
    if isinstance(data, list):
        return [safe_none(v) for v in data]
    return data

# 아이디 검색한 PLer
def search_fetch_data_and_process(cursor, search_pl_id, search_pl_gu, search_pl_lang):
    select_search_pler, params_s = search_pl_filter(search_pl_id, search_pl_gu, search_pl_lang)
    cursor.execute(select_search_pler, params_s)
    search_pl_result = cursor.fetchall()
    
    pl_columns = ['name', 'rounds_tnsw', 'rounds_tnusw', 'class_type', 'date']
    pl_df = pd.DataFrame(search_pl_result, columns=pl_columns)
    
    return pl_df

def search_by_round(pler_data):
    """
    PLER 데이터를 기반으로 round별 tnsw와 tnusw 데이터를 반환.
    """
    # 초기화
    tnsw_list = []
    tnusw_list = []
    ft_tnsw = 0
    ft_tnusw = 0
    
    # 데이터 처리
    if pler_data is not None:
        for index, row in pler_data.iterrows():
            if row['class_type'] == 'FT':
                ft_tnsw = row['rounds_tnsw']
                ft_tnusw = row['rounds_tnusw']
                if index == 0:
                    # 첫 번째 회차를 뛰어넘고 None 추가
                    continue
            else:
                tnsw_list.append(row['rounds_tnsw'])
                tnusw_list.append(row['rounds_tnusw'])
    
    return tnsw_list, tnusw_list, ft_tnsw, ft_tnusw

# 조건 검색
def fetch_data_and_process(cursor, TNSW_value, TNUSW_value, grade, past_learn, current_learn, week_n, game_exp, world, limit_round, search_result, gender, add_name):
    select_ft_pler, params = apply_filters(TNSW_value, TNUSW_value, grade, past_learn, current_learn, week_n, game_exp, world, gender)
    cursor.execute(select_ft_pler, params)
    ft_result = cursor.fetchall()
    
    return process_data(cursor, ft_result, TNSW_value, TNUSW_value, limit_round, search_result, add_name)

# 회차마다의 평균 계산
def calculate_round_averages(percentile_pler, pler_data, round_num_val):
    adjusted_round_num_val = round_num_val + 1
    
    rounds_tnsw = {round_num: [] for round_num in range(1, adjusted_round_num_val)}
    rounds_tnusw = {round_num: [] for round_num in range(1, adjusted_round_num_val)}
    
    for pler in percentile_pler:
        composite_key = (pler[0], pler[1], pler[2])  # PLER ID, 성인 구분, 언어
        pler_data_rounds = pler_data[composite_key]
        
        class_type = pler_data_rounds['class_type']
        
        # 'FT'일 때만 첫 번째 회차를 건너뛰기
        for round_num in range(1, adjusted_round_num_val):
            if class_type == 'FT' and round_num == 1:
                continue  # 'FT'이고 첫 회차라면 건너뛴다
            
            if class_type == 'PG':
                # 'PG'일 경우 추가 로직 처리
                if round_num == 1:
                    rounds_tnsw[round_num].append(None)
                    rounds_tnusw[round_num].append(None)
                    continue
                else:
                # 이후 회차는 -1된 회차의 데이터 사용
                    if round_num-1 in pler_data_rounds["rounds_tnsw"]:
                        rounds_tnsw[round_num].append(pler_data_rounds["rounds_tnsw"][round_num-1])
                    if round_num-1 in pler_data_rounds["rounds_tnusw"]:
                        rounds_tnusw[round_num].append(pler_data_rounds["rounds_tnusw"][round_num-1])
                continue
            
            if round_num in pler_data_rounds["rounds_tnsw"]:
                rounds_tnsw[round_num].append(pler_data_rounds["rounds_tnsw"][round_num])
            if round_num in pler_data_rounds["rounds_tnusw"]:
                rounds_tnusw[round_num].append(pler_data_rounds["rounds_tnusw"][round_num])
    
    # 수정된 adjusted_round_num_val을 사용하여 평균 계산
    avg_tnsw_by_round = [
        np.nanmean([val for val in rounds_tnsw[round_num] if val is not None]) if rounds_tnsw[round_num] and any(val is not None for val in rounds_tnsw[round_num]) else None for round_num in range(1, adjusted_round_num_val)
    ]
    avg_tnusw_by_round = [
        np.nanmean([val for val in rounds_tnusw[round_num] if val is not None]) if rounds_tnusw[round_num] and any(val is not None for val in rounds_tnusw[round_num]) else None for round_num in range(1, adjusted_round_num_val)
    ]
    
    avg_tnsw_by_round = safe_none(avg_tnsw_by_round)
    avg_tnusw_by_round = safe_none(avg_tnusw_by_round)
    
    return avg_tnsw_by_round, avg_tnusw_by_round

def process_data(cursor, ft_result, TNSW_value, TNUSW_value, limit_round, search_result, add_name):      
    columns = []
    if TNSW_value == 0 and TNUSW_value == 0:
        columns = ['pl_id', 'pl_gu', 'pl_language', 'name', 'ci_id', 'class_type', 'class_date']
    else:
        columns = ['tnsw', 'tnusw', 'pl_id', 'pl_gu', 'pl_language', 'name', 'ci_id', 'class_type', 'class_date']
    df = pd.DataFrame(ft_result, columns=columns)
    
    # 각 PLER의 TNSW와 TNUSW 값을 조회하여 회차별로 저장
    rounds_tnsw = {}
    rounds_tnusw = {}
    result_data = []
    live_result_data = []

    select_tnsw_tnusw = '''
        SELECT TNSW, TNUSW, ci.id, class_type
        FROM "PLEPS" p, "CLASS_INFO" ci
        WHERE ci.id = p.class_id 
        AND ci.pl_id = %s
        AND ci.pl_gu = %s
        AND ci.pl_language = %s
        ORDER BY date ASC;
    '''
    i = 0

    count_pler = '''
        SELECT COUNT(p2.id)
        FROM "PLER" p
        JOIN "CLASS_INFO" ci ON ci.pl_id = p.pler_id
                            AND ci.pl_gu = p.gu
                            AND ci.pl_language = p."language"
        join "PLEPS" p2 on ci.id = p2.class_id 
        WHERE p.pler_id = %s
        and p.gu = %s
        and p."language" = %s
        GROUP BY p.name, p.pler_id
    '''

    ft_tnsw_array = np.zeros((1))
    ft_tnusw_array = np.zeros((1))

    tnsw_array = []
    tnusw_array = []

    # 회차 선택에 따라 숫자 변경
    if limit_round == "8회차":
        round_num_val = 9
    elif limit_round == "16회차":
        round_num_val = 17
    elif limit_round == "24회차":
        round_num_val = 25

    # 각 PLER별로 TNSW와 TNUSW 값을 조회하여 회차별로 저장
    for _, row in df.iterrows():
        cursor.execute(select_tnsw_tnusw, (row['pl_id'], row['pl_gu'], row['pl_language']))
        tnsw_tnusw_results = cursor.fetchall()

        cursor.execute(count_pler, (row['pl_id'], row['pl_gu'], row['pl_language']))
        count_round = cursor.fetchone()
        
        count_round_val = count_round[0] if count_round else 0
        
        # FT + 8회차 이상 데이터가 있는 경우에만 처리
        if count_round_val >= round_num_val:
            if tnsw_tnusw_results:
                # FT일 경우 ft 배열에 따로 저장
                if row["class_type"] == 'FT':
                    # 기존 배열에서 0을 제거
                    ft_tnsw_array = ft_tnsw_array[ft_tnsw_array != 0]
                    ft_tnusw_array = ft_tnusw_array[ft_tnusw_array != 0]

                    ft_tnsw_array = np.append(ft_tnsw_array, tnsw_tnusw_results[0][0])
                    ft_tnusw_array = np.append(ft_tnusw_array, tnsw_tnusw_results[0][1])

                    # PG 데이터 분리 (첫 번째 값을 제외한 나머지 데이터)
                    tnsw_array = np.array([x[0] for x in tnsw_tnusw_results[1:]])  # 첫 번째 값 생략
                    tnusw_array = np.array([x[1] for x in tnsw_tnusw_results[1:]])
                else:
                    # PG 데이터 분리
                    tnsw_array = np.array([x[0] for x in tnsw_tnusw_results])
                    tnusw_array = np.array([x[1] for x in tnsw_tnusw_results])

            i += 1  # FT + 8회차 이상인 PLER만 카운트
            j = 0

            # 회차별 TNSW, TNUSW 저장
            for round_num, (tnsw, tnusw, ci_id, class_type) in enumerate(tnsw_tnusw_results, start=1):
                # 각 row의 class_type에 따라 current_round_num 설정
                current_round_num = round_num_val if row["class_type"] == 'FT' else round_num_val - 1

                if round_num > current_round_num:  # 최대 current_round_num까지만 표시
                    break

                if round_num not in rounds_tnsw:
                    rounds_tnsw[round_num] = []
                    rounds_tnusw[round_num] = []

                # 학생 정보와 함께 회차 정보를 리스트에 추가
                result_data.append({
                    'PLER ID': row['pl_id'],
                    '성인 구분': row['pl_gu'],
                    '이름': row['name'],
                    '언어': row['pl_language'],
                    'tnsw': tnsw,
                    'tnusw': tnusw,
                    '회차': round_num,
                    '수업 구분': class_type
                })
                live_result_data.append({
                    'PLER ID': row['pl_id'],
                    '성인 구분': row['pl_gu'],
                    '이름': row['name'],
                    '언어': row['pl_language'],
                    'tnsw': tnsw,
                    'tnusw': tnusw,
                    '회차': round_num,
                    '수업 구분': class_type
                })

                if j < len(tnsw_array):
                    rounds_tnsw[round_num].append(tnsw_array[j])
                    rounds_tnusw[round_num].append(tnusw_array[j])
                    j += 1
                    
    # 라이브 버전
    live_result_df = pd.DataFrame(live_result_data)
    
    # 스케일 버전
    result_data = scale_list_dict(result_data, ["tnsw", "tnusw"])
    result_df = pd.DataFrame(result_data)
    
    # 로직 끝난 후 마지막 회차 제거
    removed_value = rounds_tnsw.pop(round_num_val, None)
    removed_value = rounds_tnusw.pop(round_num_val, None)
    
    ## 각 회차 데이터 정제
    cleaned_rounds_tnsw = clean_dict(rounds_tnsw)
    cleaned_rounds_tnusw = clean_dict(rounds_tnusw)
    
    ## 정제된 데이터로 스케일링 적용
    scaled_rounds_tnsw = scale_dict_improved(cleaned_rounds_tnsw)
    scaled_rounds_tnusw = scale_dict_improved(cleaned_rounds_tnusw)
    
    st.subheader(f"{i}명의 평균 발화량")

    # 각 회차별 평균 계산에서 첫 번째 막대는 검색한 PLER 데이터만을 표시하고, 나머지 회차는 `1회차`부터 시작
    search_pler_tnsw = [{"value": TNSW_value}]
    search_pler_tnusw = [{"value": TNUSW_value}]
    
    live_search_pler_tnsw = [{"value": TNSW_value}]  # 검색한 PLER 데이터만 파란색, "itemStyle": {"color": "#00BFFF"}
    live_search_pler_tnusw = [{"value": TNUSW_value}]  # 검색한 PLER 데이터만 주황색, "itemStyle": {"color": "#FFA500"}

    # 각 회차별 평균 계산 (해당 회차에 데이터가 있는 학생들로만 평균 계산)
    avg_tnsw_by_round = [np.mean(scores) if scores else None for round_num, scores in sorted(scaled_rounds_tnsw.items())]
    avg_tnusw_by_round = [np.mean(scores) if scores else None for round_num, scores in sorted(scaled_rounds_tnusw.items())]
    
    # 라이브 버전
    live_avg_tnsw_by_round = [np.mean(scores) if scores else None for round_num, scores in sorted(rounds_tnsw.items())]
    live_avg_tnusw_by_round = [np.mean(scores) if scores else None for round_num, scores in sorted(rounds_tnusw.items())]

    round_labels = []
    search_name = None
    
    if search_result is not None:
        search_name = search_result.loc[0, 'name']

    if (not add_name and search_name) or (add_name and search_name):  # add_name이 None이거나 빈 문자열일 때
        round_labels = [f"{search_name}의 결과", "FT 평균"] + [f"{round_num}회차" for round_num in range(1, round_num_val)]
    elif add_name and not search_name:  # search_name이 None이거나 비어 있을 때
        round_labels = [f"{add_name}의 결과", "FT 평균"] + [f"{round_num}회차" for round_num in range(1, round_num_val)]
    elif not add_name and not search_name:  # 둘 다 None이거나 비어 있을 때
        round_labels = ["검색 PLer의 결과", "FT 평균"] + [f"{round_num}회차" for round_num in range(1, round_num_val)]

    ft_tnsw = round(np.mean(ft_tnsw_array)) if ft_tnsw_array.size > 0 else None
    ft_tnusw = round(np.mean(ft_tnusw_array)) if ft_tnusw_array.size > 0 else None
    search_pler_tnsw.append({"value": ft_tnsw})  # FT 데이터, "itemStyle": {"color": "#87CEEB"}
    search_pler_tnusw.append({"value": ft_tnusw})  # FT 데이터, "itemStyle": {"color": "#FFD700"}
    live_search_pler_tnsw.append({"value": ft_tnsw})  # FT 데이터, "itemStyle": {"color": "#87CEEB"}
    live_search_pler_tnusw.append({"value": ft_tnusw})  # FT 데이터, "itemStyle": {"color": "#FFD700"}

    # 첫 번째 회차 이후의 평균 데이터만 추가
    search_pler_tnsw += avg_tnsw_by_round
    search_pler_tnusw += avg_tnusw_by_round
    live_search_pler_tnsw += live_avg_tnsw_by_round
    live_search_pler_tnusw += live_avg_tnusw_by_round

    # 평균이 높은 순서로 PLER 선택
    live_pler_averages = []
    
    # PLER 데이터 초기화
    pler_data = defaultdict(lambda: {"tnsw": [], "tnusw": []})
    live_pler_data = defaultdict(lambda: {"tnsw": [], "tnusw": []})

    # 데이터 업데이트
    for record in result_data:
        update_pler_data(record, pler_data)

    for record in live_result_data:
        update_pler_data(record, live_pler_data)

    # PLER별 평균 계산
    pler_averages = [
        (pler_id, data["pler_gu"], data["pler_language"], data["pler_name"],
        np.mean(data["tnsw"]) if data["tnsw"] else None,
        np.mean(data["tnusw"]) if data["tnusw"] else None,
        data["class_type"])
        for (pler_id, pler_gu, pler_language), data in pler_data.items()
    ]
    
    live_pler_averages = [
        (pler_id, data["pler_gu"], data["pler_language"], data["pler_name"],
        np.mean(data["tnsw"]) if data["tnsw"] else None,
        np.mean(data["tnusw"]) if data["tnusw"] else None,
        data["class_type"])
        for (pler_id, pler_gu, pler_language), data in live_pler_data.items()
    ]

    # 상위/하위 10% 계산
    top_10_percent_pler = calculate_percentiles(pler_averages, top=True)
    bottom_10_percent_pler = calculate_percentiles(pler_averages, top=False)
    live_top_10_percent_pler = calculate_percentiles(live_pler_averages, top=True)
    live_bottom_10_percent_pler = calculate_percentiles(live_pler_averages, top=False)

    # 회차별 평균 계산
    avg_top_tnsw_by_round, avg_top_tnusw_by_round = calculate_round_averages(top_10_percent_pler, pler_data, round_num_val)
    avg_bottom_tnsw_by_round, avg_bottom_tnusw_by_round = calculate_round_averages(bottom_10_percent_pler, pler_data, round_num_val)
    live_avg_top_tnsw_by_round, live_avg_top_tnusw_by_round = calculate_round_averages(live_top_10_percent_pler, live_pler_data, round_num_val)
    live_avg_bottom_tnsw_by_round, live_avg_bottom_tnusw_by_round = calculate_round_averages(live_bottom_10_percent_pler, live_pler_data, round_num_val)
    
    search_tnsw_by_round = []
    search_tnusw_round = []
    
    # 검색한 플러 회차 생성
    if search_result is not None:
        search_tnsw_by_round, search_tnusw_round, ft_tnsw, ft_tnusw = search_by_round(search_result)
        
    # 데이터프레임 생성
    top10pler = pd.DataFrame(top_10_percent_pler, columns=["PLER ID", "성인 구분", "언어", "이름", "평균 TNSW", "평균 TNUSW", "FT 구분"])
    btm10pler = pd.DataFrame(bottom_10_percent_pler, columns=["PLER ID", "성인 구분", "언어", "이름", "평균 TNSW", "평균 TNUSW", "FT 구분"])
    live_top10pler = pd.DataFrame(live_top_10_percent_pler, columns=["PLER ID", "성인 구분", "언어", "이름", "평균 TNSW", "평균 TNUSW", "FT 구분"])
    live_btm10pler = pd.DataFrame(live_bottom_10_percent_pler, columns=["PLER ID", "성인 구분", "언어", "이름", "평균 TNSW", "평균 TNUSW", "FT 구분"])

    return round_labels, search_pler_tnsw, search_pler_tnusw, avg_top_tnsw_by_round, avg_top_tnusw_by_round, avg_bottom_tnsw_by_round, avg_bottom_tnusw_by_round, live_search_pler_tnsw, live_search_pler_tnusw, live_avg_top_tnsw_by_round, live_avg_top_tnusw_by_round, live_avg_bottom_tnsw_by_round, live_avg_bottom_tnusw_by_round, result_df, live_result_df, top10pler, btm10pler, live_top10pler, live_btm10pler, search_tnsw_by_round, search_tnusw_round, search_result, ft_tnsw, ft_tnusw