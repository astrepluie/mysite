import pandas as pd
from datetime import datetime
import os
from glob import glob

# 기준년월 컬럼과 월별수익률 컬럼을 생성하는 함수
def create_1m_rtn(_df, _ticker, _start = '2010-01-01', _end = datetime.now(), _col = 'Adj Close') :
    # 데이터프레임 복사본 생성
    df = _df.copy()
    # Date 컬럼을 인덱스로 변환
    if 'Date' in df.columns :
        df.set_index('Date', inplace = True)
    # 인덱스를 시계열 변환
    df.index = pd.to_datetime(df.index)
    # 투자의 시작 시간과 종료 시간으로 인덱스 필터링, 기준이 되는 컬럼으로 컬럼 필터링
    df = df.loc[_start : _end, [_col] ]
    # 기준연월 생성해서 index에서 연-월 추출하여 대입
    df['STD-YM'] = df.index.strftime('%Y-%m')
    # 월별 수익률 컬럼 생성
    df['1m_rtn'] = 0
    # ticker를 컬럼에 대입
    df['CODE'] = _ticker
    # 기준연월의 유니크값을 생성
    ym_list = df['STD-YM'].unique()
    return df, ym_list

# 데이터를 로드하고 월별 수익률을 계산하여 새로운 데이터프레임에 추가하는 함수
def data_load(_path = './data', _ext = 'csv', _start = '2010-01-01', _end = datetime.now(), _col = 'Adj Close') :
    # 파일 목록 가져오기
    files = glob(f'{_path}/*.{_ext}')
    # 빈 데이터프레임 생성
    stock_df = pd.DataFrame()
    month_last_df = pd.DataFrame()
    # files에 있는 file을 로드, 합친다
    for file in files :
        # 경로 파일명을 나눠준다
        folder, name = os.path.split(file)
        # 파일명에서 이름과 확장자로 나눠준다
        head, tail = os.path.splitext(name)
        # 데이터프레임 로드
        df = pd.read_csv(file)
        # create_1m_rtn 함수 호출
        price_df, ym_list = create_1m_rtn(df, head, _start, _end, _col)
        # price_df 를 stock_df에 단순 행결합
        stock_df = pd.concat([stock_df, price_df], axis = 0)
        # 두번째 반복문 생성
        for ym in ym_list : 
            # ym : 기준연월
            # 월초가격(매수가)
            buy = price_df.loc[ym, : ].iloc[0, 0]
            # 월말가격(매도가)
            sell = price_df.loc[ym, : ].iloc[-1, 0]
            # 수익률
            rtn = sell/buy
            # 수익률 대입
            price_df.loc[ym, '1m_rtn'] = rtn
            #월말 데이터를 month_last_df 단순 행결합
            last_df = price_df.loc[ym, ['CODE', '1m_rtn']].tail(1)
            month_last_df = pd.concat([month_last_df, last_df])
    return stock_df, month_last_df

# 구매 포지션을 잡아주는 함수 생성
def create_position(_df, _pct = 0.4) :
    # 복사본 생성
    month_rtn_df = _df.copy()
    # _pct 값이 1보다 크거나 같은 숫자라면 100으로 나눠준다
    if _pct >= 1 :
        _pct /= 100
    # 인덱스 리셋
    month_rtn_df.reset_index(inplace = True)
    # 테이블 재구조화
    month_rtn_df = month_rtn_df.pivot_table(
        index = 'Date',
        columns = 'CODE',
        values = '1m_rtn'
    )
    # month_rtn_df의 데이터들을 랭크화(열의 값들을 이용)
    month_rtn_df = month_rtn_df.rank(axis = 1, ascending = False, pct = True)
    # 상위 _pct 종목을 선택
    month_rtn_df = month_rtn_df.where(month_rtn_df <= _pct, 0)
    # 0이 아닌 데이터들은 1로 변환 -> 포지션 준비
    month_rtn_df[month_rtn_df != 0] = 1
    # stock_df의 code의 unique()를 변수에 저장
    stock_codes = list(month_rtn_df.columns)
    # 해당일자에 구매하려는 종목들을 딕셔너리로 생성
    sig_dict = dict()
    for idx in month_rtn_df.index :
        flag_col = (month_rtn_df.loc[idx, : ] == 1)
        ticker_list = list(month_rtn_df.loc[idx, flag_col].index)
        sig_dict[idx] = ticker_list
    
    return sig_dict, stock_codes 

# 구매 내역을 추가하는 함수
def create_trade_book(_df, _codes, _sig_dict) :
    # 복사본 생성
    df = _df.copy()
    # df 재구조화
    df = df.reset_index().pivot_table(
        index = 'Date',
        columns = 'CODE',
        values = df.columns[0]
    )
    
    for code in _codes :
        df[f"p_{code}"] = ""
        df[f"r_{code}"] = ""
    
    # sig_dict를 이용해서 구매 전 준비내역을 추가
    for date, codes in _sig_dict.items() :
        for code in codes :
            df.loc[date, f'p_{code}'] = f"ready_{code}"
        
    return df

# 보유 내역을 추가하는 함수 생성
def create_trading(_df, _codes) :
    buy_phase = False
    # 복사본 생성
    df = _df.copy()
    std_ym = ''
    # _codes(종목 리스트)를 이용하여 종목별로 순회하는 반복문 생성
    for code in _codes :
        # 인덱스를 기준으로 반복문 생성
        for idx in df.index :
            # 특정 종목의 포지션을 잡는다(전 행의 p_code 컬럼이 ready_code이고 현재 행의 p_code 컬럼이 '')
            if (df.loc[idx, f'p_{code}'] == '') & (df.shift().loc[idx, f'p_{code}'] == f'ready_{code}') :
                std_ym = idx.strftime('%Y-%m')
                buy_phase = True
            # 구매 조건 : 현재 p_code가 ""이고 index 연월과 std_ym과 같고 buy_phase가 True
            if (df.loc[idx, f'p_{code}'] == '') & (std_ym == idx.strftime('%Y-%m')) & (buy_phase) :
                df.loc[idx, f'p_{code}'] = f'buy_{code}'
            # buy_phase, std_ym 초기화
            if df.loc[idx, f'p_{code}'] == '' :
                buy_phase = False
                std_ym = ''
    return df

# 수익률 계산 함수 생성
def multi_return(_df, _codes) :
    # 복사본 생성
    df = _df.copy()
    rtn = 1
    # 매수가, 매도가 dict 형태로 구성
    buy_dict = dict()
    sell_dict = dict()
    # index 기준으로 반복문 생성 -> 날짜별 매수, 매도 확인
    for idx in df.index :
        for code in _codes :
            # 매수의 조건 : 2행 전의 p_code가 공백, 전 행의 p_code가 ready_code, 현재 행의 p_code가 buy_code
            if (df.shift(2).loc[idx, f'p_{code}'] == '') & (df.shift().loc[idx, f'p_{code}'] == f'ready_{code}') & (df.loc[idx, f'p_{code}'] == f'buy_{code}') :
                # 매수가
                buy_dict[code] = df.loc[idx, code]
                print(f"매수일 : {idx}, 매수 종목 : {code}, 매수가 : {df.loc[idx, code]}")
            # 매도의 조건 : 전 행의 p_code가 buy_code, 현재 행의 p_code가 ''
            elif (df.shift().loc[idx, f'p_{code}'] == f'buy_{code}') & (df.loc[idx, f'p_{code}'] == '') :
                # 매도가
                sell_dict[code] = df.loc[idx, code]
                # 수익률 계산
                rtn = sell_dict[code]/buy_dict[code]
                df.loc[idx, f'r_{code}'] = rtn
                print(f"매도일 : {idx}, 매도종목 : {code}, 매도가 : {sell_dict[code]}, 수익률 : {rtn}")
            # buy_dict, sell_dict의 code 안에 매수가 매도가 초기화
            if df.loc[idx, f'p_{code}'] == "" :
                buy_dict[code] = 0
                sell_dict[code] = 0
    return df

# 누적수익률 계산
def multi_arr_rtn(_df, _codes) :
    # 복사본 생성
    df = _df.copy()
    acc_rtn = 1
    # 인덱스를 기준으로 반복문 생성
    for idx in df.index :
        count = 0
        rtn = 0
        for code in _codes :
            # 수익률이 존재하는가?
            # if문 뒤에 bool 형태가 아닌 데이터를 넣으면 강제로 bool로 변환
            if df.loc[idx, f'r_{code}'] :
                # 존재하는 경우
                count += 1
                rtn += df.loc[idx, f'r_{code}']
        if (rtn != 0) & (count != 0) :
            acc_rtn *= rtn/count
            print(f"누적 - 매도일 : {idx}, 매도 종목수 : {count}, 수익률 : {round(rtn/count, 2)}")
        df.loc[idx, 'acc_rtn'] = acc_rtn
    return df, acc_rtn

    
