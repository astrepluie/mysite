import pandas as pd
from datetime import datetime
import numpy as np

def create_band(_df, _start = '2010-01-01', _end = datetime.now(), _col = 'Adj Close', _cnt = 20) :
    # 복사본 생성
    df = _df.copy()
    # Date 컬럼 인덱스 설정
    if 'Date' in df.columns :
        df.set_index('Date', inplace = True)
    # 시계열 데이터로 변경
    df.index = pd.to_datetime(df.index)
    # 기준이 되는 컬럼을 제외하고 나머지 컬럼은 모두 제거
    df = df[ [_col] ]
    # 결측치, 무한대 데이터 제거
    flag = df.isin([np.nan, np.inf, -np.inf]).any(axis = 1)
    df = df.loc[-flag, : ]
    # 이동평균선, 상단밴드, 하단밴드 생성
    df['center'] = df[_col].rolling(_cnt).mean()
    df['ub'] = df['center'] + (2 * df[_col].rolling(_cnt).std())
    df['lb'] = df['center'] - (2 * df[_col].rolling(_cnt).std())
    # 시작시간, 종료시간을 기준으로 데이터 필터링
    df = df.loc[_start : _end, : ]
    return df

def create_trade(_df) :
    df = _df.copy()
    df['trade'] = ''
    # 기준이 되는 컬럼의 이름을 변수에 저장
    col = df.columns[0]
    # 거래 내역 추가 반복문
    for idx in df.index :
        # 상단 밴드보다 기준이 되는 컬럼의 값이 크거나 같은 경우
        if df.loc[idx, col] >= df.loc[idx, 'ub'] :
            df.loc[idx, 'trade'] = ''
        # 하단 밴드보다 기준이 되는 컬럼의 값이 작거나 같은 경우
        elif df.loc[idx, col] <= df.loc[idx, 'lb'] :
            df.loc[idx, 'trade'] = 'buy'
        # 기준이 되는 컬럼의 값이 밴드 사이에 존재하는 경우
        else :
            df.loc[idx, 'trade'] = df.shift().loc[idx, 'trade']
    return df

def create_rtn(_df) :
    df = _df.copy()
    col = df.columns[0]
    df['rtn'] = 1
    for idx in df.index :
        # 매수
        if (df.shift().loc[idx, 'trade'] == '') & (df.loc[idx, 'trade'] == 'buy') :
            buy = df.loc[idx, col]
            print(f"매수일 : {idx}, 매수가 : {buy}")
        # 매도
        elif (df.shift().loc[idx, 'trade'] == 'buy') & (df.loc[idx, 'trade'] == '') :
            sell = df.loc[idx, col]
            rtn = sell/buy
            df.loc[idx, 'rtn'] = rtn
            print(f"매도일 : {idx}, 매도가 : {sell}, 수익률 : {rtn}")
    # 누적수익률
    df['acc_rtn'] = df['rtn'].cumprod()
    # 최종 누적수익률
    acc = df['acc_rtn'][-1]
    return df, acc      